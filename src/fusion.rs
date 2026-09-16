use std::{
    cmp::Ordering,
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, mpsc::sync_channel},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array,
        Int32Array, Int64Array, ListArray, StringArray,
    },
    datatypes::{DataType, Field, Schema},
    ipc::writer::FileWriter,
    record_batch::RecordBatch,
};
use base64::{Engine, engine::general_purpose::STANDARD};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;

const FRONT_CAMERA: i64 = 1;
const TOP_LIDAR: i64 = 1;
const MAX_RENDER_POINTS: usize = 18_000;

pub struct DemoSummary {
    pub html: PathBuf,
    pub arrow: PathBuf,
    pub timestamp_micros: i64,
    pub points: usize,
    pub boxes: usize,
    pub boxes_with_depth: usize,
}

pub struct CameraFrame {
    pub timestamp: i64,
    pub jpeg: Vec<u8>,
}

pub struct LidarFrame {
    pub timestamp: i64,
    pub values: Vec<f32>,
    pub shape: [usize; 3],
}

pub struct Calibration {
    pub extrinsic: [f64; 16],
    pub inclination_min: f64,
    pub inclination_max: f64,
    pub inclinations: Vec<f64>,
}

#[derive(Clone)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub range: f32,
    pub intensity: f32,
    pub elongation: f32,
    pub camera: i8,
    pub u: f32,
    pub v: f32,
}

pub struct PointCloud {
    pub batch: RecordBatch,
    pub points: Vec<Point>,
}

#[derive(Clone, Serialize)]
pub struct Detection {
    pub id: String,
    pub kind: &'static str,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub depth_m: Option<f32>,
    pub depth_points: usize,
    pub score: Option<f32>,
}

pub struct PerceptionSummary {
    pub frames: usize,
    pub points: usize,
    pub detections: usize,
    pub depth_matched: usize,
    pub elapsed: Duration,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

pub struct PerceptionConfig<'a> {
    pub frame_limit: usize,
    pub queue_size: usize,
    pub onnx_model: Option<&'a Path>,
    pub confidence: f32,
    pub nms: f32,
    pub fusion_work_ms: u64,
}

struct CameraOutput {
    timestamp: i64,
    image_width: u32,
    image_height: u32,
    image_checksum: u32,
    detections: Vec<Detection>,
    detection_batch: RecordBatch,
    ground_truth: Vec<Detection>,
    inference_ns: u64,
    stage_ns: u64,
    started: Instant,
    enqueued: Instant,
}

struct LidarOutput {
    timestamp: i64,
    cloud: PointCloud,
    stage_ns: u64,
    started: Instant,
    enqueued: Instant,
}

#[derive(Serialize)]
struct RenderPoint(f32, f32, f32, f32, f32);

#[derive(Serialize)]
struct DemoPayload<'a> {
    segment: &'a str,
    split: &'a str,
    frame_index: usize,
    timestamp_micros: i64,
    total_points: usize,
    render_points: Vec<RenderPoint>,
    boxes: Vec<Detection>,
    processing_ms: f64,
    arrow_file: String,
}

pub fn run_segment_pipeline(
    root: &Path,
    split: &str,
    segment: &str,
    config: PerceptionConfig<'_>,
    output: &Path,
) -> Result<PerceptionSummary> {
    let PerceptionConfig {
        frame_limit,
        queue_size,
        onnx_model,
        confidence,
        nms,
        fusion_work_ms,
    } = config;
    let run_started = Instant::now();
    let (camera_tx, camera_rx) = sync_channel(queue_size);
    let (lidar_tx, lidar_rx) = sync_channel(queue_size);

    let camera_path = component_path(root, split, "camera_image", segment);
    let camera_boxes_path = component_path(root, split, "camera_box", segment);
    let model_path = onnx_model.map(Path::to_path_buf);
    let camera_worker = thread::Builder::new()
        .name("camera-decode".to_owned())
        .spawn(move || -> Result<u64> {
            let setup_started = Instant::now();
            let mut detector = model_path
                .as_deref()
                .map(|path| crate::detector::YoloxDetector::load(path, confidence, nms))
                .transpose()?;
            let camera_boxes = read_all_boxes(&camera_boxes_path)?;
            let setup_ns = elapsed_ns(setup_started);
            visit_camera_frames(&camera_path, frame_limit, |frame| {
                let started = Instant::now();
                let decoded = image::load_from_memory(&frame.jpeg)
                    .context("failed to decode a Waymo camera JPEG")?;
                let rgb = decoded.to_rgb8();
                let mut checksum = crc32fast::Hasher::new();
                checksum.update(rgb.as_raw());
                let ground_truth = camera_boxes
                    .get(&frame.timestamp)
                    .cloned()
                    .unwrap_or_default();
                let inference_started = Instant::now();
                let detections = if let Some(detector) = detector.as_mut() {
                    detector
                        .detect(&decoded)?
                        .into_iter()
                        .enumerate()
                        .map(|(index, prediction)| Detection {
                            id: format!("onnx-{index}"),
                            kind: prediction.kind,
                            x: prediction.x,
                            y: prediction.y,
                            width: prediction.width,
                            height: prediction.height,
                            depth_m: None,
                            depth_points: 0,
                            score: Some(prediction.score),
                        })
                        .collect()
                } else {
                    ground_truth.clone()
                };
                let inference_ns = detector
                    .as_ref()
                    .map(|_| elapsed_ns(inference_started))
                    .unwrap_or(0);
                let detection_batch = detection_batch(&detections)?;
                let output = CameraOutput {
                    timestamp: frame.timestamp,
                    image_width: rgb.width(),
                    image_height: rgb.height(),
                    image_checksum: checksum.finalize(),
                    detections,
                    detection_batch,
                    ground_truth,
                    inference_ns,
                    stage_ns: elapsed_ns(started),
                    started,
                    enqueued: Instant::now(),
                };
                camera_tx
                    .send(output)
                    .map_err(|_| anyhow::anyhow!("fusion stage closed the camera channel"))
            })?;
            Ok(setup_ns)
        })?;

    let lidar_path = component_path(root, split, "lidar", segment);
    let projection_path = component_path(root, split, "lidar_camera_projection", segment);
    let calibration_path = component_path(root, split, "lidar_calibration", segment);
    let lidar_worker = thread::Builder::new()
        .name("lidar-arrow".to_owned())
        .spawn(move || -> Result<u64> {
            let setup_started = Instant::now();
            let calibration = read_calibration(&calibration_path)?;
            let projections = read_all_top_lidar_frames(
                &projection_path,
                "[LiDARCameraProjectionComponent]",
                frame_limit,
            )?;
            let setup_ns = elapsed_ns(setup_started);
            visit_top_lidar_frames(&lidar_path, "[LiDARComponent]", frame_limit, |frame| {
                let started = Instant::now();
                let projection = projections
                    .get(&frame.timestamp)
                    .with_context(|| format!("missing camera projection at {}", frame.timestamp))?;
                let cloud = convert_top_lidar(&frame, projection, &calibration)?;
                let output = LidarOutput {
                    timestamp: frame.timestamp,
                    cloud,
                    stage_ns: elapsed_ns(started),
                    started,
                    enqueued: Instant::now(),
                };
                lidar_tx
                    .send(output)
                    .map_err(|_| anyhow::anyhow!("fusion stage closed the LiDAR channel"))
            })?;
            Ok(setup_ns)
        })?;

    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut csv = BufWriter::new(
        File::create(output).with_context(|| format!("failed to create {}", output.display()))?,
    );
    writeln!(
        csv,
        "frame,timestamp_micros,status,camera_width,camera_height,camera_checksum,points,detections,ground_truth,true_positives,false_positives,false_negatives,depth_matched,camera_stage_ns,inference_ns,lidar_stage_ns,camera_queue_wait_ns,lidar_queue_wait_ns,sync_skew_ns,fusion_ns,end_to_end_ns"
    )?;
    let mut summary = PerceptionSummary {
        frames: 0,
        points: 0,
        detections: 0,
        depth_matched: 0,
        elapsed: Duration::ZERO,
        true_positives: 0,
        false_positives: 0,
        false_negatives: 0,
    };
    while let Ok(mut camera) = camera_rx.recv() {
        let camera_wait = elapsed_ns(camera.enqueued);
        let lidar = lidar_rx
            .recv()
            .context("LiDAR stage ended before the camera stage")?;
        let lidar_wait = elapsed_ns(lidar.enqueued);
        let skew_ns = camera.timestamp.abs_diff(lidar.timestamp) * 1_000;
        let fusion_started = Instant::now();
        if fusion_work_ms > 0 {
            thread::sleep(Duration::from_millis(fusion_work_ms));
        }
        if camera.timestamp == lidar.timestamp {
            add_depths(&mut camera.detections, &lidar.cloud.points);
        }
        let fusion_ns = elapsed_ns(fusion_started);
        let (true_positives, false_positives, false_negatives) =
            evaluate_detections(&camera.detections, &camera.ground_truth, 0.5);
        let depth_matched = camera
            .detections
            .iter()
            .filter(|detection| detection.depth_m.is_some())
            .count();
        let first_started = camera.started.min(lidar.started);
        writeln!(
            csv,
            "{},{},{},{},{},{:08x},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            summary.frames,
            camera.timestamp,
            if camera.timestamp == lidar.timestamp {
                "delivered"
            } else {
                "unsynchronized"
            },
            camera.image_width,
            camera.image_height,
            camera.image_checksum,
            lidar.cloud.batch.num_rows(),
            camera.detection_batch.num_rows(),
            camera.ground_truth.len(),
            true_positives,
            false_positives,
            false_negatives,
            depth_matched,
            camera.stage_ns,
            camera.inference_ns,
            lidar.stage_ns,
            camera_wait,
            lidar_wait,
            skew_ns,
            fusion_ns,
            elapsed_ns(first_started),
        )?;
        summary.frames += 1;
        summary.points += lidar.cloud.batch.num_rows();
        summary.detections += camera.detection_batch.num_rows();
        summary.depth_matched += depth_matched;
        summary.true_positives += true_positives;
        summary.false_positives += false_positives;
        summary.false_negatives += false_negatives;
    }
    let camera_setup_ns = camera_worker
        .join()
        .map_err(|_| anyhow::anyhow!("camera stage panicked"))??;
    let lidar_setup_ns = lidar_worker
        .join()
        .map_err(|_| anyhow::anyhow!("LiDAR stage panicked"))??;
    csv.flush()?;
    let mut setup = BufWriter::new(File::create(output.with_extension("setup.txt"))?);
    writeln!(setup, "camera_setup_ns={camera_setup_ns}")?;
    writeln!(setup, "lidar_setup_ns={lidar_setup_ns}")?;
    setup.flush()?;
    summary.elapsed = run_started.elapsed();
    Ok(summary)
}

pub fn run_camera_only(
    root: &Path,
    split: &str,
    segment: &str,
    config: PerceptionConfig<'_>,
    output: &Path,
) -> Result<PerceptionSummary> {
    let frame_limit = config.frame_limit;
    let model = config
        .onnx_model
        .context("camera-only baseline requires an ONNX model")?;
    let run = Instant::now();
    let setup = Instant::now();
    let boxes = read_all_boxes(&component_path(root, split, "camera_box", segment))?;
    let mut detector = crate::detector::YoloxDetector::load(model, config.confidence, config.nms)?;
    let setup_ns = elapsed_ns(setup);
    create_parent(output)?;
    let mut csv = BufWriter::new(File::create(output)?);
    write_perception_header(&mut csv)?;
    let mut summary = empty_summary();
    visit_camera_frames(
        &component_path(root, split, "camera_image", segment),
        frame_limit,
        |frame| {
            let started = Instant::now();
            let decoded = image::load_from_memory(&frame.jpeg)?;
            let rgb = decoded.to_rgb8();
            let mut checksum = crc32fast::Hasher::new();
            checksum.update(rgb.as_raw());
            let inference = Instant::now();
            let detections = detector.detect(&decoded)?;
            let inference_ns = elapsed_ns(inference);
            let truth = boxes.get(&frame.timestamp).cloned().unwrap_or_default();
            let predicted: Vec<Detection> = detections
                .into_iter()
                .enumerate()
                .map(|(i, p)| Detection {
                    id: format!("onnx-{i}"),
                    kind: p.kind,
                    x: p.x,
                    y: p.y,
                    width: p.width,
                    height: p.height,
                    depth_m: None,
                    depth_points: 0,
                    score: Some(p.score),
                })
                .collect();
            let (tp, fp, fn_) = evaluate_detections(&predicted, &truth, 0.5);
            writeln!(
                csv,
                "{},{},delivered,{},{},{:08x},0,{},{},{tp},{fp},{fn_},0,{},{inference_ns},0,0,0,0,0,{}",
                summary.frames,
                frame.timestamp,
                rgb.width(),
                rgb.height(),
                checksum.finalize(),
                predicted.len(),
                truth.len(),
                elapsed_ns(started),
                elapsed_ns(started)
            )?;
            summary.frames += 1;
            summary.detections += predicted.len();
            summary.true_positives += tp;
            summary.false_positives += fp;
            summary.false_negatives += fn_;
            Ok(())
        },
    )?;
    csv.flush()?;
    std::fs::write(
        output.with_extension("setup.txt"),
        format!("camera_setup_ns={setup_ns}\nlidar_setup_ns=0\n"),
    )?;
    summary.elapsed = run.elapsed();
    Ok(summary)
}

pub fn run_lidar_only(
    root: &Path,
    split: &str,
    segment: &str,
    frame_limit: usize,
    output: &Path,
) -> Result<PerceptionSummary> {
    let run = Instant::now();
    let setup = Instant::now();
    let calibration = read_calibration(&component_path(root, split, "lidar_calibration", segment))?;
    let projections = read_all_top_lidar_frames(
        &component_path(root, split, "lidar_camera_projection", segment),
        "[LiDARCameraProjectionComponent]",
        frame_limit,
    )?;
    let setup_ns = elapsed_ns(setup);
    create_parent(output)?;
    let mut csv = BufWriter::new(File::create(output)?);
    write_perception_header(&mut csv)?;
    let mut summary = empty_summary();
    visit_top_lidar_frames(
        &component_path(root, split, "lidar", segment),
        "[LiDARComponent]",
        frame_limit,
        |frame| {
            let started = Instant::now();
            let projection = projections
                .get(&frame.timestamp)
                .context("missing camera projection")?;
            let cloud = convert_top_lidar(&frame, projection, &calibration)?;
            writeln!(
                csv,
                "{},{},delivered,0,0,00000000,{},0,0,0,0,0,0,0,0,{},0,0,0,0,{}",
                summary.frames,
                frame.timestamp,
                cloud.points.len(),
                elapsed_ns(started),
                elapsed_ns(started)
            )?;
            summary.frames += 1;
            summary.points += cloud.points.len();
            Ok(())
        },
    )?;
    csv.flush()?;
    std::fs::write(
        output.with_extension("setup.txt"),
        format!("camera_setup_ns=0\nlidar_setup_ns={setup_ns}\n"),
    )?;
    summary.elapsed = run.elapsed();
    Ok(summary)
}

fn create_parent(output: &Path) -> Result<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn write_perception_header(csv: &mut impl Write) -> Result<()> {
    writeln!(
        csv,
        "frame,timestamp_micros,status,camera_width,camera_height,camera_checksum,points,detections,ground_truth,true_positives,false_positives,false_negatives,depth_matched,camera_stage_ns,inference_ns,lidar_stage_ns,camera_queue_wait_ns,lidar_queue_wait_ns,sync_skew_ns,fusion_ns,end_to_end_ns"
    )?;
    Ok(())
}

fn empty_summary() -> PerceptionSummary {
    PerceptionSummary {
        frames: 0,
        points: 0,
        detections: 0,
        depth_matched: 0,
        elapsed: Duration::ZERO,
        true_positives: 0,
        false_positives: 0,
        false_negatives: 0,
    }
}

pub fn build_demo(
    root: &Path,
    split: &str,
    segment: &str,
    frame_index: usize,
    output: &Path,
) -> Result<DemoSummary> {
    let started = Instant::now();
    let camera_path = component_path(root, split, "camera_image", segment);
    let lidar_path = component_path(root, split, "lidar", segment);
    let projection_path = component_path(root, split, "lidar_camera_projection", segment);
    let calibration_path = component_path(root, split, "lidar_calibration", segment);
    let boxes_path = component_path(root, split, "camera_box", segment);

    let camera = read_camera_frame(&camera_path, frame_index)?;
    let lidar = read_lidar_frame(&lidar_path, camera.timestamp, "[LiDARComponent]")?;
    let projection = read_lidar_frame(
        &projection_path,
        camera.timestamp,
        "[LiDARCameraProjectionComponent]",
    )?;
    anyhow::ensure!(
        lidar.shape[0..2] == projection.shape[0..2],
        "LiDAR and camera projection grids do not have matching dimensions"
    );
    let calibration = read_calibration(&calibration_path)?;
    let cloud = convert_top_lidar(&lidar, &projection, &calibration)?;
    let boxes = read_boxes(&boxes_path, camera.timestamp, &cloud.points)?;

    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let arrow_path = output.with_extension("pointcloud.arrow");
    write_arrow(&arrow_path, &cloud.batch)?;

    let render_points = downsample_for_render(&cloud.points);
    let arrow_file = arrow_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let processing_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let payload = DemoPayload {
        segment,
        split,
        frame_index,
        timestamp_micros: camera.timestamp,
        total_points: cloud.points.len(),
        render_points,
        boxes,
        processing_ms,
        arrow_file,
    };
    let boxes_count = payload.boxes.len();
    let boxes_with_depth = payload
        .boxes
        .iter()
        .filter(|detection| detection.depth_m.is_some())
        .count();
    let html = include_str!("../assets/fusion-demo.html")
        .replace("__CAMERA_JPEG__", &STANDARD.encode(&camera.jpeg))
        .replace("__DEMO_PAYLOAD__", &serde_json::to_string(&payload)?);
    std::fs::write(output, html)
        .with_context(|| format!("failed to write {}", output.display()))?;

    Ok(DemoSummary {
        html: output.to_path_buf(),
        arrow: arrow_path,
        timestamp_micros: camera.timestamp,
        points: cloud.points.len(),
        boxes: boxes_count,
        boxes_with_depth,
    })
}

pub fn component_path(root: &Path, split: &str, component: &str, segment: &str) -> PathBuf {
    root.join(split)
        .join(component)
        .join(format!("{segment}.parquet"))
}

fn reader(path: &Path) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader> {
    let file = File::open(path).with_context(|| {
        format!(
            "missing {}; rerun scripts/download_waymo_sample.py --with-labels",
            path.display()
        )
    })?;
    Ok(ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(16)
        .build()?)
}

fn read_camera_frame(path: &Path, frame_index: usize) -> Result<CameraFrame> {
    let mut matched = 0;
    for batch in reader(path)? {
        let batch = batch?;
        let timestamp = int64(&batch, "key.frame_timestamp_micros")?;
        let camera = integer(&batch, "key.camera_name")?;
        let image = binary(&batch, "[CameraImageComponent].image")?;
        for row in 0..batch.num_rows() {
            if camera(row) == FRONT_CAMERA {
                if matched == frame_index {
                    return Ok(CameraFrame {
                        timestamp: timestamp.value(row),
                        jpeg: image.value(row).to_vec(),
                    });
                }
                matched += 1;
            }
        }
    }
    bail!("front camera frame index {frame_index} does not exist")
}

fn read_lidar_frame(path: &Path, target_timestamp: i64, prefix: &str) -> Result<LidarFrame> {
    for batch in reader(path)? {
        let batch = batch?;
        let timestamp = int64(&batch, "key.frame_timestamp_micros")?;
        let laser = integer(&batch, "key.laser_name")?;
        for row in 0..batch.num_rows() {
            if timestamp.value(row) == target_timestamp && laser(row) == TOP_LIDAR {
                let values =
                    list_f32(&batch, &format!("{prefix}.range_image_return1.values"), row)?;
                let shape = fixed_i32(&batch, &format!("{prefix}.range_image_return1.shape"), row)?;
                anyhow::ensure!(shape.len() == 3, "range-image shape must have 3 dimensions");
                let shape = [shape[0] as usize, shape[1] as usize, shape[2] as usize];
                anyhow::ensure!(
                    shape.iter().product::<usize>() == values.len(),
                    "range-image shape does not match value count"
                );
                return Ok(LidarFrame {
                    timestamp: target_timestamp,
                    values,
                    shape,
                });
            }
        }
    }
    bail!("top LiDAR has no row at camera timestamp {target_timestamp}")
}

pub fn read_calibration(path: &Path) -> Result<Calibration> {
    for batch in reader(path)? {
        let batch = batch?;
        let laser = integer(&batch, "key.laser_name")?;
        for row in 0..batch.num_rows() {
            if laser(row) != TOP_LIDAR {
                continue;
            }
            let transform = fixed_f64(
                &batch,
                "[LiDARCalibrationComponent].extrinsic.transform",
                row,
            )?;
            let extrinsic: [f64; 16] = transform
                .try_into()
                .map_err(|_| anyhow::anyhow!("LiDAR extrinsic must contain 16 values"))?;
            return Ok(Calibration {
                extrinsic,
                inclination_min: float64_scalar(
                    &batch,
                    "[LiDARCalibrationComponent].beam_inclination.min",
                    row,
                )?,
                inclination_max: float64_scalar(
                    &batch,
                    "[LiDARCalibrationComponent].beam_inclination.max",
                    row,
                )?,
                inclinations: list_f64(
                    &batch,
                    "[LiDARCalibrationComponent].beam_inclination.values",
                    row,
                )?,
            });
        }
    }
    bail!("top LiDAR calibration is missing")
}

pub fn convert_top_lidar(
    lidar: &LidarFrame,
    projection: &LidarFrame,
    calibration: &Calibration,
) -> Result<PointCloud> {
    let [height, width, channels] = lidar.shape;
    anyhow::ensure!(
        channels >= 4,
        "LiDAR return must include range, intensity, and elongation"
    );
    anyhow::ensure!(
        projection.shape[2] >= 6,
        "camera projection must contain two camera/x/y triples"
    );

    let mut inclinations = if calibration.inclinations.len() == height {
        calibration.inclinations.clone()
    } else {
        (0..height)
            .map(|row| {
                calibration.inclination_min
                    + (calibration.inclination_max - calibration.inclination_min)
                        * (row as f64 + 0.5)
                        / height as f64
            })
            .collect()
    };
    inclinations.reverse();
    let azimuth_correction = calibration.extrinsic[1].atan2(calibration.extrinsic[0]);
    let mut points = Vec::with_capacity(height * width);

    for (row, inclination) in inclinations.iter().enumerate() {
        let cos_inclination = inclination.cos();
        let sin_inclination = inclination.sin();
        for column in 0..width {
            let offset = (row * width + column) * channels;
            let range = lidar.values[offset];
            if !range.is_finite() || range <= 0.0 {
                continue;
            }
            let azimuth = ((width as f64 - column as f64 - 0.5) / width as f64 * 2.0 - 1.0)
                * std::f64::consts::PI
                - azimuth_correction;
            let sensor = [
                azimuth.cos() * cos_inclination * range as f64,
                azimuth.sin() * cos_inclination * range as f64,
                sin_inclination * range as f64,
            ];
            let transform = &calibration.extrinsic;
            let x = transform[0] * sensor[0]
                + transform[1] * sensor[1]
                + transform[2] * sensor[2]
                + transform[3];
            let y = transform[4] * sensor[0]
                + transform[5] * sensor[1]
                + transform[6] * sensor[2]
                + transform[7];
            let z = transform[8] * sensor[0]
                + transform[9] * sensor[1]
                + transform[10] * sensor[2]
                + transform[11];
            let projection_offset = (row * projection.shape[1] + column) * projection.shape[2];
            let mut camera = 0_i8;
            let mut u = -1.0;
            let mut v = -1.0;
            for projection_index in [0, 3] {
                let candidate = projection.values[projection_offset + projection_index] as i8;
                if candidate == FRONT_CAMERA as i8 {
                    camera = candidate;
                    u = projection.values[projection_offset + projection_index + 1];
                    v = projection.values[projection_offset + projection_index + 2];
                    break;
                }
            }
            points.push(Point {
                x: x as f32,
                y: y as f32,
                z: z as f32,
                range,
                intensity: lidar.values[offset + 1],
                elongation: lidar.values[offset + 2],
                camera,
                u,
                v,
            });
        }
    }
    let batch = point_batch(&points)?;
    Ok(PointCloud { batch, points })
}

fn point_batch(points: &[Point]) -> Result<RecordBatch> {
    let metadata = HashMap::from([
        ("coordinate_frame".to_owned(), "vehicle".to_owned()),
        ("source".to_owned(), "waymo_top_lidar_return1".to_owned()),
        ("motion_compensated".to_owned(), "false".to_owned()),
    ]);
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
            Field::new("z", DataType::Float32, false),
            Field::new("range", DataType::Float32, false),
            Field::new("intensity", DataType::Float32, false),
            Field::new("elongation", DataType::Float32, false),
            Field::new("front_camera", DataType::Int8, false),
            Field::new("front_u", DataType::Float32, false),
            Field::new("front_v", DataType::Float32, false),
        ],
        metadata,
    ));
    let float_column = |value: fn(&Point) -> f32| -> ArrayRef {
        Arc::new(Float32Array::from_iter_values(points.iter().map(value)))
    };
    RecordBatch::try_new(
        schema,
        vec![
            float_column(|point| point.x),
            float_column(|point| point.y),
            float_column(|point| point.z),
            float_column(|point| point.range),
            float_column(|point| point.intensity),
            float_column(|point| point.elongation),
            Arc::new(Int8Array::from_iter_values(
                points.iter().map(|point| point.camera),
            )),
            float_column(|point| point.u),
            float_column(|point| point.v),
        ],
    )
    .context("failed to construct the Arrow point-cloud batch")
}

pub fn detection_batch(detections: &[Detection]) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("object_id", DataType::Utf8, false),
            Field::new("class", DataType::Utf8, false),
            Field::new("score", DataType::Float32, true),
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("width", DataType::Float64, false),
            Field::new("height", DataType::Float64, false),
        ],
        HashMap::from([(
            "coordinate_frame".to_owned(),
            "front_camera_pixels".to_owned(),
        )]),
    ));
    let float64_column = |value: fn(&Detection) -> f64| -> ArrayRef {
        Arc::new(Float64Array::from_iter_values(detections.iter().map(value)))
    };
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from_iter_values(
                detections.iter().map(|detection| detection.id.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                detections.iter().map(|detection| detection.kind),
            )),
            Arc::new(Float32Array::from_iter(
                detections.iter().map(|detection| detection.score),
            )),
            float64_column(|detection| detection.x),
            float64_column(|detection| detection.y),
            float64_column(|detection| detection.width),
            float64_column(|detection| detection.height),
        ],
    )
    .context("failed to construct the Arrow detection batch")
}

pub fn visit_camera_frames(
    path: &Path,
    limit: usize,
    mut visit: impl FnMut(CameraFrame) -> Result<()>,
) -> Result<()> {
    let mut count = 0;
    for batch in reader(path)? {
        let batch = batch?;
        let timestamp = int64(&batch, "key.frame_timestamp_micros")?;
        let camera = integer(&batch, "key.camera_name")?;
        let image = binary(&batch, "[CameraImageComponent].image")?;
        for row in 0..batch.num_rows() {
            if camera(row) != FRONT_CAMERA {
                continue;
            }
            visit(CameraFrame {
                timestamp: timestamp.value(row),
                jpeg: image.value(row).to_vec(),
            })?;
            count += 1;
            if limit > 0 && count >= limit {
                return Ok(());
            }
        }
    }
    Ok(())
}

pub fn visit_top_lidar_frames(
    path: &Path,
    prefix: &str,
    limit: usize,
    mut visit: impl FnMut(LidarFrame) -> Result<()>,
) -> Result<()> {
    let mut count = 0;
    for batch in reader(path)? {
        let batch = batch?;
        let timestamp = int64(&batch, "key.frame_timestamp_micros")?;
        let laser = integer(&batch, "key.laser_name")?;
        for row in 0..batch.num_rows() {
            if laser(row) != TOP_LIDAR {
                continue;
            }
            let values = list_f32(&batch, &format!("{prefix}.range_image_return1.values"), row)?;
            let raw_shape = fixed_i32(&batch, &format!("{prefix}.range_image_return1.shape"), row)?;
            anyhow::ensure!(
                raw_shape.len() == 3,
                "range-image shape must have 3 dimensions"
            );
            let shape = [
                raw_shape[0] as usize,
                raw_shape[1] as usize,
                raw_shape[2] as usize,
            ];
            anyhow::ensure!(shape.iter().product::<usize>() == values.len());
            visit(LidarFrame {
                timestamp: timestamp.value(row),
                values,
                shape,
            })?;
            count += 1;
            if limit > 0 && count >= limit {
                return Ok(());
            }
        }
    }
    Ok(())
}

pub fn read_all_top_lidar_frames(
    path: &Path,
    prefix: &str,
    limit: usize,
) -> Result<HashMap<i64, LidarFrame>> {
    let mut frames = HashMap::new();
    visit_top_lidar_frames(path, prefix, limit, |frame| {
        frames.insert(frame.timestamp, frame);
        Ok(())
    })?;
    Ok(frames)
}

pub fn read_all_boxes(path: &Path) -> Result<HashMap<i64, Vec<Detection>>> {
    let mut output: HashMap<i64, Vec<Detection>> = HashMap::new();
    for batch in reader(path)? {
        let batch = batch?;
        let timestamp = int64(&batch, "key.frame_timestamp_micros")?;
        let camera = integer(&batch, "key.camera_name")?;
        let id = string(&batch, "key.camera_object_id")?;
        let center_x = float64(&batch, "[CameraBoxComponent].box.center.x")?;
        let center_y = float64(&batch, "[CameraBoxComponent].box.center.y")?;
        let width = float64(&batch, "[CameraBoxComponent].box.size.x")?;
        let height = float64(&batch, "[CameraBoxComponent].box.size.y")?;
        let kind = integer(&batch, "[CameraBoxComponent].type")?;
        for row in 0..batch.num_rows() {
            if camera(row) != FRONT_CAMERA {
                continue;
            }
            let width = width.value(row);
            let height = height.value(row);
            output
                .entry(timestamp.value(row))
                .or_default()
                .push(Detection {
                    id: id.value(row).to_owned(),
                    kind: object_type(kind(row)),
                    x: center_x.value(row) - width / 2.0,
                    y: center_y.value(row) - height / 2.0,
                    width,
                    height,
                    depth_m: None,
                    depth_points: 0,
                    score: None,
                });
        }
    }
    Ok(output)
}

pub fn add_depths(detections: &mut [Detection], points: &[Point]) {
    for detection in detections {
        let mut ranges: Vec<f32> = points
            .iter()
            .filter(|point| {
                point.camera == FRONT_CAMERA as i8
                    && point.u as f64 >= detection.x
                    && point.u as f64 <= detection.x + detection.width
                    && point.v as f64 >= detection.y
                    && point.v as f64 <= detection.y + detection.height
            })
            .map(|point| point.range)
            .filter(|range| range.is_finite())
            .collect();
        ranges.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        detection.depth_points = ranges.len();
        detection.depth_m = (!ranges.is_empty()).then(|| ranges[ranges.len() / 2]);
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn read_boxes(path: &Path, timestamp_target: i64, points: &[Point]) -> Result<Vec<Detection>> {
    let mut boxes = Vec::new();
    for batch in reader(path)? {
        let batch = batch?;
        let timestamp = int64(&batch, "key.frame_timestamp_micros")?;
        let camera = integer(&batch, "key.camera_name")?;
        let id = string(&batch, "key.camera_object_id")?;
        let center_x = float64(&batch, "[CameraBoxComponent].box.center.x")?;
        let center_y = float64(&batch, "[CameraBoxComponent].box.center.y")?;
        let width = float64(&batch, "[CameraBoxComponent].box.size.x")?;
        let height = float64(&batch, "[CameraBoxComponent].box.size.y")?;
        let kind = integer(&batch, "[CameraBoxComponent].type")?;
        for row in 0..batch.num_rows() {
            if timestamp.value(row) != timestamp_target || camera(row) != FRONT_CAMERA {
                continue;
            }
            let (x, y, width, height) = (
                center_x.value(row) - width.value(row) / 2.0,
                center_y.value(row) - height.value(row) / 2.0,
                width.value(row),
                height.value(row),
            );
            let mut ranges: Vec<f32> = points
                .iter()
                .filter(|point| {
                    point.camera == FRONT_CAMERA as i8
                        && point.u as f64 >= x
                        && point.u as f64 <= x + width
                        && point.v as f64 >= y
                        && point.v as f64 <= y + height
                })
                .map(|point| point.range)
                .filter(|range| range.is_finite())
                .collect();
            ranges.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
            let depth_m = (!ranges.is_empty()).then(|| ranges[ranges.len() / 2]);
            boxes.push(Detection {
                id: id.value(row).to_owned(),
                kind: object_type(kind(row)),
                x,
                y,
                width,
                height,
                depth_m,
                depth_points: ranges.len(),
                score: None,
            });
        }
    }
    Ok(boxes)
}

fn object_type(value: i64) -> &'static str {
    match value {
        1 => "vehicle",
        2 => "pedestrian",
        3 => "sign",
        4 => "cyclist",
        _ => "unknown",
    }
}

pub fn evaluate_detections(
    predictions: &[Detection],
    ground_truth: &[Detection],
    threshold: f64,
) -> (usize, usize, usize) {
    let mut matched = vec![false; ground_truth.len()];
    let mut true_positives = 0;
    for prediction in predictions {
        let best = ground_truth
            .iter()
            .enumerate()
            .filter(|(index, truth)| !matched[*index] && truth.kind == prediction.kind)
            .map(|(index, truth)| (index, detection_iou(prediction, truth)))
            .filter(|(_, iou)| *iou >= threshold)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((index, _)) = best {
            matched[index] = true;
            true_positives += 1;
        }
    }
    (
        true_positives,
        predictions.len() - true_positives,
        ground_truth.len() - true_positives,
    )
}

fn detection_iou(a: &Detection, b: &Detection) -> f64 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    intersection / (a.width * a.height + b.width * b.height - intersection).max(f64::EPSILON)
}

fn downsample_for_render(points: &[Point]) -> Vec<RenderPoint> {
    let visible: Vec<_> = points
        .iter()
        .filter(|point| point.x >= -30.0 && point.x <= 80.0 && point.y.abs() <= 50.0)
        .collect();
    let stride = visible.len().div_ceil(MAX_RENDER_POINTS).max(1);
    visible
        .into_iter()
        .step_by(stride)
        .map(|point| RenderPoint(point.x, point.y, point.z, point.u, point.v))
        .collect()
}

fn write_arrow(path: &Path, batch: &RecordBatch) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = FileWriter::try_new(file, batch.schema().as_ref())?;
    writer.write(batch)?;
    writer.finish()?;
    Ok(())
}

fn column_index(batch: &RecordBatch, name: &str) -> Result<usize> {
    batch
        .schema()
        .index_of(name)
        .with_context(|| format!("missing Arrow column {name}"))
}

fn int64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch
        .column(column_index(batch, name)?)
        .as_any()
        .downcast_ref()
        .with_context(|| format!("{name} is not Int64"))
}

fn float64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    batch
        .column(column_index(batch, name)?)
        .as_any()
        .downcast_ref()
        .with_context(|| format!("{name} is not Float64"))
}

fn binary<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BinaryArray> {
    batch
        .column(column_index(batch, name)?)
        .as_any()
        .downcast_ref()
        .with_context(|| format!("{name} is not Binary"))
}

fn string<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column(column_index(batch, name)?)
        .as_any()
        .downcast_ref()
        .with_context(|| format!("{name} is not Utf8"))
}

fn integer<'a>(batch: &'a RecordBatch, name: &str) -> Result<Box<dyn Fn(usize) -> i64 + 'a>> {
    let column = batch.column(column_index(batch, name)?);
    if let Some(array) = column.as_any().downcast_ref::<Int8Array>() {
        return Ok(Box::new(move |row| array.value(row) as i64));
    }
    if let Some(array) = column.as_any().downcast_ref::<Int32Array>() {
        return Ok(Box::new(move |row| array.value(row) as i64));
    }
    bail!("{name} is not a supported integer type")
}

fn float64_scalar(batch: &RecordBatch, name: &str, row: usize) -> Result<f64> {
    Ok(float64(batch, name)?.value(row))
}

fn list_f32(batch: &RecordBatch, name: &str, row: usize) -> Result<Vec<f32>> {
    let list = batch
        .column(column_index(batch, name)?)
        .as_any()
        .downcast_ref::<ListArray>()
        .with_context(|| format!("{name} is not List"))?;
    let value = list.value(row);
    let array = value
        .as_any()
        .downcast_ref::<Float32Array>()
        .with_context(|| format!("{name} values are not Float32"))?;
    Ok(array.values().to_vec())
}

fn list_f64(batch: &RecordBatch, name: &str, row: usize) -> Result<Vec<f64>> {
    let list = batch
        .column(column_index(batch, name)?)
        .as_any()
        .downcast_ref::<ListArray>()
        .with_context(|| format!("{name} is not List"))?;
    let value = list.value(row);
    let array = value
        .as_any()
        .downcast_ref::<Float64Array>()
        .with_context(|| format!("{name} values are not Float64"))?;
    Ok(array.values().to_vec())
}

fn fixed_i32(batch: &RecordBatch, name: &str, row: usize) -> Result<Vec<i32>> {
    let list = batch
        .column(column_index(batch, name)?)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .with_context(|| format!("{name} is not FixedSizeList"))?;
    let value = list.value(row);
    let array = value
        .as_any()
        .downcast_ref::<Int32Array>()
        .with_context(|| format!("{name} values are not Int32"))?;
    Ok(array.values().to_vec())
}

fn fixed_f64(batch: &RecordBatch, name: &str, row: usize) -> Result<Vec<f64>> {
    let list = batch
        .column(column_index(batch, name)?)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .with_context(|| format!("{name} is not FixedSizeList"))?;
    let value = list.value(row);
    let array = value
        .as_any()
        .downcast_ref::<Float64Array>()
        .with_context(|| format!("{name} values are not Float64"))?;
    Ok(array.values().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_valid_ranges_to_an_arrow_point_cloud() {
        let lidar = LidarFrame {
            timestamp: 123,
            values: vec![10.0, 0.5, 0.1, 0.0, 0.0, 0.2, 0.3, 0.0],
            shape: [1, 2, 4],
        };
        let projection = LidarFrame {
            timestamp: 123,
            values: vec![
                1.0, 100.0, 200.0, 0.0, 0.0, 0.0, 1.0, 120.0, 220.0, 0.0, 0.0, 0.0,
            ],
            shape: [1, 2, 6],
        };
        let calibration = Calibration {
            extrinsic: [
                1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0,
            ],
            inclination_min: 0.0,
            inclination_max: 0.0,
            inclinations: vec![0.0],
        };

        let cloud = convert_top_lidar(&lidar, &projection, &calibration).unwrap();
        assert_eq!(cloud.points.len(), 1, "zero ranges must be discarded");
        assert_eq!(cloud.batch.num_rows(), 1);
        assert_eq!(cloud.batch.num_columns(), 9);
        assert_eq!(cloud.points[0].camera, 1);
        assert_eq!((cloud.points[0].u, cloud.points[0].v), (100.0, 200.0));
        assert!(cloud.points[0].x.is_finite());
        assert!(cloud.points[0].y.is_finite());
        assert_eq!(cloud.points[0].z, 3.0);
        assert_eq!(
            cloud.batch.schema().metadata().get("coordinate_frame"),
            Some(&"vehicle".to_owned())
        );
    }
}
