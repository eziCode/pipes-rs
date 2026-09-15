use std::{
    cmp::Ordering,
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
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

struct CameraFrame {
    timestamp: i64,
    jpeg: Vec<u8>,
}

struct LidarFrame {
    values: Vec<f32>,
    shape: [usize; 3],
}

struct Calibration {
    extrinsic: [f64; 16],
    inclination_min: f64,
    inclination_max: f64,
    inclinations: Vec<f64>,
}

#[derive(Clone)]
struct Point {
    x: f32,
    y: f32,
    z: f32,
    range: f32,
    intensity: f32,
    elongation: f32,
    camera: i8,
    u: f32,
    v: f32,
}

struct PointCloud {
    batch: RecordBatch,
    points: Vec<Point>,
}

#[derive(Serialize)]
struct Detection {
    id: String,
    kind: &'static str,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    depth_m: Option<f32>,
    depth_points: usize,
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

fn component_path(root: &Path, split: &str, component: &str, segment: &str) -> PathBuf {
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
        .with_batch_size(256)
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
                return Ok(LidarFrame { values, shape });
            }
        }
    }
    bail!("top LiDAR has no row at camera timestamp {target_timestamp}")
}

fn read_calibration(path: &Path) -> Result<Calibration> {
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

fn convert_top_lidar(
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
            values: vec![10.0, 0.5, 0.1, 0.0, 0.0, 0.2, 0.3, 0.0],
            shape: [1, 2, 4],
        };
        let projection = LidarFrame {
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
