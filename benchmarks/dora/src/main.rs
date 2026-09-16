use anyhow::{Context, Result};
use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, Float32Array, Int8Array, Int64Array, ListArray, StructArray,
        UInt32Array, UInt64Array,
    },
    buffer::{OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field},
};
use clap::{Parser, ValueEnum};
use dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, MetadataParameters};
use pipes_rs::{
    detector::YoloxDetector,
    fusion::{self, Detection, Point},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, ValueEnum)]
enum Role {
    Camera,
    Lidar,
    Fusion,
    CameraSink,
    LidarSink,
    TransportSource,
    TransportSink,
}
#[derive(Parser)]
struct Args {
    #[arg(value_enum)]
    role: Role,
}
#[derive(Clone, Deserialize, Serialize)]
struct WireDetection {
    id: String,
    kind: String,
    score: Option<f32>,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}
struct CameraMessage {
    timestamp: i64,
    predictions: Vec<WireDetection>,
    truth: Vec<WireDetection>,
    stage_ns: u64,
    inference_ns: u64,
    width: u32,
    height: u32,
    checksum: u32,
    started_unix_ns: u64,
    sent_unix_ns: u64,
}
struct LidarMessage {
    timestamp: i64,
    points: Vec<Point>,
    stage_ns: u64,
    started_unix_ns: u64,
    sent_unix_ns: u64,
}
struct CameraArrayMeta {
    timestamp: i64,
    stage_ns: u64,
    inference_ns: u64,
    width: u32,
    height: u32,
    checksum: u32,
    started_unix_ns: u64,
    sent_unix_ns: u64,
}

fn main() -> Result<()> {
    match Args::parse().role {
        Role::Camera => camera(),
        Role::Lidar => lidar(),
        Role::Fusion => fusion_node(),
        Role::CameraSink => camera_sink(),
        Role::LidarSink => lidar_sink(),
        Role::TransportSource => transport_source(),
        Role::TransportSink => transport_sink(),
    }
}
fn settings() -> (PathBuf, String, String, usize) {
    (
        env::var("DATA_ROOT")
            .unwrap_or_else(|_| "data/waymo-v2-sample".into())
            .into(),
        env::var("SPLIT").unwrap_or_else(|_| "training".into()),
        env::var("SEGMENT").unwrap_or_else(|_| "10023947602400723454_1120_000_1140_000".into()),
        env::var("FRAME_LIMIT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )
}

fn camera() -> Result<()> {
    let (mut node, _events) = DoraNode::init_from_env()?;
    let setup_started = Instant::now();
    let (root, split, segment, limit) = settings();
    let model = env::var("MODEL_PATH").unwrap_or_else(|_| "models/yolox_nano.onnx".into());
    let mut detector = YoloxDetector::load(
        Path::new(&model),
        env_f32("CONFIDENCE", 0.3),
        env_f32("NMS", 0.45),
    )?;
    let boxes = fusion::read_all_boxes(&fusion::component_path(
        &root,
        &split,
        "camera_box",
        &segment,
    ))?;
    write_setup("camera_setup_ns", ns(setup_started))?;
    let output = DataId::from("detections".to_owned());
    fusion::visit_camera_frames(
        &fusion::component_path(&root, &split, "camera_image", &segment),
        limit,
        |frame| {
            let started = Instant::now();
            let started_unix_ns = unix_ns();
            let image = image::load_from_memory(&frame.jpeg)?;
            let rgb = image.to_rgb8();
            let mut checksum = crc32fast::Hasher::new();
            checksum.update(rgb.as_raw());
            let inference = Instant::now();
            let predictions: Vec<_> = detector
                .detect(&image)?
                .into_iter()
                .enumerate()
                .map(|(i, p)| WireDetection {
                    id: format!("onnx-{i}"),
                    kind: p.kind.into(),
                    score: Some(p.score),
                    x: p.x,
                    y: p.y,
                    width: p.width,
                    height: p.height,
                })
                .collect();
            let inference_ns = ns(inference);
            let truth = boxes
                .get(&frame.timestamp)
                .map(|v| v.iter().map(WireDetection::from).collect::<Vec<_>>())
                .unwrap_or_default();
            let sent_unix_ns = unix_ns();
            let message = camera_array(
                &predictions,
                &truth,
                CameraArrayMeta {
                    timestamp: frame.timestamp,
                    stage_ns: ns(started),
                    inference_ns,
                    width: rgb.width(),
                    height: rgb.height(),
                    checksum: checksum.finalize(),
                    started_unix_ns,
                    sent_unix_ns,
                },
            )?;
            node.send_output(output.clone(), MetadataParameters::default(), message)?;
            Ok(())
        },
    )
}
fn lidar() -> Result<()> {
    let (mut node, _events) = DoraNode::init_from_env()?;
    let setup_started = Instant::now();
    let (root, split, segment, limit) = settings();
    let calibration = fusion::read_calibration(&fusion::component_path(
        &root,
        &split,
        "lidar_calibration",
        &segment,
    ))?;
    let projections = fusion::read_all_top_lidar_frames(
        &fusion::component_path(&root, &split, "lidar_camera_projection", &segment),
        "[LiDARCameraProjectionComponent]",
        limit,
    )?;
    write_setup("lidar_setup_ns", ns(setup_started))?;
    let output = DataId::from("pointcloud".to_owned());
    fusion::visit_top_lidar_frames(
        &fusion::component_path(&root, &split, "lidar", &segment),
        "[LiDARComponent]",
        limit,
        |frame| {
            let started = Instant::now();
            let started_unix_ns = unix_ns();
            let projection = projections
                .get(&frame.timestamp)
                .context("missing LiDAR camera projection")?;
            let cloud = fusion::convert_top_lidar(&frame, projection, &calibration)?;
            let sent_unix_ns = unix_ns();
            node.send_output(
                output.clone(),
                MetadataParameters::default(),
                lidar_array(
                    frame.timestamp,
                    &cloud.points,
                    ns(started),
                    started_unix_ns,
                    sent_unix_ns,
                )?,
            )?;
            Ok(())
        },
    )
}
fn fusion_node() -> Result<()> {
    let (_node, mut events) = DoraNode::init_from_env()?;
    let path = env::var("OUTPUT_CSV").unwrap_or_else(|_| "dora-perception.csv".into());
    if let Some(parent) = Path::new(&path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut csv = BufWriter::new(File::create(&path)?);
    writeln!(
        csv,
        "frame,timestamp_micros,status,camera_width,camera_height,camera_checksum,points,detections,ground_truth,true_positives,false_positives,false_negatives,depth_matched,camera_stage_ns,inference_ns,lidar_stage_ns,camera_queue_wait_ns,lidar_queue_wait_ns,sync_skew_ns,fusion_ns,end_to_end_ns"
    )?;
    let run = Instant::now();
    let mut cameras = HashMap::new();
    let mut lidars = HashMap::new();
    let mut frame = 0usize;
    let (mut camera_closed, mut lidar_closed) = (false, false);
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, .. } if id.as_str() == "detections" => {
                let m = parse_camera(data.as_array().as_ref())?;
                cameras.insert(m.timestamp, m);
            }
            Event::Input { id, data, .. } if id.as_str() == "pointcloud" => {
                let m = parse_lidar(data.as_array().as_ref())?;
                lidars.insert(m.timestamp, m);
            }
            Event::InputClosed { id } if id.as_str() == "detections" => camera_closed = true,
            Event::InputClosed { id } if id.as_str() == "pointcloud" => lidar_closed = true,
            Event::Stop(_) => break,
            _ => {}
        }
        let ready: Vec<i64> = cameras
            .keys()
            .filter(|t| lidars.contains_key(t))
            .copied()
            .collect();
        for timestamp in ready {
            let camera = cameras.remove(&timestamp).unwrap();
            let lidar = lidars.remove(&timestamp).unwrap();
            let received_unix_ns = unix_ns();
            let fuse = Instant::now();
            if let Ok(ms) = env::var("FUSION_WORK_MS")
                .unwrap_or_default()
                .parse::<u64>()
            {
                std::thread::sleep(std::time::Duration::from_millis(ms));
            }
            let mut predicted: Vec<Detection> =
                camera.predictions.iter().map(Detection::from).collect();
            let truth: Vec<Detection> = camera.truth.iter().map(Detection::from).collect();
            fusion::add_depths(&mut predicted, &lidar.points);
            let (tp, fp, fn_) = fusion::evaluate_detections(&predicted, &truth, 0.5);
            let depth = predicted.iter().filter(|d| d.depth_m.is_some()).count();
            writeln!(
                csv,
                "{frame},{timestamp},delivered,{},{},{:08x},{},{},{},{tp},{fp},{fn_},{depth},{},{},{},{},{},0,{},{}",
                camera.width,
                camera.height,
                camera.checksum,
                lidar.points.len(),
                predicted.len(),
                truth.len(),
                camera.stage_ns,
                camera.inference_ns,
                lidar.stage_ns,
                received_unix_ns.saturating_sub(camera.sent_unix_ns),
                received_unix_ns.saturating_sub(lidar.sent_unix_ns),
                ns(fuse),
                unix_ns().saturating_sub(camera.started_unix_ns.min(lidar.started_unix_ns))
            )?;
            frame += 1;
        }
        if camera_closed && lidar_closed {
            break;
        }
    }
    csv.flush()?;
    println!(
        "dora perception: frames={frame} elapsed={:.2}s rate={:.2}Hz metrics={path}",
        run.elapsed().as_secs_f64(),
        frame as f64 / run.elapsed().as_secs_f64().max(f64::EPSILON)
    );
    Ok(())
}

fn camera_sink() -> Result<()> {
    let (_node, mut events) = DoraNode::init_from_env()?;
    let mut csv = output_csv()?;
    let mut frame = 0usize;
    while let Some(event) = events.recv() {
        match event {
            Event::Input { data, .. } => {
                let received = unix_ns();
                let camera = parse_camera(data.as_array().as_ref())?;
                let predicted: Vec<Detection> =
                    camera.predictions.iter().map(Detection::from).collect();
                let truth: Vec<Detection> = camera.truth.iter().map(Detection::from).collect();
                let (tp, fp, fn_) = fusion::evaluate_detections(&predicted, &truth, 0.5);
                writeln!(
                    csv,
                    "{frame},{},delivered,{},{},{:08x},0,{},{},{tp},{fp},{fn_},0,{},{},0,{},0,0,0,{}",
                    camera.timestamp,
                    camera.width,
                    camera.height,
                    camera.checksum,
                    predicted.len(),
                    truth.len(),
                    camera.stage_ns,
                    camera.inference_ns,
                    received.saturating_sub(camera.sent_unix_ns),
                    unix_ns().saturating_sub(camera.started_unix_ns)
                )?;
                frame += 1;
            }
            Event::InputClosed { .. } | Event::Stop(_) => break,
            _ => {}
        }
    }
    csv.flush()?;
    println!("dora camera-only: frames={frame}");
    Ok(())
}

fn lidar_sink() -> Result<()> {
    let (_node, mut events) = DoraNode::init_from_env()?;
    let mut csv = output_csv()?;
    let mut frame = 0usize;
    while let Some(event) = events.recv() {
        match event {
            Event::Input { data, .. } => {
                let received = unix_ns();
                let lidar = parse_lidar(data.as_array().as_ref())?;
                writeln!(
                    csv,
                    "{frame},{},delivered,0,0,00000000,{},0,0,0,0,0,0,0,0,{},0,{},0,0,{}",
                    lidar.timestamp,
                    lidar.points.len(),
                    lidar.stage_ns,
                    received.saturating_sub(lidar.sent_unix_ns),
                    unix_ns().saturating_sub(lidar.started_unix_ns)
                )?;
                frame += 1;
            }
            Event::InputClosed { .. } | Event::Stop(_) => break,
            _ => {}
        }
    }
    csv.flush()?;
    println!("dora lidar-only: frames={frame}");
    Ok(())
}

fn output_csv() -> Result<BufWriter<File>> {
    let path = env::var("OUTPUT_CSV").unwrap_or_else(|_| "dora-perception.csv".into());
    let mut csv = BufWriter::new(File::create(path)?);
    writeln!(
        csv,
        "frame,timestamp_micros,status,camera_width,camera_height,camera_checksum,points,detections,ground_truth,true_positives,false_positives,false_negatives,depth_matched,camera_stage_ns,inference_ns,lidar_stage_ns,camera_queue_wait_ns,lidar_queue_wait_ns,sync_skew_ns,fusion_ns,end_to_end_ns"
    )?;
    Ok(csv)
}

fn transport_source() -> Result<()> {
    let (mut node, _events) = DoraNode::init_from_env()?;
    let frames = env_usize("TRANSPORT_FRAMES", 199);
    let points = env_usize("TRANSPORT_POINTS", 149_796);
    let base = transport_columns(points);
    let output = DataId::from("payload".to_owned());
    for sequence in 0..frames {
        let started = unix_ns();
        let sent = unix_ns();
        let mut columns = base.clone();
        columns.push((
            "sequence".into(),
            Arc::new(UInt64Array::from(vec![sequence as u64])),
        ));
        columns.push((
            "started_unix_ns".into(),
            Arc::new(UInt64Array::from(vec![started])),
        ));
        columns.push((
            "sent_unix_ns".into(),
            Arc::new(UInt64Array::from(vec![sent])),
        ));
        node.send_output(
            output.clone(),
            MetadataParameters::default(),
            struct_array(columns)?,
        )?;
    }
    Ok(())
}

fn transport_sink() -> Result<()> {
    let (_node, mut events) = DoraNode::init_from_env()?;
    let path = env::var("OUTPUT_CSV").unwrap_or_else(|_| "transport.csv".into());
    let mut csv = BufWriter::new(File::create(path)?);
    writeln!(
        csv,
        "frame,status,points,payload_bytes,queue_wait_ns,end_to_end_ns"
    )?;
    while let Some(event) = events.recv() {
        match event {
            Event::Input { data, .. } => {
                let received = unix_ns();
                let array = data.as_array();
                let value = array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .context("transport message is not StructArray")?;
                let sequence = u64_col(value, "sequence")?;
                let started = u64_col(value, "started_unix_ns")?;
                let sent = u64_col(value, "sent_unix_ns")?;
                let points = value
                    .column_by_name("x")
                    .and_then(|a| a.as_any().downcast_ref::<ListArray>())
                    .context("missing x payload")?
                    .value_length(0);
                writeln!(
                    csv,
                    "{sequence},delivered,{points},{},{},{}",
                    points as usize * 33,
                    received.saturating_sub(sent),
                    unix_ns().saturating_sub(started)
                )?;
            }
            Event::InputClosed { .. } | Event::Stop(_) => break,
            _ => {}
        }
    }
    csv.flush()?;
    Ok(())
}

fn transport_columns(points: usize) -> Vec<(String, ArrayRef)> {
    let offsets = || OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, points as i32]));
    let values: Vec<f32> = (0..points).map(|i| i as f32).collect();
    let list = |name: &str| {
        (
            name.to_owned(),
            Arc::new(ListArray::new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                offsets(),
                Arc::new(Float32Array::from(values.clone())),
                None,
            )) as ArrayRef,
        )
    };
    let mut columns = vec![
        list("x"),
        list("y"),
        list("z"),
        list("range"),
        list("intensity"),
        list("elongation"),
    ];
    columns.push((
        "camera".into(),
        Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int8, false)),
            offsets(),
            Arc::new(Int8Array::from(vec![1_i8; points])),
            None,
        )),
    ));
    columns.push(list("u"));
    columns.push(list("v"));
    columns
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn camera_array(
    predictions: &[WireDetection],
    truth: &[WireDetection],
    meta: CameraArrayMeta,
) -> Result<StructArray> {
    let p = serde_json::to_vec(predictions)?;
    let t = serde_json::to_vec(truth)?;
    struct_array(vec![
        (
            "timestamp",
            Arc::new(Int64Array::from(vec![meta.timestamp])) as ArrayRef,
        ),
        (
            "predictions",
            Arc::new(BinaryArray::from_vec(vec![p.as_slice()])),
        ),
        ("truth", Arc::new(BinaryArray::from_vec(vec![t.as_slice()]))),
        ("stage_ns", Arc::new(UInt64Array::from(vec![meta.stage_ns]))),
        (
            "inference_ns",
            Arc::new(UInt64Array::from(vec![meta.inference_ns])),
        ),
        ("width", Arc::new(UInt32Array::from(vec![meta.width]))),
        ("height", Arc::new(UInt32Array::from(vec![meta.height]))),
        ("checksum", Arc::new(UInt32Array::from(vec![meta.checksum]))),
        (
            "started_unix_ns",
            Arc::new(UInt64Array::from(vec![meta.started_unix_ns])),
        ),
        (
            "sent_unix_ns",
            Arc::new(UInt64Array::from(vec![meta.sent_unix_ns])),
        ),
    ])
}
fn lidar_array(
    timestamp: i64,
    points: &[Point],
    stage: u64,
    started_unix_ns: u64,
    sent_unix_ns: u64,
) -> Result<StructArray> {
    let list = |name: &str, values: Vec<f32>| {
        (
            name.to_owned(),
            Arc::new(ListArray::new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, values.len() as i32])),
                Arc::new(Float32Array::from(values)),
                None,
            )) as ArrayRef,
        )
    };
    struct_array(vec![
        (
            "timestamp".into(),
            Arc::new(Int64Array::from(vec![timestamp])) as ArrayRef,
        ),
        list("x", points.iter().map(|p| p.x).collect()),
        list("y", points.iter().map(|p| p.y).collect()),
        list("z", points.iter().map(|p| p.z).collect()),
        list("range", points.iter().map(|p| p.range).collect()),
        list("intensity", points.iter().map(|p| p.intensity).collect()),
        list("elongation", points.iter().map(|p| p.elongation).collect()),
        (
            "camera".into(),
            Arc::new(ListArray::new(
                Arc::new(Field::new("item", DataType::Int8, false)),
                OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, points.len() as i32])),
                Arc::new(Int8Array::from_iter_values(points.iter().map(|p| p.camera))),
                None,
            )),
        ),
        list("u", points.iter().map(|p| p.u).collect()),
        list("v", points.iter().map(|p| p.v).collect()),
        ("stage_ns".into(), Arc::new(UInt64Array::from(vec![stage]))),
        (
            "started_unix_ns".into(),
            Arc::new(UInt64Array::from(vec![started_unix_ns])),
        ),
        (
            "sent_unix_ns".into(),
            Arc::new(UInt64Array::from(vec![sent_unix_ns])),
        ),
    ])
}
fn struct_array<I, S>(columns: I) -> Result<StructArray>
where
    I: IntoIterator<Item = (S, ArrayRef)>,
    S: Into<String>,
{
    let columns: Vec<_> = columns.into_iter().map(|(n, a)| (n.into(), a)).collect();
    let fields = columns
        .iter()
        .map(|(n, a)| Arc::new(Field::new(n, a.data_type().clone(), false)))
        .collect::<Vec<_>>();
    Ok(StructArray::try_new(
        fields.into(),
        columns.into_iter().map(|(_, a)| a).collect(),
        None,
    )?)
}
fn parse_camera(data: &dyn Array) -> Result<CameraMessage> {
    let s = data
        .as_any()
        .downcast_ref::<StructArray>()
        .context("camera message is not StructArray")?;
    Ok(CameraMessage {
        timestamp: i64_col(s, "timestamp")?,
        predictions: serde_json::from_slice(binary_col(s, "predictions")?)?,
        truth: serde_json::from_slice(binary_col(s, "truth")?)?,
        stage_ns: u64_col(s, "stage_ns")?,
        inference_ns: u64_col(s, "inference_ns")?,
        width: u32_col(s, "width")?,
        height: u32_col(s, "height")?,
        checksum: u32_col(s, "checksum")?,
        started_unix_ns: u64_col(s, "started_unix_ns")?,
        sent_unix_ns: u64_col(s, "sent_unix_ns")?,
    })
}
fn parse_lidar(data: &dyn Array) -> Result<LidarMessage> {
    let s = data
        .as_any()
        .downcast_ref::<StructArray>()
        .context("LiDAR message is not StructArray")?;
    let x = list_col(s, "x")?;
    let y = list_col(s, "y")?;
    let z = list_col(s, "z")?;
    let range = list_col(s, "range")?;
    let intensity = list_col(s, "intensity")?;
    let elongation = list_col(s, "elongation")?;
    let camera = i8_list_col(s, "camera")?;
    let u = list_col(s, "u")?;
    let v = list_col(s, "v")?;
    let points = (0..x.len())
        .map(|i| Point {
            x: x[i],
            y: y[i],
            z: z[i],
            range: range[i],
            intensity: intensity[i],
            elongation: elongation[i],
            camera: camera[i],
            u: u[i],
            v: v[i],
        })
        .collect();
    Ok(LidarMessage {
        timestamp: i64_col(s, "timestamp")?,
        points,
        stage_ns: u64_col(s, "stage_ns")?,
        started_unix_ns: u64_col(s, "started_unix_ns")?,
        sent_unix_ns: u64_col(s, "sent_unix_ns")?,
    })
}
fn i64_col(s: &StructArray, n: &str) -> Result<i64> {
    Ok(s.column_by_name(n)
        .context("missing column")?
        .as_any()
        .downcast_ref::<Int64Array>()
        .context("wrong type")?
        .value(0))
}
fn u64_col(s: &StructArray, n: &str) -> Result<u64> {
    Ok(s.column_by_name(n)
        .context("missing column")?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .context("wrong type")?
        .value(0))
}
fn u32_col(s: &StructArray, n: &str) -> Result<u32> {
    Ok(s.column_by_name(n)
        .context("missing column")?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .context("wrong type")?
        .value(0))
}
fn binary_col<'a>(s: &'a StructArray, n: &str) -> Result<&'a [u8]> {
    Ok(s.column_by_name(n)
        .context("missing column")?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .context("wrong type")?
        .value(0))
}
fn list_col(s: &StructArray, n: &str) -> Result<Vec<f32>> {
    let list = s
        .column_by_name(n)
        .context("missing list")?
        .as_any()
        .downcast_ref::<ListArray>()
        .context("wrong list type")?;
    let value = list.value(0);
    let values = value
        .as_any()
        .downcast_ref::<Float32Array>()
        .context("wrong value type")?;
    Ok(values.values().to_vec())
}
fn i8_list_col(s: &StructArray, n: &str) -> Result<Vec<i8>> {
    let list = s
        .column_by_name(n)
        .context("missing list")?
        .as_any()
        .downcast_ref::<ListArray>()
        .context("wrong list type")?;
    let value = list.value(0);
    let values = value
        .as_any()
        .downcast_ref::<Int8Array>()
        .context("wrong value type")?;
    Ok(values.values().to_vec())
}
fn ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}
fn unix_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
fn write_setup(name: &str, value: u64) -> Result<()> {
    if let Ok(dir) = env::var("OUTPUT_DIR") {
        std::fs::write(
            Path::new(&dir).join(format!("{name}.txt")),
            format!("{name}={value}\n"),
        )?;
    }
    Ok(())
}
fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
impl From<&Detection> for WireDetection {
    fn from(d: &Detection) -> Self {
        Self {
            id: d.id.clone(),
            kind: d.kind.into(),
            score: d.score,
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
        }
    }
}
impl From<&WireDetection> for Detection {
    fn from(d: &WireDetection) -> Self {
        Self {
            id: d.id.clone(),
            kind: match d.kind.as_str() {
                "vehicle" => "vehicle",
                "pedestrian" => "pedestrian",
                "cyclist" => "cyclist",
                "sign" => "sign",
                _ => "unknown",
            },
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
            depth_m: None,
            depth_points: 0,
            score: d.score,
        }
    }
}
