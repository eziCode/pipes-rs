use anyhow::{Context, Result};
use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, Float32Array, Int64Array, ListArray, StructArray, UInt64Array,
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
    time::Instant,
};

#[derive(Clone, Copy, ValueEnum)]
enum Role {
    Camera,
    Lidar,
    Fusion,
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
}
struct LidarMessage {
    timestamp: i64,
    points: Vec<Point>,
    stage_ns: u64,
}

fn main() -> Result<()> {
    match Args::parse().role {
        Role::Camera => camera(),
        Role::Lidar => lidar(),
        Role::Fusion => fusion_node(),
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
    let output = DataId::from("detections".to_owned());
    fusion::visit_camera_frames(
        &fusion::component_path(&root, &split, "camera_image", &segment),
        limit,
        |frame| {
            let started = Instant::now();
            let image = image::load_from_memory(&frame.jpeg)?;
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
            let message = camera_array(
                frame.timestamp,
                &predictions,
                &truth,
                ns(started),
                inference_ns,
            )?;
            node.send_output(output.clone(), MetadataParameters::default(), message)?;
            Ok(())
        },
    )
}
fn lidar() -> Result<()> {
    let (mut node, _events) = DoraNode::init_from_env()?;
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
    let output = DataId::from("pointcloud".to_owned());
    fusion::visit_top_lidar_frames(
        &fusion::component_path(&root, &split, "lidar", &segment),
        "[LiDARComponent]",
        limit,
        |frame| {
            let started = Instant::now();
            let projection = projections
                .get(&frame.timestamp)
                .context("missing LiDAR camera projection")?;
            let cloud = fusion::convert_top_lidar(&frame, projection, &calibration)?;
            node.send_output(
                output.clone(),
                MetadataParameters::default(),
                lidar_array(frame.timestamp, &cloud.points, ns(started))?,
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
        "frame,timestamp_micros,points,detections,ground_truth,true_positives,false_positives,false_negatives,depth_matched,camera_stage_ns,inference_ns,lidar_stage_ns,dora_fusion_ns,end_to_end_ns"
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
            let fuse = Instant::now();
            let mut predicted: Vec<Detection> =
                camera.predictions.iter().map(Detection::from).collect();
            let truth: Vec<Detection> = camera.truth.iter().map(Detection::from).collect();
            fusion::add_depths(&mut predicted, &lidar.points);
            let (tp, fp, fn_) = fusion::evaluate_detections(&predicted, &truth, 0.5);
            let depth = predicted.iter().filter(|d| d.depth_m.is_some()).count();
            writeln!(
                csv,
                "{frame},{timestamp},{},{},{},{tp},{fp},{fn_},{depth},{},{},{},{},{}",
                lidar.points.len(),
                predicted.len(),
                truth.len(),
                camera.stage_ns,
                camera.inference_ns,
                lidar.stage_ns,
                ns(fuse),
                ns(run)
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

fn camera_array(
    timestamp: i64,
    predictions: &[WireDetection],
    truth: &[WireDetection],
    stage: u64,
    inference: u64,
) -> Result<StructArray> {
    let p = serde_json::to_vec(predictions)?;
    let t = serde_json::to_vec(truth)?;
    struct_array(vec![
        (
            "timestamp",
            Arc::new(Int64Array::from(vec![timestamp])) as ArrayRef,
        ),
        (
            "predictions",
            Arc::new(BinaryArray::from_vec(vec![p.as_slice()])),
        ),
        ("truth", Arc::new(BinaryArray::from_vec(vec![t.as_slice()]))),
        ("stage_ns", Arc::new(UInt64Array::from(vec![stage]))),
        ("inference_ns", Arc::new(UInt64Array::from(vec![inference]))),
    ])
}
fn lidar_array(timestamp: i64, points: &[Point], stage: u64) -> Result<StructArray> {
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
        list("u", points.iter().map(|p| p.u).collect()),
        list("v", points.iter().map(|p| p.v).collect()),
        ("stage_ns".into(), Arc::new(UInt64Array::from(vec![stage]))),
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
    let u = list_col(s, "u")?;
    let v = list_col(s, "v")?;
    let points = (0..x.len())
        .map(|i| Point {
            x: x[i],
            y: y[i],
            z: z[i],
            range: range[i],
            intensity: 0.0,
            elongation: 0.0,
            camera: if u[i] >= 0.0 { 1 } else { 0 },
            u: u[i],
            v: v[i],
        })
        .collect();
    Ok(LidarMessage {
        timestamp: i64_col(s, "timestamp")?,
        points,
        stage_ns: u64_col(s, "stage_ns")?,
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
fn ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
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
