mod detector;
mod fusion;
mod measurement;
mod metrics;
mod queue;
mod replay;
mod waymo;

use std::{
    path::PathBuf,
    str::FromStr,
    sync::Arc,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use measurement::SensorId;
use metrics::{Delivery, RunMetrics};
use queue::{BoundedQueue, PushResult, QueuePolicy};
use replay::ReplayClock;

#[derive(Debug, Parser)]
#[command(about = "Replay Waymo camera and LiDAR measurements through bounded Arrow queues")]
struct Args {
    /// Waymo v2 segment context name.
    #[arg(long)]
    segment: String,

    /// Dataset split containing the segment.
    #[arg(long, default_value = "training")]
    split: String,

    /// Root created by scripts/download_waymo_sample.py.
    #[arg(long, default_value = "data/waymo-v2-sample")]
    data_root: PathBuf,

    #[arg(long, default_value = "front")]
    camera: String,

    #[arg(long, default_value = "top")]
    lidar: String,

    #[arg(long, default_value_t = 8)]
    queue_size: usize,

    #[arg(long, value_enum, default_value_t = PolicyArg::DropOldest)]
    queue_policy: PolicyArg,

    /// Replay speed. 1.0 is real time, 0 disables pacing.
    #[arg(long, default_value_t = 1.0)]
    speed: f64,

    /// Simulated camera processing time per measurement.
    #[arg(long, default_value_t = 0)]
    camera_work_ms: u64,

    /// Simulated LiDAR processing time per measurement.
    #[arg(long, default_value_t = 0)]
    lidar_work_ms: u64,

    /// Write one delivered/dropped measurement record per CSV row.
    #[arg(long)]
    csv: Option<PathBuf>,

    /// Build a synchronized camera + LiDAR inspection page instead of replaying.
    #[arg(long)]
    demo_output: Option<PathBuf>,

    /// Zero-based front-camera frame used by --demo-output.
    #[arg(long, default_value_t = 0)]
    frame_index: usize,

    /// Run the full staged camera/LiDAR perception workload and write frame metrics.
    #[arg(long)]
    perception_csv: Option<PathBuf>,

    /// Stop a perception run after this many frames; 0 processes the whole segment.
    #[arg(long, default_value_t = 0)]
    frame_limit: usize,

    /// Capacity of each processing-stage channel.
    #[arg(long, default_value_t = 4)]
    stage_queue_size: usize,

    /// YOLOX ONNX model. When omitted, Waymo labels provide the validation baseline.
    #[arg(long)]
    onnx_model: Option<PathBuf>,

    #[arg(long, default_value_t = 0.3)]
    confidence_threshold: f32,

    #[arg(long, default_value_t = 0.45)]
    nms_threshold: f32,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PolicyArg {
    Backpressure,
    DropOldest,
}

impl From<PolicyArg> for QueuePolicy {
    fn from(value: PolicyArg) -> Self {
        match value {
            PolicyArg::Backpressure => Self::Backpressure,
            PolicyArg::DropOldest => Self::DropOldest,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(output) = &args.perception_csv {
        anyhow::ensure!(
            args.stage_queue_size > 0,
            "--stage-queue-size must be at least 1"
        );
        let summary = fusion::run_segment_pipeline(
            &args.data_root,
            &args.split,
            &args.segment,
            fusion::PerceptionConfig {
                frame_limit: args.frame_limit,
                queue_size: args.stage_queue_size,
                onnx_model: args.onnx_model.as_deref(),
                confidence: args.confidence_threshold,
                nms: args.nms_threshold,
            },
            output,
        )?;
        println!(
            "perception: frames={} points={} detections={} depth_matched={} elapsed={:.2}s rate={:.2}Hz",
            summary.frames,
            summary.points,
            summary.detections,
            summary.depth_matched,
            summary.elapsed.as_secs_f64(),
            summary.frames as f64 / summary.elapsed.as_secs_f64().max(f64::EPSILON),
        );
        let precision = summary.true_positives as f64
            / (summary.true_positives + summary.false_positives).max(1) as f64;
        let recall = summary.true_positives as f64
            / (summary.true_positives + summary.false_negatives).max(1) as f64;
        println!(
            "evaluation@IoU0.5: tp={} fp={} fn={} precision={:.3} recall={:.3}",
            summary.true_positives,
            summary.false_positives,
            summary.false_negatives,
            precision,
            recall
        );
        println!("frame metrics: {}", output.display());
        println!(
            "detection source: {}",
            if args.onnx_model.is_some() {
                "YOLOX ONNX predictions"
            } else {
                "Waymo ground truth"
            }
        );
        return Ok(());
    }

    if let Some(output) = &args.demo_output {
        let summary = fusion::build_demo(
            &args.data_root,
            &args.split,
            &args.segment,
            args.frame_index,
            output,
        )?;
        println!("fusion demo: {}", summary.html.display());
        println!("Arrow point cloud: {}", summary.arrow.display());
        println!(
            "frame={} timestamp={} points={} boxes={} boxes_with_depth={}",
            args.frame_index,
            summary.timestamp_micros,
            summary.points,
            summary.boxes,
            summary.boxes_with_depth
        );
        return Ok(());
    }
    anyhow::ensure!(args.speed >= 0.0, "--speed must be non-negative");
    anyhow::ensure!(args.queue_size > 0, "--queue-size must be at least 1");

    let camera = SensorId::from_str(&format!("camera:{}", args.camera))?;
    let lidar = SensorId::from_str(&format!("lidar:{}", args.lidar))?;
    let mut measurements =
        waymo::load_segment(&args.data_root, &args.split, &args.segment, &camera, &lidar)?;
    anyhow::ensure!(
        !measurements.is_empty(),
        "the selected files contained no matching measurements"
    );
    measurements.sort_by_key(|m| (m.sensor_timestamp_ns, m.sensor));

    let camera_queue = Arc::new(BoundedQueue::new(args.queue_size, args.queue_policy.into()));
    let lidar_queue = Arc::new(BoundedQueue::new(args.queue_size, args.queue_policy.into()));
    let metrics = Arc::new(RunMetrics::default());

    let camera_worker = spawn_consumer(
        camera,
        Arc::clone(&camera_queue),
        Arc::clone(&metrics),
        Duration::from_millis(args.camera_work_ms),
    );
    let lidar_worker = spawn_consumer(
        lidar,
        Arc::clone(&lidar_queue),
        Arc::clone(&metrics),
        Duration::from_millis(args.lidar_work_ms),
    );

    println!(
        "replaying {} measurements at {}x with queue_size={} policy={:?}",
        measurements.len(),
        args.speed,
        args.queue_size,
        args.queue_policy
    );
    let clock = ReplayClock::new(measurements[0].sensor_timestamp_ns, args.speed);
    for mut measurement in measurements {
        clock.wait_until(measurement.sensor_timestamp_ns);
        measurement.due_at = clock.due_at(measurement.sensor_timestamp_ns);
        measurement.enqueued_at = std::time::Instant::now();
        measurement.host_arrival_unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .min(u64::MAX as u128) as u64;
        let sensor = measurement.sensor;
        metrics.for_sensor(sensor).record_submitted();
        let queue = match sensor.kind() {
            measurement::SensorKind::Camera => &camera_queue,
            measurement::SensorKind::Lidar => &lidar_queue,
        };
        match queue.push(measurement) {
            PushResult::Accepted { depth, dropped } => {
                metrics.for_sensor(sensor).record_accepted(depth);
                if let Some(dropped) = dropped {
                    metrics.for_sensor(dropped.sensor).record_dropped(
                        dropped.sequence,
                        dropped.sensor_timestamp_ns,
                        dropped.host_arrival_unix_ns,
                    );
                }
            }
            PushResult::Closed => break,
        }
    }

    camera_queue.close();
    lidar_queue.close();
    camera_worker.join().expect("camera worker panicked")?;
    lidar_worker.join().expect("LiDAR worker panicked")?;

    metrics.print_summary();
    if let Some(path) = &args.csv {
        metrics.write_csv(path)?;
        println!("measurement records: {}", path.display());
    }
    Ok(())
}

fn spawn_consumer(
    sensor: SensorId,
    queue: Arc<BoundedQueue<measurement::Measurement>>,
    metrics: Arc<RunMetrics>,
    work: Duration,
) -> thread::JoinHandle<Result<()>> {
    thread::Builder::new()
        .name(sensor.to_string())
        .spawn(move || {
            while let Some(measurement) = queue.pop() {
                let received_at = std::time::Instant::now();
                let queue_wait = received_at.saturating_duration_since(measurement.enqueued_at);
                let age = received_at.saturating_duration_since(measurement.due_at);
                let encoded = measurement.encode_ipc().with_context(|| {
                    format!(
                        "failed to encode {} sequence {}",
                        sensor, measurement.sequence
                    )
                })?;
                let checksum = crc32fast::hash(&encoded);
                if !work.is_zero() {
                    thread::sleep(work);
                }
                metrics.for_sensor(sensor).record_delivered(Delivery {
                    sequence: measurement.sequence,
                    sensor_timestamp_ns: measurement.sensor_timestamp_ns,
                    host_arrival_unix_ns: measurement.host_arrival_unix_ns,
                    bytes: encoded.len() as u64,
                    queue_wait,
                    age,
                    checksum,
                });
            }
            Ok(())
        })
        .expect("failed to spawn sensor worker")
}
