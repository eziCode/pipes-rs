use anyhow::{Context, Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use clap::Parser;
use pipes_rs::{detector::YoloxDetector, fusion::*};
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    data_root: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "training")]
    split: String,
    #[arg(long, default_value = "10023947602400723454_1120_000_1140_000")]
    segment: String,
    #[arg(long, default_value_t = 8000)]
    render_points: usize,
}
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.
}
fn round(v: f32) -> f32 {
    (v * 100.).round() / 100.
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.render_points > 0, "render point limit must be positive");
    ensure!(
        !a.output.exists(),
        "output already exists; choose a new export folder"
    );
    fs::create_dir_all(a.output.join("frames"))?;
    let path = |c| component_path(&a.data_root, &a.split, c, &a.segment);
    let calibration = read_calibration(&path("lidar_calibration"))?;
    let mut lidar = read_all_top_lidar_frames(&path("lidar"), "[LiDARComponent]", 0)?;
    let mut projection = read_all_top_lidar_frames(
        &path("lidar_camera_projection"),
        "[LiDARCameraProjectionComponent]",
        0,
    )?;
    let boxes = read_all_boxes(&path("camera_box"))?;
    let mut detector = YoloxDetector::load(&a.model, 0.3, 0.45)?;
    let mut timeline = Vec::new();
    let mut first = None;
    let mut previous = None;
    visit_camera_frames(&path("camera_image"), 0, |camera| {
        let index = timeline.len();
        if let Some(p) = previous {
            ensure!(
                camera.timestamp > p,
                "camera timestamps not strictly increasing"
            );
        }
        previous = Some(camera.timestamp);
        let origin = *first.get_or_insert(camera.timestamp);
        let t_ms = (camera.timestamp - origin) as f64 / 1000.;
        let t = Instant::now();
        let image = image::load_from_memory(&camera.jpeg)?;
        let decode_ms = ms(t);
        let t = Instant::now();
        let mut detections: Vec<Detection> = detector
            .detect(&image)?
            .into_iter()
            .enumerate()
            .map(|(i, p)| Detection {
                id: format!("frame-{index}-box-{i}"),
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
        let inference_ms = ms(t);
        let t = Instant::now();
        let lf = lidar
            .remove(&camera.timestamp)
            .context("missing timestamp-matched LiDAR")?;
        let pf = projection
            .remove(&camera.timestamp)
            .context("missing timestamp-matched projection")?;
        let points = convert_top_lidar_points(&lf, &pf, &calibration)?;
        let lidar_ms = ms(t);
        let t = Instant::now();
        add_depths(&mut detections, &points);
        let fusion_ms = ms(t);
        let step = points.len().div_ceil(a.render_points).max(1);
        let render: Vec<[f32; 6]> = points
            .iter()
            .step_by(step)
            .filter(|p| {
                p.x.is_finite()
                    && p.y.is_finite()
                    && p.range.is_finite()
                    && p.u.is_finite()
                    && p.v.is_finite()
            })
            .map(|p| {
                [
                    round(p.x),
                    round(p.y),
                    round(p.range),
                    p.camera as f32,
                    round(p.u),
                    round(p.v),
                ]
            })
            .collect();
        let thumbnail = image.resize(1280, 900, image::imageops::FilterType::Triangle);
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 82)
            .encode_image(&thumbnail)?;
        let data = json!({"index":index,"t_ms":t_ms,"timestamp_micros":camera.timestamp,
    "width":image.width(),"height":image.height(),"jpeg":STANDARD.encode(&jpeg),
    "points":render,"total_points":points.len(),"detections":detections,
    "annotations":boxes.get(&camera.timestamp).cloned().unwrap_or_default(),
    "timing":{"decode_ms":decode_ms,"inference_ms":inference_ms,"lidar_ms":lidar_ms,"fusion_ms":fusion_ms}});
        fs::write(
            a.output.join(format!("frames/{index:04}.js")),
            format!(
                "window.receiveSceneFrame({});\n",
                serde_json::to_string(&data)?
            ),
        )?;
        timeline.push(json!({"index":index,"t_ms":t_ms,"timestamp_micros":camera.timestamp,"detections":detections.len()}));
        println!(
            "exported frame {index}: {t_ms:.0} ms, {} detections",
            detections.len()
        );
        Ok(())
    })?;
    ensure!(timeline.len() > 1, "not enough frames for playback");
    let manifest = json!({"version":1,"segment":a.segment,"split":a.split,"frames":timeline,
  "model":"YOLOX Nano","confidence":0.3,"nms":0.45,"inference":"offline CPU inference",
  "description":"Entire front-camera sequence with timestamp-matched top LiDAR. Display points are downsampled; fusion uses all points. Detections are per-frame, not tracked objects."});
    fs::write(
        a.output.join("scene.js"),
        format!("window.SCENE={};\n", serde_json::to_string(&manifest)?),
    )?;
    fs::write(
        a.output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::write(
        a.output.join("index.html"),
        include_str!("../../assets/scene-player.html"),
    )?;
    fs::write(
        a.output.join("README.txt"),
        "WAYMO SCENE PLAYER\nExtract the whole ZIP, then open index.html in Chrome, Edge, Firefox or Safari. Keep frames/ and scene.js beside it. No server or internet is needed.\n\nSpace: play/pause. Left/right arrows: previous/next frame. Drag the timeline or scroll over the time wheel to scrub. Select a detection to highlight its projected LiDAR points.\n\nThis is replay of precomputed YOLOX detections, not live browser inference. The camera and top LiDAR are synchronized by dataset timestamp. Rendering is downsampled; fusion uses all points. Depth labels are median LiDAR range, not forward-axis distance. Waymo points here are not motion compensated. Boxes are per-frame detections, not tracked identities. Stage times are offline export timings, not benchmark results.\n",
    )?;
    println!(
        "Completed {} frames: {}",
        timeline.len(),
        a.output.display()
    );
    Ok(())
}
