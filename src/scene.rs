//! Bounded-memory scene export from the actual staged perception results.
use crate::fusion::{Detection, Point};
use anyhow::{Result, ensure};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub struct SceneWriter {
    directory: PathBuf,
    segment: String,
    model: bool,
    timestamps: Vec<i64>,
}
impl SceneWriter {
    pub fn new(directory: &Path, segment: &str, model: bool) -> Result<Self> {
        fs::create_dir_all(directory)?;
        // Never leave an old completed manifest pointing into an in-progress export.
        let manifest = directory.join("manifest.js");
        if manifest.exists() {
            fs::remove_file(manifest)?;
        }
        Ok(Self {
            directory: directory.into(),
            segment: segment.into(),
            model,
            timestamps: vec![],
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &mut self,
        timestamp: i64,
        jpeg: &[u8],
        width: u32,
        height: u32,
        points: &[Point],
        detections: &[Detection],
        timings: [u64; 8],
    ) -> Result<()> {
        let index = self.timestamps.len();
        let stride = points.len().div_ceil(18000).max(1);
        let rendered: Vec<_> = points
            .iter()
            .step_by(stride)
            .map(|p| [p.x, p.y, p.z, p.u, p.v])
            .collect();
        fs::write(self.directory.join(format!("frame-{index}.jpg")), jpeg)?;
        let payload = json!({"index":index,"timestamp":timestamp,"width":width,"height":height,
            "points": rendered,"totalPoints":points.len(),"boxes":detections,
            "timings":timings.map(|ns| ns as f64 / 1_000_000.0)});
        fs::write(
            self.directory.join(format!("frame-{index}.js")),
            format!("window.receiveFrame({payload});"),
        )?;
        self.timestamps.push(timestamp);
        if index.is_multiple_of(20) {
            eprintln!("scene: exported {} frames", index + 1);
        }
        Ok(())
    }
    pub fn finish(self) -> Result<()> {
        ensure!(
            !self.timestamps.is_empty(),
            "no camera frames available for scene export"
        );
        ensure!(
            self.timestamps.windows(2).all(|w| w[0] < w[1]),
            "scene timestamps must be strictly increasing"
        );
        let manifest = json!({"segment":self.segment,"timestamps":self.timestamps,
            "source":if self.model {"YOLOX ONNX predictions"} else {"Waymo ground-truth labels"}});
        fs::write(
            self.directory.join("manifest.js"),
            format!("window.sceneManifest={manifest};"),
        )?;
        fs::write(
            self.directory.join("index.html"),
            include_str!("../assets/scene.html"),
        )?;
        println!(
            "Scene viewer: {}",
            self.directory.join("index.html").display()
        );
        Ok(())
    }
}
