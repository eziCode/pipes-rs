use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, imageops::FilterType};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

const SIZE: usize = 416;
const CLASSES: usize = 80;

pub struct Prediction {
    pub kind: &'static str,
    pub score: f32,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

pub struct YoloxDetector {
    session: Session,
    input_name: String,
    output_name: String,
    confidence: f32,
    nms: f32,
}

impl YoloxDetector {
    pub fn load(path: &Path, confidence: f32, nms: f32) -> Result<Self> {
        anyhow::ensure!(
            path.is_file(),
            "ONNX model does not exist: {}",
            path.display()
        );
        anyhow::ensure!((0.0..=1.0).contains(&confidence));
        anyhow::ensure!((0.0..=1.0).contains(&nms));
        ort::init().with_name("pipes-rs-yolox").commit();
        let builder = Session::builder()?;
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let mut builder = builder
            .with_intra_threads(1)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let session = builder
            .commit_from_file(path)
            .with_context(|| format!("failed to load YOLOX model {}", path.display()))?;
        let input_name = session
            .inputs()
            .first()
            .context("model has no input")?
            .name()
            .to_owned();
        let output_name = session
            .outputs()
            .first()
            .context("model has no output")?
            .name()
            .to_owned();
        Ok(Self {
            session,
            input_name,
            output_name,
            confidence,
            nms,
        })
    }

    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<Prediction>> {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let ratio = (SIZE as f32 / width as f32).min(SIZE as f32 / height as f32);
        let resized_width = (width as f32 * ratio) as u32;
        let resized_height = (height as f32 * ratio) as u32;
        let resized = image
            .resize_exact(resized_width, resized_height, FilterType::Triangle)
            .to_rgb8();
        let mut input = vec![114.0_f32; 3 * SIZE * SIZE];
        for y in 0..resized_height as usize {
            for x in 0..resized_width as usize {
                let pixel = resized.get_pixel(x as u32, y as u32).0;
                let offset = y * SIZE + x;
                input[offset] = pixel[2] as f32;
                input[SIZE * SIZE + offset] = pixel[1] as f32;
                input[2 * SIZE * SIZE + offset] = pixel[0] as f32;
            }
        }
        let tensor = Tensor::from_array((vec![1, 3, SIZE, SIZE], input))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])?;
        let output = outputs[self.output_name.as_str()].try_extract_array::<f32>()?;
        let raw = output
            .as_slice()
            .context("YOLOX output is not contiguous")?;
        let rows = raw.len() / (5 + CLASSES);
        anyhow::ensure!(
            rows * (5 + CLASSES) == raw.len(),
            "unexpected YOLOX output shape"
        );
        let mut candidates = Vec::new();
        let mut row = 0;
        for stride in [8_usize, 16, 32] {
            let grid = SIZE / stride;
            for gy in 0..grid {
                for gx in 0..grid {
                    let values = &raw[row * (5 + CLASSES)..(row + 1) * (5 + CLASSES)];
                    row += 1;
                    let (class, class_score) = values[5..]
                        .iter()
                        .copied()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .unwrap();
                    let score = values[4] * class_score;
                    let Some(kind) = waymo_kind(class) else {
                        continue;
                    };
                    if score < self.confidence {
                        continue;
                    }
                    let cx = (values[0] + gx as f32) * stride as f32 / ratio;
                    let cy = (values[1] + gy as f32) * stride as f32 / ratio;
                    let w = values[2].exp() * stride as f32 / ratio;
                    let h = values[3].exp() * stride as f32 / ratio;
                    candidates.push(Prediction {
                        kind,
                        score,
                        x: (cx - w / 2.0) as f64,
                        y: (cy - h / 2.0) as f64,
                        width: w as f64,
                        height: h as f64,
                    });
                }
            }
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Prediction> = Vec::new();
        for candidate in candidates {
            if kept.iter().all(|other| iou(&candidate, other) <= self.nms) {
                kept.push(candidate);
            }
        }
        Ok(kept)
    }
}

fn waymo_kind(class: usize) -> Option<&'static str> {
    match class {
        0 => Some("pedestrian"),
        1 | 3 => Some("cyclist"),
        2 | 5 | 7 => Some("vehicle"),
        11 => Some("sign"),
        _ => None,
    }
}

fn iou(a: &Prediction, b: &Prediction) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    (intersection / (a.width * a.height + b.width * b.height - intersection).max(f64::EPSILON))
        as f32
}
