//! Matched depth-association kernels for a representation experiment.
use crate::fusion::{Detection, Point};
use anyhow::{Context, Result, ensure};
use arrow::{
    array::{Array, Float32Array, Int8Array},
    record_batch::RecordBatch,
};

/// All nine fields are retained, matching the Arrow point schema.
pub struct Columns {
    pub floats: [Vec<f32>; 8],
    pub camera: Vec<i8>,
}
impl Columns {
    pub fn from_points(points: &[Point]) -> Self {
        let fields: [fn(&Point) -> f32; 8] = [
            |p| p.x,
            |p| p.y,
            |p| p.z,
            |p| p.range,
            |p| p.intensity,
            |p| p.elongation,
            |p| p.u,
            |p| p.v,
        ];
        Self {
            floats: fields.map(|f| points.iter().map(f).collect()),
            camera: points.iter().map(|p| p.camera).collect(),
        }
    }
    pub fn fuse(&self, detections: &mut [Detection]) {
        associate(detections, self.camera.len(), |i| {
            (
                self.camera[i],
                self.floats[6][i],
                self.floats[7][i],
                self.floats[3][i],
            )
        });
    }
}
fn associate(
    detections: &mut [Detection],
    len: usize,
    point: impl Fn(usize) -> (i8, f32, f32, f32),
) {
    for d in detections {
        let mut ranges: Vec<f32> = (0..len)
            .filter_map(|i| {
                let (camera, u, v, range) = point(i);
                (camera == 1
                    && u as f64 >= d.x
                    && u as f64 <= d.x + d.width
                    && v as f64 >= d.y
                    && v as f64 <= d.y + d.height
                    && range.is_finite())
                .then_some(range)
            })
            .collect();
        ranges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        d.depth_points = ranges.len();
        d.depth_m = (!ranges.is_empty()).then(|| ranges[ranges.len() / 2]);
    }
}
pub fn fuse_native(d: &mut [Detection], points: &[Point]) {
    associate(d, points.len(), |i| {
        let p = &points[i];
        (p.camera, p.u, p.v, p.range)
    });
}
pub fn fuse_arrow(d: &mut [Detection], batch: &RecordBatch) -> Result<()> {
    fn float<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a Float32Array> {
        let a = b
            .column_by_name(name)
            .with_context(|| format!("missing {name}"))?
            .as_any()
            .downcast_ref::<Float32Array>()
            .context("expected Float32")?;
        ensure!(a.null_count() == 0, "null point fields unsupported");
        Ok(a)
    }
    let camera = batch
        .column_by_name("front_camera")
        .context("missing camera")?
        .as_any()
        .downcast_ref::<Int8Array>()
        .context("expected Int8 camera")?;
    ensure!(camera.null_count() == 0, "null camera unsupported");
    let u = float(batch, "front_u")?;
    let v = float(batch, "front_v")?;
    let range = float(batch, "range")?;
    associate(d, batch.num_rows(), |i| {
        (camera.value(i), u.value(i), v.value(i), range.value(i))
    });
    Ok(())
}
pub fn check_depths(expected: &[Detection], actual: &[Detection]) -> Result<()> {
    ensure!(expected.len() == actual.len(), "detection count mismatch");
    for (a, b) in expected.iter().zip(actual) {
        ensure!(
            a.id == b.id
                && a.depth_points == b.depth_points
                && a.depth_m.map(f32::to_bits) == b.depth_m.map(f32::to_bits),
            "depth mismatch for {}",
            a.id
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::{add_depths, point_batch};
    fn boxes() -> Vec<Detection> {
        vec![Detection {
            id: "box".into(),
            kind: "vehicle",
            x: 0.,
            y: 0.,
            width: 10.,
            height: 10.,
            depth_m: None,
            depth_points: 0,
            score: None,
        }]
    }
    fn point(camera: i8, u: f32, v: f32, range: f32) -> Point {
        Point {
            x: 0.,
            y: 0.,
            z: 0.,
            range,
            intensity: 0.,
            elongation: 0.,
            camera,
            u,
            v,
        }
    }
    #[test]
    fn representations_match_reference_on_boundaries_and_invalid_ranges() {
        let points = vec![
            point(1, 0., 0., 2.),
            point(1, 10., 10., 4.),
            point(2, 5., 5., 1.),
            point(1, 11., 5., 1.),
            point(1, 5., 5., f32::NAN),
            point(1, 5., 5., f32::INFINITY),
            point(1, f32::NAN, 5., 1.),
        ];
        let mut expected = boxes();
        add_depths(&mut expected, &points);
        assert_eq!(expected[0].depth_points, 2);
        assert_eq!(expected[0].depth_m, Some(4.));
        let mut native = boxes();
        fuse_native(&mut native, &points);
        check_depths(&expected, &native).unwrap();
        let mut cols = boxes();
        Columns::from_points(&points).fuse(&mut cols);
        check_depths(&expected, &cols).unwrap();
        let mut arrow = boxes();
        fuse_arrow(&mut arrow, &point_batch(&points).unwrap()).unwrap();
        check_depths(&expected, &arrow).unwrap();
    }
    #[test]
    fn empty_and_sliced_arrow_match_native() {
        let points = vec![point(1, 5., 5., 99.), point(1, 5., 5., 3.)];
        for (start, len) in [(1, 1), (0, 0)] {
            let mut expected = boxes();
            add_depths(&mut expected, &points[start..start + len]);
            let mut actual = boxes();
            fuse_arrow(
                &mut actual,
                &point_batch(&points).unwrap().slice(start, len),
            )
            .unwrap();
            check_depths(&expected, &actual).unwrap();
        }
    }
}
