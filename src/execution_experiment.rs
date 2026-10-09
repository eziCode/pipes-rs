//! C ABI shared by the native-channel and composed-ROS execution pilot.
//! Context and result handles are opaque and remain owned by this library.
use crate::{arrow_experiment::check_depths, fusion::*, pilot::monotonic_ns};
use anyhow::{Context, Result, ensure};
use std::{
    ffi::{CStr, c_char},
    fs::File,
    io::{BufWriter, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::mpsc::sync_channel,
};
struct Frame {
    points: Vec<Point>,
    boxes: Vec<Detection>,
    expected: Vec<Detection>,
    timestamp: i64,
}
pub struct Dataset {
    frames: Vec<Frame>,
}
pub struct Output {
    detections: Vec<Detection>,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Sample {
    pub index: u64,
    pub started: u64,
    pub received: u64,
    pub finished: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Task {
    index: u64,
    started: u64,
    context: usize,
}
fn load(root: &Path, split: &str, segment: &str, limit: usize) -> Result<Dataset> {
    ensure!(limit > 0, "positive frame limit required");
    let path = |c| component_path(root, split, c, segment);
    let calibration = read_calibration(&path("lidar_calibration"))?;
    let projections = read_all_top_lidar_frames(
        &path("lidar_camera_projection"),
        "[LiDARCameraProjectionComponent]",
        limit,
    )?;
    let boxes = read_all_boxes(&path("camera_box"))?;
    let mut frames = Vec::new();
    visit_top_lidar_frames(&path("lidar"), "[LiDARComponent]", limit, |f| {
        let points = convert_top_lidar_points(
            &f,
            projections
                .get(&f.timestamp)
                .context("missing projection")?,
            &calibration,
        )?;
        let boxes = boxes.get(&f.timestamp).cloned().unwrap_or_default();
        let mut expected = boxes.clone();
        add_depths(&mut expected, &points);
        frames.push(Frame {
            points,
            boxes,
            expected,
            timestamp: f.timestamp,
        });
        Ok(())
    })?;
    ensure!(frames.len() == limit, "frame limit not satisfied");
    ensure!(
        frames.iter().any(|f| !f.boxes.is_empty()),
        "no annotation boxes"
    );
    Ok(Dataset { frames })
}
fn fuse(dataset: &Dataset, index: usize) -> Result<Output> {
    let frame = dataset.frames.get(index).context("invalid frame index")?;
    let mut detections = frame.boxes.clone();
    add_depths(&mut detections, &frame.points);
    Ok(Output { detections })
}
fn guard<T>(fallback: T, f: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            eprintln!("execution pilot: {e:#}");
            fallback
        }
        Err(_) => {
            eprintln!("execution pilot panicked");
            fallback
        }
    }
}
// All callers must pass valid live handles, strings, and output buffers. No Rust
// references escape the ABI; context is immutable until both execution threads stop.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn px_load(
    root: *const c_char,
    split: *const c_char,
    segment: *const c_char,
    frames: usize,
) -> *mut Dataset {
    guard(std::ptr::null_mut(), || unsafe {
        ensure!(
            !root.is_null() && !split.is_null() && !segment.is_null(),
            "null string"
        );
        Ok(Box::into_raw(Box::new(load(
            Path::new(CStr::from_ptr(root).to_str()?),
            CStr::from_ptr(split).to_str()?,
            CStr::from_ptr(segment).to_str()?,
            frames,
        )?)))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn px_free(dataset: *mut Dataset) {
    if !dataset.is_null() {
        unsafe {
            drop(Box::from_raw(dataset));
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn px_now() -> u64 {
    monotonic_ns()
}
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn px_fuse(dataset: *const Dataset, index: usize) -> *mut Output {
    guard(std::ptr::null_mut(), || unsafe {
        ensure!(!dataset.is_null(), "null context");
        Ok(Box::into_raw(Box::new(fuse(&*dataset, index)?)))
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn px_validate_free(
    dataset: *const Dataset,
    index: usize,
    output: *mut Output,
) -> i32 {
    guard(-1, || unsafe {
        ensure!(!output.is_null(), "null output");
        let output = Box::from_raw(output);
        ensure!(!dataset.is_null(), "null context");
        let frame = (&*dataset)
            .frames
            .get(index)
            .context("invalid frame index")?;
        check_depths(&frame.expected, &output.detections)?;
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn px_native(
    dataset: *const Dataset,
    capacity: usize,
    samples: *mut Sample,
    len: usize,
) -> i32 {
    guard(-1, || unsafe {
        ensure!(
            !dataset.is_null() && !samples.is_null() && capacity > 0,
            "invalid arguments"
        );
        let data = &*dataset;
        ensure!(len == data.frames.len(), "wrong sample count");
        let (tx, rx) = sync_channel::<Task>(capacity);
        let (ack_tx, ack_rx) = sync_channel::<()>(capacity);
        let collected = std::thread::scope(|scope| -> Result<Vec<Sample>> {
            let worker = scope.spawn(move || -> Result<Vec<Sample>> {
                let mut rows = Vec::with_capacity(len);
                for task in rx {
                    let received = px_now();
                    ensure!(
                        task.context == data as *const Dataset as usize,
                        "context changed"
                    );
                    let result = px_fuse(data, task.index as usize);
                    let finished = px_now();
                    ensure!(
                        px_validate_free(data, task.index as usize, result) == 0,
                        "output validation failed"
                    );
                    rows.push(Sample {
                        index: task.index,
                        started: task.started,
                        received,
                        finished,
                    });
                    ack_tx.send(())?;
                }
                Ok(rows)
            });
            for index in 0..len {
                if index >= capacity {
                    ack_rx.recv()?;
                }
                tx.send(Task {
                    index: index as u64,
                    started: px_now(),
                    context: data as *const Dataset as usize,
                })?;
            }
            drop(tx);
            // Keep the receiver live until all outstanding results complete.
            for _ in 0..capacity.min(len) {
                ack_rx.recv()?;
            }
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("native worker panic"))?
        })?;
        ensure!(collected.len() == len, "incomplete native run");
        std::ptr::copy_nonoverlapping(collected.as_ptr(), samples, len);
        Ok(0)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn px_write(
    dataset: *const Dataset,
    samples: *const Sample,
    len: usize,
    path: *const c_char,
) -> i32 {
    guard(-1, || unsafe {
        ensure!(
            !dataset.is_null() && !samples.is_null() && !path.is_null(),
            "null argument"
        );
        let data = &*dataset;
        ensure!(len == data.frames.len(), "incomplete output");
        let rows = std::slice::from_raw_parts(samples, len);
        let mut writer = BufWriter::new(File::create(CStr::from_ptr(path).to_str()?)?);
        writeln!(
            writer,
            "index,timestamp,points,boxes,started_ns,received_ns,finished_ns,handoff_ns,fusion_ns,result_ns"
        )?;
        for (i, s) in rows.iter().enumerate() {
            ensure!(
                s.index == i as u64 && s.started <= s.received && s.received <= s.finished,
                "invalid timing/order"
            );
            let f = &data.frames[i];
            writeln!(
                writer,
                "{i},{},{},{},{},{},{},{},{},{}",
                f.timestamp,
                f.points.len(),
                f.boxes.len(),
                s.started,
                s.received,
                s.finished,
                s.received - s.started,
                s.finished - s.received,
                s.finished - s.started
            )?;
        }
        writer.flush()?;
        Ok(0)
    })
}
// Keep C ABI layout testable without ROS or dataset files.
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_completion_credit_and_shared_kernel() {
        let d = Dataset {
            frames: (0..9)
                .map(|i| Frame {
                    points: vec![],
                    boxes: vec![],
                    expected: vec![],
                    timestamp: i,
                })
                .collect(),
        };
        for capacity in [1, 4, 16] {
            let mut rows = vec![
                Sample {
                    index: 0,
                    started: 0,
                    received: 0,
                    finished: 0
                };
                9
            ];
            assert_eq!(unsafe { px_native(&d, capacity, rows.as_mut_ptr(), 9) }, 0);
            assert!(rows.iter().enumerate().all(|(i, r)| r.index == i as u64
                && r.started <= r.received
                && r.received <= r.finished));
        }
    }
    #[test]
    fn ffi_rejects_bad_counts_and_nulls() {
        assert!(unsafe { px_fuse(std::ptr::null(), 0) }.is_null());
        assert_eq!(std::mem::size_of::<Sample>(), 32);
    }
}
