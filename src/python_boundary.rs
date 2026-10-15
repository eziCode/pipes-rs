//! In-process Rust -> Python representation experiment. No ROS/executor changes.
//! Rust streams one timestamp bundle, owns native columns and optionally builds
//! Arrow columns. Python imports those buffers and runs the same model/fusion.
//!
//! # FFI safety
//! Callers must supply live handles and valid output pointers. Source handles are
//! single-threaded. Native views must not outlive their frame; imported Arrow
//! arrays own a reference to their buffers independently of the frame.
use crate::fusion::{self, Calibration};
use anyhow::{Context, Result, ensure};
use arrow::{
    array::{Array, ArrayRef, Float32Array, UInt8Array},
    buffer::{Buffer as ArrowBuffer, ScalarBuffer},
    ffi::{FFI_ArrowArray, FFI_ArrowSchema, to_ffi},
    record_batch::RecordBatch,
};
use serde::Deserialize;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    ffi::{CStr, CString, c_char},
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::Arc,
};

const FIELDS: [&str; 14] = [
    "x",
    "y",
    "z",
    "range",
    "intensity",
    "elongation",
    "camera1",
    "u1",
    "v1",
    "camera2",
    "u2",
    "v2",
    "lidar",
    "return",
];
#[derive(Deserialize)]
pub struct Config {
    pub data_root: String,
    pub split: String,
    pub segment: String,
    pub cameras: Vec<i64>,
    pub lidars: Vec<i64>,
    pub returns: Vec<usize>,
    pub frame_limit: usize,
}
struct Groups {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    batch: Option<RecordBatch>,
    row: usize,
    pending: Option<RecordBatch>,
    previous: Option<i64>,
}
impl Groups {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            reader: fusion::reader(path)?,
            batch: None,
            row: 0,
            pending: None,
            previous: None,
        })
    }
    fn row(&mut self) -> Result<Option<RecordBatch>> {
        loop {
            if let Some(batch) = &self.batch
                && self.row < batch.num_rows()
            {
                let row = batch.slice(self.row, 1);
                self.row += 1;
                return Ok(Some(row));
            }
            self.batch = self.reader.next().transpose()?;
            self.row = 0;
            if self.batch.is_none() {
                return Ok(None);
            }
        }
    }
    fn next(&mut self) -> Result<Option<(i64, Vec<RecordBatch>)>> {
        let first = if let Some(row) = self.pending.take() {
            Some(row)
        } else {
            self.row()?
        };
        let Some(first) = first else {
            return Ok(None);
        };
        let ts = fusion::int64(&first, "key.frame_timestamp_micros")?.value(0);
        ensure!(
            self.previous.is_none_or(|p| p < ts),
            "component timestamps are not increasing"
        );
        self.previous = Some(ts);
        let mut rows = vec![first];
        while let Some(row) = self.row()? {
            if fusion::int64(&row, "key.frame_timestamp_micros")?.value(0) != ts {
                self.pending = Some(row);
                break;
            }
            rows.push(row);
        }
        Ok(Some((ts, rows)))
    }
}
pub struct Source {
    config: Config,
    cameras: Groups,
    lidars: Groups,
    projections: Groups,
    calibrations: BTreeMap<i64, Calibration>,
    count: usize,
}
#[derive(Clone)]
enum Values {
    F32(Vec<f32>),
    U8(Vec<u8>),
    Mapped(ArrowBuffer, u32, usize),
}
#[derive(Clone)]
struct Buffer {
    name: String,
    shape: Vec<usize>,
    values: Values,
}
#[derive(Clone)]
pub struct Frame {
    buffers: Vec<Buffer>,
    arrow: Option<Vec<ArrayRef>>,
    native_copy: Option<Vec<Arc<NativeAllocation>>>,
    metadata: CString,
}

// Plain Rust-owned bytes, with the same allocation alignment and capacity
// rounding as ArrowBuffer::from_slice_ref. No Arrow arrays or C Data exports.
struct NativeAllocation {
    address: usize,
    layout: std::alloc::Layout,
}
impl NativeAllocation {
    fn copy(bytes: &[u8]) -> Result<Self> {
        let capacity = bytes.len().checked_add(63).context("capacity overflow")? & !63;
        let layout = std::alloc::Layout::from_size_align(capacity, arrow::alloc::ALIGNMENT)?;
        let address = if capacity == 0 {
            layout.align()
        } else {
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
            ptr as usize
        };
        Ok(Self { address, layout })
    }
}
impl Drop for NativeAllocation {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            unsafe { std::alloc::dealloc(self.address as *mut u8, self.layout) };
        }
    }
}
impl Source {
    pub fn open(config: Config) -> Result<Self> {
        for ids in [&config.cameras, &config.lidars] {
            ensure!(
                !ids.is_empty() && ids.iter().all(|i| (1..=5).contains(i)),
                "sensor IDs must be 1..5"
            );
            let mut unique = ids.to_vec();
            unique.sort();
            unique.dedup();
            ensure!(unique.len() == ids.len(), "duplicate sensor IDs");
        }
        ensure!(
            !config.returns.is_empty() && config.returns.iter().all(|r| *r == 1 || *r == 2),
            "returns must be 1 or 2"
        );
        let mut unique = config.returns.clone();
        unique.sort();
        unique.dedup();
        ensure!(unique.len() == config.returns.len(), "duplicate returns");
        let path = |name| {
            fusion::component_path(
                Path::new(&config.data_root),
                &config.split,
                name,
                &config.segment,
            )
        };
        let cameras = Groups::open(&path("camera_image"))?;
        let lidars = Groups::open(&path("lidar"))?;
        let projections = Groups::open(&path("lidar_camera_projection"))?;
        let calibrations = config
            .lidars
            .iter()
            .map(|&id| {
                Ok((
                    id,
                    fusion::read_sensor_calibration(&path("lidar_calibration"), id)?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            config,
            cameras,
            lidars,
            projections,
            calibrations,
            count: 0,
        })
    }
    fn next(&mut self) -> Result<Option<Frame>> {
        if self.config.frame_limit > 0 && self.count >= self.config.frame_limit {
            return Ok(None);
        }
        let Some((ts, cameras)) = self.cameras.next()? else {
            ensure!(
                self.lidars.next()?.is_none() && self.projections.next()?.is_none(),
                "unmatched trailing LiDAR frames"
            );
            if self.config.frame_limit > 0 {
                ensure!(
                    self.count == self.config.frame_limit,
                    "requested frame count unavailable"
                );
            }
            return Ok(None);
        };
        let (lts, lidars) = self.lidars.next()?.context("LiDAR ended before cameras")?;
        let (pts, projections) = self
            .projections
            .next()?
            .context("projections ended before cameras")?;
        ensure!(
            ts == lts && ts == pts,
            "camera, LiDAR and projection timestamps differ"
        );
        let sensor_map =
            |rows: Vec<RecordBatch>, key: &str| -> Result<BTreeMap<i64, RecordBatch>> {
                let mut out = BTreeMap::new();
                for row in rows {
                    let id = fusion::integer(&row, key)?(0);
                    ensure!(out.insert(id, row).is_none(), "duplicate sensor at {ts}");
                }
                Ok(out)
            };
        let cameras = sensor_map(cameras, "key.camera_name")?;
        let lidars = sensor_map(lidars, "key.laser_name")?;
        let projections = sensor_map(projections, "key.laser_name")?;
        let mut buffers = Vec::new();
        for id in &self.config.cameras {
            let row = cameras
                .get(id)
                .with_context(|| format!("missing camera {id} at {ts}"))?;
            let jpeg = fusion::binary(row, "[CameraImageComponent].image")?.value(0);
            let rgb = image::load_from_memory(jpeg)?.to_rgb8();
            buffers.push(Buffer {
                name: format!("camera_{id}"),
                shape: vec![rgb.height() as usize, rgb.width() as usize, 3],
                values: Values::U8(rgb.into_raw()),
            });
        }
        let mut columns: [Vec<f32>; 14] = std::array::from_fn(|_| Vec::new());
        let mut sensors = Vec::new();
        for id in &self.config.lidars {
            let row = lidars
                .get(id)
                .with_context(|| format!("missing LiDAR {id} at {ts}"))?;
            let projection = projections
                .get(id)
                .with_context(|| format!("missing projection for LiDAR {id}"))?;
            for &ret in &self.config.returns {
                let prefix = format!("[LiDARComponent].range_image_return{ret}");
                let pp = format!("[LiDARCameraProjectionComponent].range_image_return{ret}");
                let values = fusion::list_f32(row, &format!("{prefix}.values"), 0)?;
                let shape = fusion::fixed_i32(row, &format!("{prefix}.shape"), 0)?;
                let pv = fusion::list_f32(projection, &format!("{pp}.values"), 0)?;
                let ps = fusion::fixed_i32(projection, &format!("{pp}.shape"), 0)?;
                let before = columns[0].len();
                append_points(
                    &mut columns,
                    &values,
                    &shape,
                    &pv,
                    &ps,
                    &self.calibrations[id],
                    *id,
                    ret,
                )?;
                sensors.push(
                    serde_json::json!({"lidar":id,"return":ret,"points":columns[0].len()-before}),
                );
            }
        }
        for (name, values) in FIELDS.into_iter().zip(columns) {
            buffers.push(Buffer {
                name: name.into(),
                shape: vec![values.len()],
                values: Values::F32(values),
            });
        }
        let descriptors: Vec<_> = buffers
            .iter()
            .map(|b| {
                serde_json::json!({"name":b.name,"shape":b.shape,
            "dtype":match b.values { Values::F32(_) => "float32", Values::U8(_) => "uint8", Values::Mapped(_, _, _) => unreachable!() }})
            })
            .collect();
        let metadata = CString::new(serde_json::to_string(
            &serde_json::json!({"timestamp":ts,"index":self.count,
            "cameras":self.config.cameras,"sensors":sensors,"buffers":descriptors}),
        )?)?;
        self.count += 1;
        Ok(Some(Frame {
            buffers,
            arrow: None,
            native_copy: None,
            metadata,
        }))
    }
}
#[allow(clippy::too_many_arguments)]
fn append_points(
    out: &mut [Vec<f32>; 14],
    values: &[f32],
    shape: &[i32],
    proj: &[f32],
    ps: &[i32],
    cal: &Calibration,
    lidar: i64,
    ret: usize,
) -> Result<()> {
    ensure!(
        shape.len() == 3 && ps.len() == 3 && shape.iter().all(|&n| n > 0),
        "invalid range shape"
    );
    let (h, w, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    ensure!(
        c >= 4 && ps[0] == shape[0] && ps[1] == shape[1] && ps[2] >= 6,
        "projection shape mismatch"
    );
    ensure!(
        values.len() == h * w * c && proj.len() == h * w * ps[2] as usize,
        "range/projection buffer length mismatch"
    );
    ensure!(
        cal.inclinations.is_empty() || cal.inclinations.len() == h,
        "beam inclination length mismatch"
    );
    let t = &cal.extrinsic;
    let correction = t[1].atan2(t[0]);
    for row in 0..h {
        let beam = h - 1 - row;
        let inc = if cal.inclinations.is_empty() {
            cal.inclination_min
                + (cal.inclination_max - cal.inclination_min) * (beam as f64 + 0.5) / h as f64
        } else {
            cal.inclinations[beam]
        };
        for col in 0..w {
            let i = (row * w + col) * c;
            let range = values[i];
            if !range.is_finite() || range <= 0.0 {
                continue;
            }
            let az = ((w as f64 - col as f64 - 0.5) / w as f64 * 2.0 - 1.0) * std::f64::consts::PI
                - correction;
            let (x, y, z) = (
                az.cos() * inc.cos() * range as f64,
                az.sin() * inc.cos() * range as f64,
                inc.sin() * range as f64,
            );
            let p = (row * w + col) * ps[2] as usize;
            let fields = [
                (t[0] * x + t[1] * y + t[2] * z + t[3]) as f32,
                (t[4] * x + t[5] * y + t[6] * z + t[7]) as f32,
                (t[8] * x + t[9] * y + t[10] * z + t[11]) as f32,
                range,
                values[i + 1],
                values[i + 2],
                proj[p],
                proj[p + 1],
                proj[p + 2],
                proj[p + 3],
                proj[p + 4],
                proj[p + 5],
                lidar as f32,
                ret as f32,
            ];
            for (column, value) in out.iter_mut().zip(fields) {
                column.push(value);
            }
        }
    }
    Ok(())
}
thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn guard<T>(fallback: T, f: impl FnOnce() -> Result<T>) -> T {
    ERROR.with(|e| *e.borrow_mut() = CString::new("").unwrap());
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        other => {
            let message = match other {
                Ok(Err(e)) => format!("{e:#}"),
                _ => "Rust boundary panic".into(),
            };
            ERROR.with(|e| *e.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
            fallback
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn pb_error() -> *const c_char {
    ERROR.with(|e| e.borrow().as_ptr())
}
/// # Safety
/// `config` must point to a readable NUL-terminated UTF-8 JSON string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_open(config: *const c_char) -> *mut Source {
    guard(std::ptr::null_mut(), || {
        ensure!(!config.is_null(), "null configuration");
        let config = serde_json::from_slice(unsafe { CStr::from_ptr(config) }.to_bytes())?;
        Ok(Box::into_raw(Box::new(Source::open(config)?)))
    })
}
/// # Safety
/// `source` must be null or a live handle returned by `pb_open`, freed only once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_source_free(source: *mut Source) {
    if !source.is_null() {
        drop(unsafe { Box::from_raw(source) });
    }
}
/// # Safety
/// `source` must be a live handle exclusively accessed during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_next(source: *mut Source) -> *mut Frame {
    guard(std::ptr::null_mut(), || {
        ensure!(!source.is_null(), "null source");
        Ok(unsafe { &mut *source }
            .next()?
            .map(|f| Box::into_raw(Box::new(f)))
            .unwrap_or(std::ptr::null_mut()))
    })
}
/// # Safety
/// `frame` must be null or live, freed once, with no remaining native buffer views.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_frame_free(frame: *mut Frame) {
    if !frame.is_null() {
        drop(unsafe { Box::from_raw(frame) });
    }
}
/// # Safety
/// `frame` must be null or live. The returned string must not outlive it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_metadata(frame: *const Frame) -> *const c_char {
    if frame.is_null() {
        std::ptr::null()
    } else {
        unsafe { &*frame }.metadata.as_ptr()
    }
}
#[repr(C)]
pub struct View {
    pub data: *const u8,
    pub len: usize,
    pub kind: u32,
}
/// # Safety
/// `frame` must be live and `out` writable/aligned. Views must not outlive the frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_view(frame: *const Frame, index: usize, out: *mut View) -> i32 {
    guard(-1, || {
        ensure!(!frame.is_null() && !out.is_null(), "null view arguments");
        let buffer = unsafe { &*frame }
            .buffers
            .get(index)
            .context("invalid buffer index")?;
        let mut view = match &buffer.values {
            Values::F32(v) => View {
                data: v.as_ptr().cast(),
                len: v.len(),
                kind: 1,
            },
            Values::U8(v) => View {
                data: v.as_ptr(),
                len: v.len(),
                kind: 2,
            },
            Values::Mapped(buffer, kind, len) => View {
                data: buffer.as_ptr(),
                len: *len,
                kind: *kind,
            },
        };
        if let Some(copies) = &unsafe { &*frame }.native_copy {
            view.data = copies[index].address as *const u8;
        }
        unsafe { out.write(view) };
        Ok(0)
    })
}
/// # Safety
/// `frame` must be live and exclusively accessed during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_prepare_arrow(frame: *mut Frame) -> i32 {
    guard(-1, || {
        ensure!(!frame.is_null(), "null frame");
        let frame = unsafe { &mut *frame };
        ensure!(frame.arrow.is_none(), "Arrow already prepared");
        ensure!(frame.native_copy.is_none(), "native copy already prepared");
        frame.arrow = Some(
            frame
                .buffers
                .iter()
                .map(|b| match &b.values {
                    Values::F32(v) => Arc::new(Float32Array::from(v.clone())) as ArrayRef,
                    Values::U8(v) => Arc::new(UInt8Array::from(v.clone())) as ArrayRef,
                    Values::Mapped(buffer, kind, len) => {
                        let copied = ArrowBuffer::from_slice_ref(buffer.as_slice());
                        mapped_array(copied, *kind, *len)
                    }
                })
                .collect(),
        );
        Ok(0)
    })
}
/// # Safety
/// `frame` must be null or live and exclusively accessed during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_clear_arrow(frame: *mut Frame) {
    if !frame.is_null() {
        unsafe { &mut *frame }.arrow = None;
    }
}
/// # Safety
/// `frame` must be live; both outputs must be writable, aligned, empty C Data structures.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_export(
    frame: *const Frame,
    index: usize,
    array: *mut FFI_ArrowArray,
    schema: *mut FFI_ArrowSchema,
) -> i32 {
    guard(-1, || {
        ensure!(
            !frame.is_null() && !array.is_null() && !schema.is_null(),
            "null export arguments"
        );
        let value = unsafe { &*frame }
            .arrow
            .as_ref()
            .context("Arrow not prepared")?
            .get(index)
            .context("invalid Arrow index")?;
        let (a, s) = to_ffi(&value.to_data())?;
        unsafe {
            array.write(a);
            schema.write(s);
        }
        Ok(0)
    })
}
/// Release unconsumed exports after a failed Python import. Consumed structures
/// have null release callbacks; dropping those is also safe.
/// # Safety
/// Non-null pointers must refer to initialized exports from `pb_export`, released only once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_export_release(
    array: *mut FFI_ArrowArray,
    schema: *mut FFI_ArrowSchema,
) {
    if !array.is_null() {
        unsafe {
            std::ptr::drop_in_place(array);
        }
    }
    if !schema.is_null() {
        unsafe {
            std::ptr::drop_in_place(schema);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_copy_matches_arrow_allocation_policy() {
        for len in [0, 1, 63, 64, 65, 127, 513] {
            let bytes = vec![42u8; len];
            let native = NativeAllocation::copy(&bytes).unwrap();
            let arrow = ArrowBuffer::from_slice_ref(&bytes);
            assert_eq!(native.layout.align(), arrow::alloc::ALIGNMENT);
            assert_eq!(native.layout.size(), arrow.capacity());
            assert_eq!(native.address % arrow::alloc::ALIGNMENT, 0);
            let actual = unsafe { std::slice::from_raw_parts(native.address as *const u8, len) };
            assert_eq!(actual, arrow.as_slice());
        }
    }
    #[test]
    fn preserves_both_projections_and_sensor_identity() {
        let cal = Calibration {
            extrinsic: [
                1., 0., 0., 2., 0., 1., 0., 3., 0., 0., 1., 4., 0., 0., 0., 1.,
            ],
            inclination_min: 0.,
            inclination_max: 0.,
            inclinations: vec![0.],
        };
        let mut columns = std::array::from_fn(|_| Vec::new());
        append_points(
            &mut columns,
            &[10., 0.5, 0.1, 0., 0., 0., 0., 0.],
            &[1, 2, 4],
            &[2., 3., 4., 5., 6., 7., 0., 0., 0., 0., 0., 0.],
            &[1, 2, 6],
            &cal,
            4,
            2,
        )
        .unwrap();
        assert_eq!(columns[0].len(), 1);
        assert_eq!(columns[6], vec![2.]);
        assert_eq!(columns[9], vec![5.]);
        assert_eq!(columns[12], vec![4.]);
        assert_eq!(columns[13], vec![2.]);
        assert_eq!(columns[2], vec![4.]);
    }
    #[test]
    fn bad_shapes_fail_before_indexing() {
        let cal = Calibration {
            extrinsic: [0.; 16],
            inclination_min: 0.,
            inclination_max: 0.,
            inclinations: vec![],
        };
        assert!(
            append_points(
                &mut std::array::from_fn(|_| Vec::new()),
                &[],
                &[1, 1, 4],
                &[],
                &[2, 1, 6],
                &cal,
                1,
                1
            )
            .is_err()
        );
    }
}

/// The reduced Python result returns through the same native ABI in every arm.
/// Each row is camera, x, y, width, height, score, class, depth, point count.
/// # Safety
/// `data` must point to `len` readable, aligned f32 values, including a non-null pointer for zero length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_consume_result(data: *const f32, len: usize) -> u64 {
    guard(u64::MAX, || {
        ensure!(
            !data.is_null() && len.is_multiple_of(9),
            "invalid result shape"
        );
        let values = unsafe { std::slice::from_raw_parts(data, len) };
        let mut hash = crc32fast::Hasher::new();
        for row in values.as_chunks::<9>().0 {
            ensure!(
                row[..7].iter().all(|x| x.is_finite())
                    && !row[7].is_infinite()
                    && row[8].is_finite()
                    && row[8] >= 0.0,
                "invalid result values"
            );
            for v in row {
                hash.update(&v.to_le_bytes());
            }
        }
        Ok(hash.finalize() as u64)
    })
}

// Read-only prepared inputs. The mapping is reference-counted by both native
// frames and exported Arrow arrays. No frame payload is copied when queued.
#[cfg(unix)]
struct Mapping {
    address: usize,
    len: usize,
}
#[cfg(unix)]
impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.address as *mut libc::c_void, self.len);
        }
    }
}
fn mapped_array(buffer: ArrowBuffer, kind: u32, len: usize) -> ArrayRef {
    if kind == 1 {
        Arc::new(Float32Array::new(ScalarBuffer::new(buffer, 0, len), None))
    } else {
        Arc::new(UInt8Array::new(ScalarBuffer::new(buffer, 0, len), None))
    }
}
#[cfg(unix)]
fn cached_frame(path: &Path, metadata: &str) -> Result<Frame> {
    use std::{fs::File, os::fd::AsRawFd, ptr::NonNull};
    ensure!(
        cfg!(target_endian = "little"),
        "prepared cache requires little endian"
    );
    let meta: serde_json::Value = serde_json::from_str(metadata)?;
    let descriptors = meta["buffers"].as_array().context("missing buffers")?;
    let file = File::open(path)?;
    let len = usize::try_from(file.metadata()?.len())?;
    ensure!(len > 0, "empty cache file");
    let address = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };
    ensure!(
        address != libc::MAP_FAILED,
        "mmap failed: {}",
        std::io::Error::last_os_error()
    );
    let owner = Arc::new(Mapping {
        address: address as usize,
        len,
    });
    // mmap owns a page-aligned immutable range until its last ArrowBuffer dies.
    let mapped = unsafe {
        ArrowBuffer::from_custom_allocation(
            NonNull::new(address.cast()).context("null mapping")?,
            len,
            owner,
        )
    };
    let mut buffers = Vec::new();
    for d in descriptors {
        let name = d["name"]
            .as_str()
            .context("missing buffer name")?
            .to_owned();
        let shape: Vec<usize> = serde_json::from_value(d["shape"].clone())?;
        let count = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .context("shape overflow")?;
        let kind = match d["dtype"].as_str() {
            Some("float32") => 1,
            Some("uint8") => 2,
            _ => anyhow::bail!("invalid cache dtype"),
        };
        let bytes = count
            .checked_mul(if kind == 1 { 4 } else { 1 })
            .context("buffer overflow")?;
        let offset = usize::try_from(d["offset"].as_u64().context("missing offset")?)?;
        ensure!(
            offset.is_multiple_of(64) && offset.checked_add(bytes).is_some_and(|n| n <= len),
            "unaligned or truncated cache buffer"
        );
        buffers.push(Buffer {
            name,
            shape,
            values: Values::Mapped(mapped.slice_with_length(offset, bytes), kind, count),
        });
    }
    Ok(Frame {
        buffers,
        arrow: None,
        native_copy: None,
        metadata: CString::new(metadata)?,
    })
}
/// # Safety
/// Both arguments must be valid NUL-terminated UTF-8 strings. The cache file
/// must not be modified or truncated while any imported view remains live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_cached_open(
    path: *const c_char,
    metadata: *const c_char,
) -> *mut Frame {
    guard(std::ptr::null_mut(), || {
        ensure!(
            !path.is_null() && !metadata.is_null(),
            "null cache arguments"
        );
        #[cfg(unix)]
        {
            let frame = cached_frame(
                Path::new(unsafe { CStr::from_ptr(path) }.to_str()?),
                unsafe { CStr::from_ptr(metadata) }.to_str()?,
            )?;
            Ok(Box::into_raw(Box::new(frame)))
        }
        #[cfg(not(unix))]
        anyhow::bail!("prepared cache requires Unix")
    })
}
/// # Safety
/// `frame` must point to a live mapped frame, not concurrently mutated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_frame_clone(frame: *const Frame) -> *mut Frame {
    guard(std::ptr::null_mut(), || {
        ensure!(!frame.is_null(), "null frame");
        let frame = unsafe { &*frame };
        ensure!(
            frame.native_copy.is_none(),
            "clone the mapped base before copying"
        );
        ensure!(
            frame
                .buffers
                .iter()
                .all(|b| matches!(b.values, Values::Mapped(..))),
            "only mapped frames support shallow queue handles"
        );
        Ok(Box::into_raw(Box::new(frame.clone())))
    })
}
/// # Safety
/// `frame` must point to a live exclusively-accessed mapped frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_prepare_arrow_shared(frame: *mut Frame) -> i32 {
    guard(-1, || {
        ensure!(!frame.is_null(), "null frame");
        let frame = unsafe { &mut *frame };
        ensure!(frame.arrow.is_none(), "Arrow already prepared");
        ensure!(frame.native_copy.is_none(), "native copy already prepared");
        frame.arrow = Some(
            frame
                .buffers
                .iter()
                .map(|b| {
                    if let Values::Mapped(buffer, kind, len) = &b.values {
                        Ok(mapped_array(buffer.clone(), *kind, *len))
                    } else {
                        anyhow::bail!("shared Arrow preparation requires mapped cache")
                    }
                })
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(0)
    })
}

/// Allocate and copy each mapped payload on the calling (producer) thread.
/// # Safety
/// `frame` must point to a live exclusively-accessed mapped frame, with no
/// native views already exported. Copies remain owned until the frame dies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pb_prepare_native_copy(frame: *mut Frame) -> i32 {
    guard(-1, || {
        ensure!(!frame.is_null(), "null frame");
        let frame = unsafe { &mut *frame };
        ensure!(
            frame.arrow.is_none() && frame.native_copy.is_none(),
            "representation already prepared"
        );
        frame.native_copy = Some(
            frame
                .buffers
                .iter()
                .map(|b| {
                    if let Values::Mapped(buffer, _, _) = &b.values {
                        NativeAllocation::copy(buffer.as_slice()).map(Arc::new)
                    } else {
                        anyhow::bail!("producer copy requires mapped cache")
                    }
                })
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(0)
    })
}
