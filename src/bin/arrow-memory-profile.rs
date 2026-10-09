//! Linux per-thread counters scoped to prebuilt depth association.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use pipes_rs::{arrow_experiment::*, fusion::*};
use std::{
    fs::{self, File},
    hint::black_box,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

mod counters {
    use anyhow::{Result, ensure};
    #[repr(C)]
    #[derive(Default)]
    struct Attr {
        type_: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup: u32,
        bp_type: u32,
        config1: u64,
    }
    pub const EVENTS: [(&str, u32, u64); 8] = [
        ("cycles", 0, 0),
        ("instructions", 0, 1),
        ("branches", 0, 4),
        ("branch_misses", 0, 5),
        ("l1d_read_accesses", 3, 0),
        ("l1d_read_misses", 3, 1 << 16),
        ("llc_read_accesses", 3, 2),
        ("llc_read_misses", 3, 2 | (1 << 16)),
    ];
    pub struct Counter {
        fd: i32,
        base: [u64; 3],
    }
    impl Counter {
        pub fn open(type_: u32, config: u64) -> Result<Self> {
            ensure!(cfg!(target_os = "linux"), "Linux counters required");
            let attr = Attr {
                type_,
                size: std::mem::size_of::<Attr>() as u32,
                config,
                read_format: 3,
                flags: 1 | (1 << 5) | (1 << 6),
                ..Default::default()
            };
            // Current thread, user-space only, no system-wide/uncore access.
            let fd = unsafe {
                libc::syscall(libc::SYS_perf_event_open, &attr, 0i32, -1i32, -1i32, 8u64)
            };
            if fd < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(Self {
                fd: fd as i32,
                base: [0; 3],
            })
        }
        fn ioctl(&self, request: libc::c_ulong) -> Result<()> {
            if unsafe { libc::ioctl(self.fd, request, 0) } < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(())
        }
        fn read(&self) -> Result<[u64; 3]> {
            let mut values = [0u64; 3];
            let n = unsafe { libc::read(self.fd, values.as_mut_ptr().cast(), 24) };
            if n < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            ensure!(n == 24, "counter read returned {n} bytes, expected 24");
            Ok(values)
        }
        pub fn start(&mut self) -> Result<()> {
            // RESET resets the event count but not accumulated enabled/running time.
            self.base = self.read()?;
            self.ioctl(0x2403)?;
            self.ioctl(0x2400)
        }
        pub fn stop(&self) -> Result<[u64; 3]> {
            self.ioctl(0x2401)?;
            let now = self.read()?;
            Ok([now[0], now[1] - self.base[1], now[2] - self.base[2]])
        }
    }
    impl Drop for Counter {
        fn drop(&mut self) {
            unsafe {
                libc::close(self.fd);
            }
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn linux_abi_and_event_encoding() {
            assert_eq!(std::mem::size_of::<Attr>(), 64);
            assert_eq!(EVENTS[7].2, 65538);
        }
        #[test]
        fn invalid_event_is_rejected() {
            assert!(Counter::open(u32::MAX, u64::MAX).is_err());
        }
    }
}
#[derive(Parser)]
struct Args {
    #[arg(long)]
    data_root: PathBuf,
    #[arg(long)]
    segment: String,
    #[arg(long, default_value = "training")]
    split: String,
    #[arg(long, default_value_t = 20)]
    frames: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.frames > 0 && a.repeats > 0,
        "positive frames/repeats required"
    );
    fs::create_dir_all(&a.output)?;
    let mut availability = Vec::new();
    let mut counters = Vec::new();
    for (name, type_, config) in counters::EVENTS {
        let probe = (|| -> Result<_> {
            let mut counter = counters::Counter::open(type_, config)?;
            counter.start()?;
            let mut value = 1u64;
            for i in 1..10000 {
                value = black_box(value.wrapping_add(black_box(i)));
            }
            black_box(value);
            let sample = counter.stop()?;
            ensure!(sample[2] > 0, "event never scheduled");
            // Record empty-bracket overhead separately, never silently subtract it.
            counter.start()?;
            black_box(0u64);
            let overhead = counter.stop()?;
            Ok((counter, overhead))
        })();
        match probe {
            Ok((counter,overhead))=>{availability.push(serde_json::json!({"event":name,"type":type_,"config":config,"available":true,"empty_bracket_count":overhead[0]}));counters.push((name,counter));},
            Err(e)=>availability.push(serde_json::json!({"event":name,"type":type_,"config":config,"available":false,"error":format!("{e:#}")})),
        }
    }
    fs::write(
        a.output.join("counter-availability.json"),
        serde_json::to_vec_pretty(&availability)?,
    )?;
    ensure!(
        !counters.is_empty(),
        "No hardware counters available; see counter-availability.json. No memory-read conclusions are possible."
    );
    let mut csv = BufWriter::new(File::create(a.output.join("counters.csv"))?);
    writeln!(
        csv,
        "timestamp,points,boxes,point_examinations,event,repeat,position,representation,count,time_enabled_ns,time_running_ns,running_fraction,elapsed_ns"
    )?;
    let path = |c| component_path(&a.data_root, &a.split, c, &a.segment);
    let calibration = read_calibration(&path("lidar_calibration"))?;
    let projections = read_all_top_lidar_frames(
        &path("lidar_camera_projection"),
        "[LiDARCameraProjectionComponent]",
        a.frames,
    )?;
    let boxes = read_all_boxes(&path("camera_box"))?;
    let mut seen = 0;
    let mut nonempty = 0;
    visit_top_lidar_frames(&path("lidar"), "[LiDARComponent]", a.frames, |frame| {
        let points = convert_top_lidar_points(
            &frame,
            projections
                .get(&frame.timestamp)
                .context("missing projection")?,
            &calibration,
        )?;
        let input = boxes.get(&frame.timestamp).cloned().unwrap_or_default();
        if !input.is_empty() {
            nonempty += 1;
        }
        let mut expected = input.clone();
        add_depths(&mut expected, &points);
        let columns = Columns::from_points(&points);
        let arrow = point_batch(&points)?;
        let run = |variant: usize, d: &mut [Detection]| -> Result<()> {
            match variant {
                0 => fuse_native(black_box(d), black_box(&points)),
                1 => black_box(&columns).fuse(black_box(d)),
                _ => fuse_arrow(black_box(d), black_box(&arrow))?,
            }
            Ok(())
        };
        for (event_index, (name, counter)) in counters.iter_mut().enumerate() {
            for repeat in 0..a.repeats {
                for position in 0..3 {
                    let variant = (position + repeat + seen + event_index) % 3;
                    for _ in 0..a.warmup {
                        let mut d = input.clone();
                        run(variant, &mut d)?;
                        check_depths(&expected, &d)?;
                    }
                    let mut actual = input.clone();
                    counter.start()?;
                    let started = Instant::now();
                    run(variant, &mut actual)?;
                    black_box(&actual);
                    let elapsed = started.elapsed().as_nanos();
                    let [count, enabled, running] = counter.stop()?;
                    check_depths(&expected, &actual)?;
                    ensure!(
                        enabled > 0 && running > 0 && running <= enabled,
                        "counter not scheduled correctly"
                    );
                    writeln!(
                        csv,
                        "{},{},{},{},{},{},{},{},{},{},{},{:.6},{}",
                        frame.timestamp,
                        points.len(),
                        input.len(),
                        points.len() * input.len(),
                        name,
                        repeat,
                        position,
                        ["native_aos", "native_soa", "arrow"][variant],
                        count,
                        enabled,
                        running,
                        running as f64 / enabled as f64,
                        elapsed
                    )?;
                }
            }
        }
        seen += 1;
        println!("profiled and validated frame {seen}");
        Ok(())
    })?;
    csv.flush()?;
    ensure!(
        seen == a.frames && nonempty > 0,
        "incomplete/empty workload"
    );
    fs::write(
        a.output.join("completed.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "frames":seen,"frames_with_boxes":nonempty,"repeats":a.repeats,"available_events":counters.len(),
            "scope":"user-space fusion on prebuilt buffers; includes range allocation/sort, excludes input preparation, conversion and validation",
            "dram_bytes_measured":false,"outputs":"exact depth/count match","counter_scope":"current thread only"
        }))?,
    )?;
    Ok(())
}
