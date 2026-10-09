//! Shared same-host timing and validation contract for the full perception pilot.
use crate::fusion::{Detection, Point};
use anyhow::{Result, ensure};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

pub const HEADER: &str = "frame,timestamp_micros,status,camera_width,camera_height,camera_checksum,points,detections,ground_truth,true_positives,false_positives,false_negatives,depth_matched,camera_stage_ns,inference_ns,lidar_stage_ns,camera_queue_wait_ns,lidar_queue_wait_ns,sync_skew_ns,fusion_ns,end_to_end_ns,camera_started_ns,lidar_started_ns,camera_sent_ns,lidar_sent_ns,camera_received_ns,lidar_received_ns,pair_ready_ns,fusion_finished_ns,camera_pair_wait_ns,lidar_pair_wait_ns";

/// CLOCK_MONOTONIC has a shared epoch across processes on the same host.
/// These timestamps must never be compared across hosts or with dataset time.
pub fn monotonic_ns() -> u64 {
    #[cfg(unix)]
    {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: ts is a valid writable timespec; CLOCK_MONOTONIC is supported.
        let status = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
        assert_eq!(status, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }
    #[cfg(not(unix))]
    panic!("the cross-process pilot clock requires Unix")
}

pub struct Timing {
    pub camera_started: u64,
    pub lidar_started: u64,
    pub camera_sent: u64,
    pub lidar_sent: u64,
    pub camera_received: u64,
    pub lidar_received: u64,
    pub pair_ready: u64,
    pub fusion_finished: u64,
}

impl Timing {
    pub fn validate(&self) -> Result<()> {
        for (start, send, receive) in [
            (self.camera_started, self.camera_sent, self.camera_received),
            (self.lidar_started, self.lidar_sent, self.lidar_received),
        ] {
            ensure!(
                start <= send
                    && send <= receive
                    && receive <= self.pair_ready
                    && self.pair_ready <= self.fusion_finished,
                "invalid pilot event ordering"
            );
        }
        Ok(())
    }
    pub fn age(&self) -> u64 {
        self.fusion_finished - self.camera_started.min(self.lidar_started)
    }
    pub fn write_suffix(&self, out: &mut impl Write) -> Result<()> {
        self.validate()?;
        writeln!(
            out,
            ",{},{},{},{},{},{},{},{},{},{}",
            self.camera_started,
            self.lidar_started,
            self.camera_sent,
            self.lidar_sent,
            self.camera_received,
            self.lidar_received,
            self.pair_ready,
            self.fusion_finished,
            self.pair_ready - self.camera_received,
            self.pair_ready - self.lidar_received
        )?;
        Ok(())
    }
}

pub struct Validation(Option<BufWriter<File>>);
impl Validation {
    pub fn new(csv: &Path) -> Result<Self> {
        let file = if std::env::var("PILOT_VALIDATE").as_deref() == Ok("1") {
            Some(BufWriter::new(File::create(
                csv.with_extension("validation.jsonl"),
            )?))
        } else {
            None
        };
        Ok(Self(file))
    }
    pub fn write(
        &mut self,
        timestamp: i64,
        points: &[Point],
        detections: &[Detection],
    ) -> Result<()> {
        if let Some(out) = &mut self.0 {
            for d in detections {
                ensure!(
                    [d.x, d.y, d.width, d.height].iter().all(|v| v.is_finite())
                        && d.score.is_none_or(|v| v.is_finite())
                        && d.depth_m.is_none_or(|v| v.is_finite()),
                    "non-finite detection value in validation"
                );
            }
            // Ordered, bit-exact fingerprint covers all nine point fields, not padding.
            // This is an accidental-corruption check, not a cryptographic proof.
            let mut hash = crc32fast::Hasher::new();
            for p in points {
                for value in [p.x, p.y, p.z, p.range, p.intensity, p.elongation, p.u, p.v] {
                    hash.update(&value.to_le_bytes());
                }
                hash.update(&p.camera.to_le_bytes());
            }
            serde_json::to_writer(
                &mut *out,
                &serde_json::json!({
                    "timestamp_micros": timestamp, "points": points.len(),
                    "points_crc32": format!("{:08x}", hash.finalize()), "detections": detections
                }),
            )?;
            writeln!(out)?;
        }
        Ok(())
    }
    pub fn flush(&mut self) -> Result<()> {
        if let Some(out) = &mut self.0 {
            out.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separates_transport_pairing_and_age() {
        let t = Timing {
            camera_started: 1,
            lidar_started: 2,
            camera_sent: 3,
            lidar_sent: 4,
            camera_received: 5,
            lidar_received: 8,
            pair_ready: 9,
            fusion_finished: 12,
        };
        t.validate().unwrap();
        assert_eq!(t.age(), 11);
        let mut out = Vec::new();
        t.write_suffix(&mut out).unwrap();
        assert!(String::from_utf8(out).unwrap().ends_with(",4,1\n"));
        assert!(
            Timing {
                camera_received: 2,
                ..t
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn monotonic_clock_does_not_go_backwards() {
        let a = monotonic_ns();
        let b = monotonic_ns();
        assert!(b >= a);
    }
}
