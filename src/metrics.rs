use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::{Mutex, MutexGuard},
    time::Duration,
};

use anyhow::{Context, Result};

use crate::measurement::SensorId;

#[derive(Default)]
pub struct RunMetrics {
    sensors: Mutex<HashMap<SensorId, Metrics>>,
}

pub struct SensorMetrics<'a> {
    sensor: SensorId,
    guard: MutexGuard<'a, HashMap<SensorId, Metrics>>,
}

#[derive(Default)]
pub struct Metrics {
    submitted: u64,
    accepted: u64,
    delivered: u64,
    dropped: u64,
    sequence_gaps: u64,
    bytes: u64,
    max_depth: usize,
    queue_wait_ns: Vec<u64>,
    age_ns: Vec<u64>,
    first_delivery: Option<std::time::Instant>,
    last_delivery: Option<std::time::Instant>,
    last_sequence: Option<u64>,
    checksum_xor: u32,
    records: Vec<MeasurementRecord>,
}

struct MeasurementRecord {
    sequence: u64,
    status: &'static str,
    sensor_timestamp_ns: i64,
    host_arrival_unix_ns: u64,
    bytes: u64,
    queue_wait_ns: u64,
    age_ns: u64,
    checksum: u32,
}

pub struct Delivery {
    pub sequence: u64,
    pub sensor_timestamp_ns: i64,
    pub host_arrival_unix_ns: u64,
    pub bytes: u64,
    pub queue_wait: Duration,
    pub age: Duration,
    pub checksum: u32,
}

impl RunMetrics {
    pub fn for_sensor(&self, sensor: SensorId) -> SensorMetrics<'_> {
        SensorMetrics {
            sensor,
            guard: self.sensors.lock().expect("metrics mutex poisoned"),
        }
    }

    pub fn print_summary(&self) {
        println!("\nsummary:");
        let guard = self.sensors.lock().expect("metrics mutex poisoned");
        let mut rows: Vec<_> = guard.iter().collect();
        rows.sort_by_key(|(sensor, _)| **sensor);
        for (sensor, metrics) in rows {
            let elapsed = match (metrics.first_delivery, metrics.last_delivery) {
                (Some(first), Some(last)) => last.saturating_duration_since(first).as_secs_f64(),
                _ => 0.0,
            };
            let rate = if elapsed > 0.0 && metrics.delivered > 1 {
                (metrics.delivered - 1) as f64 / elapsed
            } else {
                0.0
            };
            println!(
                "{sensor}: submitted={} accepted={} delivered={} dropped={} gaps={} rate={rate:.2}Hz bytes={} max_depth={} p95_wait={} p95_age={} checksum={:08x}",
                metrics.submitted,
                metrics.accepted,
                metrics.delivered,
                metrics.dropped,
                metrics.sequence_gaps,
                metrics.bytes,
                metrics.max_depth,
                format_duration(percentile(&metrics.queue_wait_ns, 0.95)),
                format_duration(percentile(&metrics.age_ns, 0.95)),
                metrics.checksum_xor,
            );
        }
    }

    pub fn write_csv(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .with_context(|| format!("failed to create metrics CSV at {}", path.display()))?;
        let mut output = BufWriter::new(file);
        writeln!(
            output,
            "sensor,sequence,status,sensor_timestamp_ns,host_arrival_unix_ns,payload_bytes,queue_wait_ns,measurement_age_ns,checksum"
        )?;
        let guard = self.sensors.lock().expect("metrics mutex poisoned");
        let mut rows: Vec<_> = guard.iter().collect();
        rows.sort_by_key(|(sensor, _)| **sensor);
        for (sensor, metrics) in rows {
            let mut records: Vec<_> = metrics.records.iter().collect();
            records.sort_by_key(|record| record.sequence);
            for record in records {
                writeln!(
                    output,
                    "{sensor},{},{},{},{},{},{},{},{:08x}",
                    record.sequence,
                    record.status,
                    record.sensor_timestamp_ns,
                    record.host_arrival_unix_ns,
                    record.bytes,
                    record.queue_wait_ns,
                    record.age_ns,
                    record.checksum
                )?;
            }
        }
        output.flush()?;
        Ok(())
    }
}

impl SensorMetrics<'_> {
    fn metrics(&mut self) -> &mut Metrics {
        self.guard.entry(self.sensor).or_default()
    }

    pub fn record_submitted(mut self) {
        self.metrics().submitted += 1;
    }

    pub fn record_accepted(mut self, depth: usize) {
        let metrics = self.metrics();
        metrics.accepted += 1;
        metrics.max_depth = metrics.max_depth.max(depth);
    }

    pub fn record_dropped(
        mut self,
        sequence: u64,
        sensor_timestamp_ns: i64,
        host_arrival_unix_ns: u64,
    ) {
        let metrics = self.metrics();
        metrics.dropped += 1;
        metrics.records.push(MeasurementRecord {
            sequence,
            status: "dropped",
            sensor_timestamp_ns,
            host_arrival_unix_ns,
            bytes: 0,
            queue_wait_ns: 0,
            age_ns: 0,
            checksum: 0,
        });
    }

    pub fn record_delivered(mut self, delivery: Delivery) {
        let now = std::time::Instant::now();
        let metrics = self.metrics();
        metrics.delivered += 1;
        metrics.bytes += delivery.bytes;
        let queue_wait_ns = delivery.queue_wait.as_nanos().min(u64::MAX as u128) as u64;
        let age_ns = delivery.age.as_nanos().min(u64::MAX as u128) as u64;
        metrics.queue_wait_ns.push(queue_wait_ns);
        metrics.age_ns.push(age_ns);
        metrics.first_delivery.get_or_insert(now);
        metrics.last_delivery = Some(now);
        if let Some(previous) = metrics.last_sequence {
            metrics.sequence_gaps += delivery.sequence.saturating_sub(previous + 1);
        }
        metrics.last_sequence = Some(delivery.sequence);
        metrics.checksum_xor ^= delivery.checksum;
        metrics.records.push(MeasurementRecord {
            sequence: delivery.sequence,
            status: "delivered",
            sensor_timestamp_ns: delivery.sensor_timestamp_ns,
            host_arrival_unix_ns: delivery.host_arrival_unix_ns,
            bytes: delivery.bytes,
            queue_wait_ns,
            age_ns,
            checksum: delivery.checksum,
        });
    }
}

fn percentile(values: &[u64], quantile: f64) -> Duration {
    if values.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    Duration::from_nanos(sorted[index])
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{:.2}ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.2}us", duration.as_secs_f64() * 1_000_000.0)
    }
}
