use std::{
    thread,
    time::{Duration, Instant},
};

pub struct ReplayClock {
    first_timestamp_ns: i64,
    started_at: Instant,
    speed: f64,
}

impl ReplayClock {
    pub fn new(first_timestamp_ns: i64, speed: f64) -> Self {
        Self {
            first_timestamp_ns,
            started_at: Instant::now(),
            speed,
        }
    }

    pub fn due_at(&self, timestamp_ns: i64) -> Instant {
        if self.speed == 0.0 {
            return self.started_at;
        }
        let delta = timestamp_ns.saturating_sub(self.first_timestamp_ns) as f64 / self.speed;
        self.started_at + Duration::from_nanos(delta.max(0.0) as u64)
    }

    pub fn wait_until(&self, timestamp_ns: i64) {
        let due = self.due_at(timestamp_ns);
        if let Some(wait) = due.checked_duration_since(Instant::now()) {
            thread::sleep(wait);
        }
    }
}
