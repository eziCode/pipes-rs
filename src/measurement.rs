use std::{fmt, str::FromStr, time::Instant};

use anyhow::{Result, bail};
use arrow::{ipc::writer::StreamWriter, record_batch::RecordBatch};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SensorId {
    Camera(CameraName),
    Lidar(LidarName),
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CameraName {
    Front = 1,
    FrontLeft = 2,
    FrontRight = 3,
    SideLeft = 4,
    SideRight = 5,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LidarName {
    Top = 1,
    Front = 2,
    SideLeft = 3,
    SideRight = 4,
    Rear = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SensorKind {
    Camera,
    Lidar,
}

impl SensorId {
    pub fn kind(self) -> SensorKind {
        match self {
            Self::Camera(_) => SensorKind::Camera,
            Self::Lidar(_) => SensorKind::Lidar,
        }
    }

    pub fn waymo_id(self) -> i64 {
        match self {
            Self::Camera(name) => name as i64,
            Self::Lidar(name) => name as i64,
        }
    }
}

impl FromStr for SensorId {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let Some((kind, name)) = value.split_once(':') else {
            bail!("sensor must have the form camera:front or lidar:top")
        };
        let normalized = name.to_ascii_lowercase().replace('-', "_");
        match (kind, normalized.as_str()) {
            ("camera", "front") => Ok(Self::Camera(CameraName::Front)),
            ("camera", "front_left") => Ok(Self::Camera(CameraName::FrontLeft)),
            ("camera", "front_right") => Ok(Self::Camera(CameraName::FrontRight)),
            ("camera", "side_left") => Ok(Self::Camera(CameraName::SideLeft)),
            ("camera", "side_right") => Ok(Self::Camera(CameraName::SideRight)),
            ("lidar", "top") => Ok(Self::Lidar(LidarName::Top)),
            ("lidar", "front") => Ok(Self::Lidar(LidarName::Front)),
            ("lidar", "side_left") => Ok(Self::Lidar(LidarName::SideLeft)),
            ("lidar", "side_right") => Ok(Self::Lidar(LidarName::SideRight)),
            ("lidar", "rear") => Ok(Self::Lidar(LidarName::Rear)),
            _ => bail!("unsupported sensor `{value}`"),
        }
    }
}

impl fmt::Display for SensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Camera(name) => write!(f, "camera:{}", name.as_str()),
            Self::Lidar(name) => write!(f, "lidar:{}", name.as_str()),
        }
    }
}

impl CameraName {
    fn as_str(self) -> &'static str {
        match self {
            Self::Front => "front",
            Self::FrontLeft => "front_left",
            Self::FrontRight => "front_right",
            Self::SideLeft => "side_left",
            Self::SideRight => "side_right",
        }
    }
}

impl LidarName {
    fn as_str(self) -> &'static str {
        match self {
            Self::Top => "top",
            Self::Front => "front",
            Self::SideLeft => "side_left",
            Self::SideRight => "side_right",
            Self::Rear => "rear",
        }
    }
}

#[derive(Debug)]
pub struct Measurement {
    pub sensor: SensorId,
    pub sequence: u64,
    pub sensor_timestamp_ns: i64,
    pub host_arrival_unix_ns: u64,
    pub due_at: Instant,
    pub enqueued_at: Instant,
    pub payload: RecordBatch,
}

impl Measurement {
    pub fn encode_ipc(&self) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut encoded, &self.payload.schema())?;
            writer.write(&self.payload)?;
            writer.finish()?;
        }
        Ok(encoded)
    }
}
