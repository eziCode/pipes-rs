use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result, bail};
use arrow::{
    array::{
        Array, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array,
        UInt64Array,
    },
    datatypes::Schema,
    record_batch::RecordBatch,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::measurement::{Measurement, SensorId, SensorKind};

pub fn load_segment(
    root: &Path,
    split: &str,
    segment: &str,
    camera: &SensorId,
    lidar: &SensorId,
) -> Result<Vec<Measurement>> {
    let mut measurements = Vec::new();
    load_component(
        &component_path(root, split, "camera_image", segment),
        *camera,
        &mut measurements,
    )?;
    load_component(
        &component_path(root, split, "lidar", segment),
        *lidar,
        &mut measurements,
    )?;
    Ok(measurements)
}

fn component_path(root: &Path, split: &str, component: &str, segment: &str) -> PathBuf {
    root.join(split)
        .join(component)
        .join(format!("{segment}.parquet"))
}

fn load_component(path: &Path, sensor: SensorId, output: &mut Vec<Measurement>) -> Result<()> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to read Parquet metadata from {}", path.display()))?;
    let schema = builder.schema().clone();
    let timestamp_index = find_field(&schema, "key.frame_timestamp_micros")?;
    let sensor_key = match sensor.kind() {
        SensorKind::Camera => "key.camera_name",
        SensorKind::Lidar => "key.laser_name",
    };
    let sensor_index = find_field(&schema, sensor_key)?;
    let reader = builder.with_batch_size(32).build()?;
    let mut sequence = 0u64;

    for batch in reader {
        let batch = batch?;
        let timestamps = batch
            .column(timestamp_index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .with_context(|| {
                format!(
                    "`key.frame_timestamp_micros` in {} is not Int64",
                    path.display()
                )
            })?;
        for row in 0..batch.num_rows() {
            if integer_value(batch.column(sensor_index).as_ref(), row)? != sensor.waymo_id() {
                continue;
            }
            let payload = RecordBatch::try_new(
                batch.schema(),
                batch.columns().iter().map(|a| a.slice(row, 1)).collect(),
            )?;
            output.push(Measurement {
                sensor,
                sequence,
                sensor_timestamp_ns: timestamps.value(row).saturating_mul(1_000),
                host_arrival_unix_ns: 0,
                due_at: Instant::now(),
                enqueued_at: Instant::now(),
                payload,
            });
            sequence += 1;
        }
    }
    println!(
        "loaded {sequence} {sensor} measurements from {}",
        path.display()
    );
    Ok(())
}

fn find_field(schema: &Arc<Schema>, name: &str) -> Result<usize> {
    schema
        .fields()
        .iter()
        .position(|field| field.name() == name)
        .with_context(|| {
            let available = schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!("missing `{name}`; available columns: {available}")
        })
}

fn integer_value(array: &dyn Array, row: usize) -> Result<i64> {
    if array.is_null(row) {
        bail!("sensor name is null at row {row}")
    }
    if let Some(values) = array.as_any().downcast_ref::<Int8Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int16Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt8Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt16Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(values.value(row) as i64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return i64::try_from(values.value(row)).context("sensor name does not fit in i64");
    }
    bail!(
        "sensor name column has unsupported Arrow type {}",
        array.data_type()
    )
}
