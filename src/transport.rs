use anyhow::Result;
use arrow::{
    array::{ArrayRef, Float32Array, Int8Array, ListArray, StructArray, UInt64Array},
    buffer::{OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field},
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::{Arc, mpsc::sync_channel},
    thread,
    time::{Duration, Instant},
};

struct Message {
    data: Arc<StructArray>,
    started: Instant,
    sent: Instant,
}

pub fn run(output: &Path, frames: usize, points: usize, queue_size: usize) -> Result<Duration> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let payload = Arc::new(payload(points)?);
    let (tx, rx) = sync_channel::<Message>(queue_size);
    let run = Instant::now();
    let producer = thread::spawn(move || -> Result<()> {
        for _ in 0..frames {
            let started = Instant::now();
            tx.send(Message {
                data: Arc::clone(&payload),
                started,
                sent: Instant::now(),
            })?;
        }
        Ok(())
    });
    let mut csv = BufWriter::new(File::create(output)?);
    writeln!(
        csv,
        "frame,status,points,payload_bytes,queue_wait_ns,end_to_end_ns"
    )?;
    for (frame, message) in rx.into_iter().enumerate() {
        let wait = message.sent.elapsed().as_nanos();
        let rows = message
            .data
            .column_by_name("x")
            .and_then(|a| a.as_any().downcast_ref::<ListArray>())
            .map(|a| a.value_length(0))
            .unwrap_or(0);
        let bytes = points * (8 * 4 + 1);
        writeln!(
            csv,
            "{frame},delivered,{rows},{bytes},{wait},{}",
            message.started.elapsed().as_nanos()
        )?;
    }
    producer
        .join()
        .map_err(|_| anyhow::anyhow!("transport producer panicked"))??;
    csv.flush()?;
    Ok(run.elapsed())
}

fn payload(points: usize) -> Result<StructArray> {
    let offsets = || OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, points as i32]));
    let floats: Vec<f32> = (0..points).map(|i| i as f32).collect();
    let list = |name: &str| {
        (
            name.to_owned(),
            Arc::new(ListArray::new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                offsets(),
                Arc::new(Float32Array::from(floats.clone())),
                None,
            )) as ArrayRef,
        )
    };
    let mut columns = vec![
        list("x"),
        list("y"),
        list("z"),
        list("range"),
        list("intensity"),
        list("elongation"),
    ];
    columns.push((
        "camera".into(),
        Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::Int8, false)),
            offsets(),
            Arc::new(Int8Array::from(vec![1_i8; points])),
            None,
        )),
    ));
    columns.push(list("u"));
    columns.push(list("v"));
    columns.push(("sequence".into(), Arc::new(UInt64Array::from(vec![0]))));
    let fields = columns
        .iter()
        .map(|(n, a)| Arc::new(Field::new(n, a.data_type().clone(), false)))
        .collect::<Vec<_>>();
    Ok(StructArray::try_new(
        fields.into(),
        columns.into_iter().map(|(_, a)| a).collect(),
        None,
    )?)
}
