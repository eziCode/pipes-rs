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
#[derive(Parser)]
#[command(
    about = "Paired Waymo depth-association representation experiment; CPU kernels, fixed annotation boxes"
)]
struct Args {
    #[arg(long)]
    data_root: PathBuf,
    #[arg(long)]
    segment: String,
    #[arg(long, default_value = "training")]
    split: String,
    #[arg(long, default_value_t = 20)]
    frames: usize,
    #[arg(long, default_value_t = 6)]
    repeats: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.frames > 0 && args.repeats > 0,
        "frames and repeats must be positive"
    );
    fs::create_dir_all(&args.output)?;
    let mut csv = BufWriter::new(File::create(args.output.join("raw.csv"))?);
    writeln!(
        csv,
        "timestamp,points,boxes,repeat,position,condition,representation,conversion_ns,fusion_ns,total_ns"
    )?;
    let path = |component| component_path(&args.data_root, &args.split, component, &args.segment);
    let calibration = read_calibration(&path("lidar_calibration"))?;
    let projections = read_all_top_lidar_frames(
        &path("lidar_camera_projection"),
        "[LiDARCameraProjectionComponent]",
        args.frames,
    )?;
    let boxes = read_all_boxes(&path("camera_box"))?;
    let mut frames = 0;
    let mut frames_with_boxes = 0;
    visit_top_lidar_frames(&path("lidar"), "[LiDARComponent]", args.frames, |frame| {
        let projection = projections
            .get(&frame.timestamp)
            .context("missing projection")?;
        let points = convert_top_lidar_points(&frame, projection, &calibration)?;
        let input = boxes.get(&frame.timestamp).cloned().unwrap_or_default();
        if !input.is_empty() {
            frames_with_boxes += 1;
        }
        let mut expected = input.clone();
        add_depths(&mut expected, &points);
        // Each frame's immutable prepared representations are shared across repeats.
        let arrow = point_batch(&points)?;
        let columns = Columns::from_points(&points);
        for variant in 0..3 {
            let mut actual = input.clone();
            match variant {
                0 => fuse_native(&mut actual, &points),
                1 => columns.fuse(&mut actual),
                _ => fuse_arrow(&mut actual, &arrow)?,
            }
            check_depths(&expected, &actual)?;
        }
        // Six permutations balance all three variants' order for the default six repeats.
        const ORDERS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
            [1, 0, 2],
            [0, 2, 1],
        ];
        for iteration in 0..args.warmup + args.repeats {
            for condition_index in 0..2 {
                let fresh = (condition_index + iteration) % 2 == 1;
                let condition = if fresh {
                    "convert_then_fuse"
                } else {
                    "prebuilt"
                };
                for (position, variant) in
                    ORDERS[(iteration + frames) % 6].iter().copied().enumerate()
                {
                    let mut actual = input.clone(); // Same output initialization, outside timers.
                    let start = Instant::now();
                    let fresh_arrow = if fresh && variant == 2 {
                        Some(point_batch(black_box(&points))?)
                    } else {
                        None
                    };
                    let fresh_columns = if fresh && variant == 1 {
                        Some(Columns::from_points(black_box(&points)))
                    } else {
                        None
                    };
                    let ready = Instant::now();
                    match variant {
                        0 => fuse_native(black_box(&mut actual), black_box(&points)),
                        1 => black_box(fresh_columns.as_ref().unwrap_or(&columns))
                            .fuse(black_box(&mut actual)),
                        _ => fuse_arrow(
                            black_box(&mut actual),
                            black_box(fresh_arrow.as_ref().unwrap_or(&arrow)),
                        )?,
                    }
                    black_box(&actual);
                    let finish = Instant::now();
                    check_depths(&expected, &actual)?; // Every warmup and timed output checked outside timer.
                    if iteration >= args.warmup {
                        let conversion = if fresh && variant != 0 {
                            ready.duration_since(start).as_nanos()
                        } else {
                            0
                        };
                        writeln!(
                            csv,
                            "{},{},{},{},{},{},{},{},{},{}",
                            frame.timestamp,
                            points.len(),
                            input.len(),
                            iteration - args.warmup,
                            position,
                            condition,
                            ["native_aos", "native_soa", "arrow"][variant],
                            conversion,
                            finish.duration_since(ready).as_nanos(),
                            finish.duration_since(start).as_nanos()
                        )?;
                    }
                    // Representation destruction is outside the timed interval for every variant.
                    black_box(fresh_arrow);
                    black_box(fresh_columns);
                }
            }
        }
        frames += 1;
        println!(
            "validated frame {}: {} points, {} boxes",
            frames,
            points.len(),
            input.len()
        );
        Ok(())
    })?;
    csv.flush()?;
    ensure!(
        frames == args.frames,
        "requested {} frames, found {frames}",
        args.frames
    );
    ensure!(
        frames_with_boxes > 0,
        "no annotation boxes; experiment has no useful fusion work"
    );
    fs::write(
        args.output.join("completed.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "contract":1,"segment":args.segment,"split":args.split,"frames":frames,"frames_with_boxes":frames_with_boxes,
            "repeats":args.repeats,"warmup_iterations_per_frame":args.warmup,"outputs":"bit-exact depth and count match against original fusion",
            "box_source":"Waymo front camera annotations, not YOLOX predictions","timing":"CPU, single thread, in-memory; input loading and range conversion excluded"
        }))?,
    )?;
    Ok(())
}
