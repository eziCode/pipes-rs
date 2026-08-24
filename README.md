# pipes-rs

Pipes is a Rust project for running sensor processing and fusion
in front of ROS 2.

## The architectural bet

Pipes owns the path from sensor drivers through processing and fusion. Sensor
drivers write directly into Pipes-owned Apache Arrow buffers. Pipeline stages
share those buffers without repeatedly copying or serializing the sensor data.
After fusion, Pipes converts the much smaller results into standard ROS 2
messages.

```text
camera / lidar / radar / IMU
              |
              v
     Rust sensor drivers
              |
              v
 Pipes metadata + Arrow buffers
              |
              v
 processing, synchronization, and fusion
              |
              v
 detections / tracks / state / health
              |
              v
       standard ROS 2 topics
```

Significant performance gains are expected because the largest data stays inside
one controlled pipeline:

- a sensor can write into its pipeline buffer once;
- processing, recording, visualization, and fusion can share that data;
- large images and point clouds do not need to be serialized and reconstructed
  between every stage; and
- ROS conversion happens once, after the data has been reduced to the result the
  rest of the robot needs.

Pipes should interoperate cleanly with ROS rather than replace it. ROS can keep
handling control, planning, mission logic, user interfaces, and communication
across the robot. Existing ROS nodes consume normal topics and do not need to
know that Pipes is running the sensor pipeline.

The same pipeline can run from live sensors, simulation, or recorded MCAP data.
Pipes records sensor timing, input order, synchronization decisions, delays, and
drops. This lets developers isolate the sensor and fusion subsystem, replay the
same workload, tune it quickly, and determine whether a bad result came from the
fusion algorithm or from data arriving late or going missing.

## Fusion in Motion

[Fusion in Motion](https://github.com/EthanMBoos/fusion-in-motion) is a separate
simulation playground. It can generate repeatable sensor workloads for Pipes,
but neither project depends on the other.
