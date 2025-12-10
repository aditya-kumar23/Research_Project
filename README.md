# cosmo_slam

Instrumented centralised and decentralised SLAM backends for COSMO-bench datasets. The
codebase couples a robust JRL loader, iSAM2/batch optimisation, a DDF-SAM style
decentralised solver, and KPI instrumentation that works for in-process and ROS 2
transports.

*Status: actively used for KPI studies; packaging is not finalised yet.*

## What's inside

- Centralised backend that builds a global factor graph, runs iSAM2 or batch LM, and
  exports trajectories, plots, and KPIs.
- DDF-SAM inspired decentralised backend with per-robot agents, optional ROS 2 peer bus,
  and multi-process execution.
- ROS 2 shims for factor batches, interface messages, ACK topics, QoS parsing, and
  optional network impairments.
- KPI collectors for latency, bandwidth, resource usage, and derived KPIs (loop-closure
  correction, convergence time).
- Experiment scripts to orchestrate KPI runs, replay datasets, compare runs, and export
  long-format CSVs.

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | CLI entry-point for the centralised and decentralised backends. |
| `cosmo_slam_centralised/` | Loader, graph builder, solvers, KPI instrumentation, visualisation helpers. |
| `cosmo_slam_decentralised/` | DDF-SAM agents, ROS/in-process communication layers, multi-process runner. |
| `cosmo_slam_ros2/` | ROS 2 transport helpers (factor payloads, interface messages, ACKs, QoS, impairments). |
| `tools/` | Orchestration scripts, KPI exporters, comparison utilities, ROS factor publisher. |
| `datasets/` | COSMO-bench `.jrl` datasets organised by modality (`proradio`, `wifi`). |
| `tests/` | pytest suites for loader, ROS 2 shims, KPI utilities, and deterministic unit tests. |
| `scaling_trends/` | Example KPI summaries and plots from previous runs (reference only). |
| `output*` | Sample output trees from earlier experiments (safe to delete). |

## Requirements

### Runtime

- Python 3.8 or newer.
- GTSAM Python bindings (for example `pip install gtsam==4.2.0.4`). Both the centralised
  and decentralised pipelines require GTSAM.
- Base Python deps: `pip install -e .` installs `numpy`, `psutil`, and `msgpack`.
- Plotting: `pip install matplotlib` (used by `cosmo_slam_centralised.viz`).
- Testing: `pip install pytest`.

### Optional extras

- `pip install pynvml` to record GPU utilisation in `ResourceMonitor`.
- A ROS 2 environment with `rclpy` for ROS transports (`--central-transport ros2`,
  `--ddf-transport ros2`).
- `scipy` unlocks the Welch t-test summaries in `tools/orchestrate_kpi.py`.
- Set `ROS_DOMAIN_ID`, `FASTRTPS_DEFAULT_PROFILES_FILE`, etc., as needed for your ROS 2
  network.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
pip install matplotlib pytest
pip install gtsam==4.2.0.4  # or another wheel matching your platform
# Optional extras
pip install pynvml scipy
```

For ROS 2 features, source your ROS 2 workspace before running any scripts so that
`rclpy` is importable.

## Quickstart

### Centralised iSAM2 (in-process)

```bash
python main.py \
  --jrl datasets/COSMO/proradio/main_campus_proradio.jrl \
  --export-path output/main_campus_isam2 \
  --backend centralised \
  --solver isam2
```

This ingests factors directly from the `.jrl`, updates iSAM2 in batches, and writes
trajectories, plots, KPI JSONL, and resource metrics under `output/main_campus_isam2/`.

### Centralised ROS 2 ingest

Requires a running ROS 2 graph. Launch a publisher in one terminal and the solver in
another:

```bash
# Terminal 1: replay factors over ROS 2
python tools/ros2_factor_publisher.py \
  --jrl datasets/COSMO/proradio/main_campus_proradio.jrl \
  --topic-prefix /cosmo/factor_batch

# Terminal 2: subscribe and solve
python main.py \
  --jrl datasets/COSMO/proradio/main_campus_proradio.jrl \
  --export-path output/main_campus_ros2 \
  --backend centralised \
  --solver isam2 \
  --central-transport ros2 \
  --central-ros2-topic /cosmo/factor_batch
```

ACK messages are published on `/cosmo/factor_batch/ack/<robot>` when batches enter
optimisation.

### Decentralised DDF-SAM (in-process)

```bash
python main.py \
  --jrl datasets/COSMO/proradio/main_campus_proradio.jrl \
  --export-path output/main_campus_ddf \
  --backend decentralised \
  --ddf-transport inproc \
  --ddf-rounds 6
```

Per-robot estimates, interface summaries, and KPI metrics are written under the export
directory.

### Decentralised ROS 2 multi-process

```bash
python main.py \
  --jrl datasets/COSMO/proradio/main_campus_proradio.jrl \
  --export-path output/main_campus_ddf_ros2 \
  --backend decentralised \
  --ddf-transport ros2 \
  --ddf-multiprocess \
  --ddf-ros2-topic-prefix /cosmo/iface \
  --ddf-rounds 8 \
  --ddf-interface-time-scale 1e9 \
  --ddf-interface-max-sleep 0.1
```

Each agent runs in its own process and communicates through ROS 2 topics. The parent
process monitors bandwidth, latency, and resource usage, then merges per-agent
trajectories into `trajectories/trajectory_<robot>.csv`.

## CLI flag guide

### Common flags

| Flag(s) | Purpose |
| --- | --- |
| `--jrl PATH`, `--export-path DIR` | Select the input dataset and the output directory (both required). |
| `--backend {centralised,decentralised}` | Choose the solver pipeline. Spelling with `-ized` is accepted. |
| `--solver {isam2,batch}` | Centralised solver mode. `batch` runs LM on the full graph. |
| `--robust {none,huber,cauchy}`, `--robust-k FLOAT` | Configure robust kernels applied when constructing factors. |
| `--batch-size INT`, `--relin-th FLOAT`, `--relin-skip INT` | iSAM2 update sizing and relinearisation controls. |
| `--quat-order {wxyz,xyzw}` | Quaternion ordering used by the dataset. |
| `--include-potential-outliers` | Keep factors flagged as potential outliers in the `.jrl`. |
| `--eval-gt` | Align estimates to ground truth and export ATE/RPE metrics and overlay plots. |
| `--resource-interval FLOAT` | Sampling period (seconds) for `ResourceMonitor` (default 0.5). |
| `--log LEVEL` | Logging verbosity (`INFO`, `DEBUG`, ...). |
| `--use-sim-time` | Honour ROS 2 simulation time (propagates `COSMO_USE_SIM_TIME=1`). |

### Centralised ROS 2 ingest

| Flag | Purpose |
| --- | --- |
| `--central-transport {inproc,ros2}` | Switch between direct `.jrl` ingestion and ROS 2 topics. |
| `--central-ros2-topic` | Topic prefix for factor batches (default `/cosmo/factor_batch`). |
| `--central-ros2-reliability`, `--central-ros2-durability` | QoS reliability/durability settings. |
| `--central-ros2-depth` | Subscription queue depth. |
| `--central-ros2-queue-size` | Internal buffering before optimisation (`0` = unbounded). |
| `--central-ros2-idle-timeout` | Idle timeout (seconds) after the last message before shutting down. |
| `--central-ros2-max-batches` | Optional early stop after N batches (0 disables). |

*ROS 2 factor ingest currently supports only `--solver isam2`.*

### Decentralised controls

| Flag | Purpose |
| --- | --- |
| `--ddf-rounds`, `--ddf-convergence`, `--ddf-rot-convergence` | Iteration and convergence thresholds for interface updates. |
| `--ddf-local-iters`, `--ddf-relaxation` | LM iterations per robot and relaxation factor (1.0 disables relaxation). |
| `--ddf-transport {inproc,ros2}` | Transport for interface summaries. |
| `--ddf-ros2-topic-prefix` | ROS 2 topic prefix for interfaces (default `/cosmo/iface`). |
| `--ddf-ros2-reliability`, `--ddf-ros2-durability`, `--ddf-ros2-depth` | ROS 2 QoS for decentralised transport. |
| `--ddf-multiprocess` | Spawn one process per agent (requires `--ddf-transport ros2`). |
| `--ddf-ingest-time-scale`, `--ddf-ingest-max-sleep` | Pace factor ingestion using dataset stamps (0 disables). |
| `--ddf-interface-time-scale`, `--ddf-interface-max-sleep` | Pace interface broadcasts (0 disables). |

## Outputs

| Artifact | Description |
| --- | --- |
| `trajectories/trajectory_<robot>.csv` | Pose traces per robot (quaternion + translation). |
| `graph_stats.json` | Factor counts and final graph error (centralised runs). |
| `decentralised_stats.json` | Iteration counts, convergence flags, delivered messages (decentralised runs). |
| `trajectories_xy.png`, `trajectories_3d.png` | Quick-look trajectory plots (centralised and aggregated decentralised outputs). |
| `trajectories_xy_est_vs_gt.png`, `trajectories_3d_est_vs_gt.png` | Produced when `--eval-gt` is set. |
| `gt_metrics.json`, `rpe_metrics.csv` | Ground-truth alignment metrics (when enabled). |
| `kpi_metrics/kpi_events.jsonl` | Structured timeline emitted by `KPILogger`. |
| `kpi_metrics/latency_metrics.json` | Latency histograms and per-event entries (ingest/use/broadcast). |
| `kpi_metrics/bandwidth_stats.json` | Topic-level uplink/downlink counters in bytes/messages. |
| `kpi_metrics/resource_profile.json` | CPU, RSS, (optional) GPU samples plus metadata. |
| `kpi_metrics/robustness_metrics.json` | Network impairment summary when `cosmo_slam_ros2.impair` is active. |
| `kpi_metrics/derived_kpis.json` | Optional derived metrics written by `tools/export_kpis.py`. |

## KPI instrumentation

- **LatencyTracker** (`cosmo_slam_centralised.latency`) captures ingest→optimise,
  ingest→broadcast, and end-to-end send→use latencies. ROS 2 paths populate message IDs
  and producer timestamps so ACK topics can close the loop.
- **BandwidthTracker** (`cosmo_slam_centralised.bandwidth`) records estimated payload sizes
  for in-process runs and byte-accurate counts for ROS 2 topics.
- **ResourceMonitor** (`cosmo_slam_centralised.resource_monitor`) samples CPU/RSS for the
  main process and any extra PIDs supplied via `COSMO_RESOURCE_PIDS`. GPU stats are
  collected when `pynvml` is available.
- **KPILogger** (`cosmo_slam_centralised.kpi_logging`) writes JSONL events used by
  downstream aggregation (`tools/export_kpis.py`, `tools/compare_backends.py`).
- **Derived KPIs** (`cosmo_slam_centralised.kpi_derive`) extract loop-closure correction
  times, decentralised interface correction times, and coarse convergence times from
  existing JSON exports.

## ROS 2 helpers

- `cosmo_slam_ros2.factor_batch` encodes/decodes binary factor batches that match the CLI
  expectations.
- `cosmo_slam_ros2.interface_msg` serialises decentralised interface summaries with
  metadata for latency KPIs.
- `cosmo_slam_ros2.ack.Ros2AckPublisher` publishes ACK payloads for send→use latency
  tracking.
- `cosmo_slam_ros2.qos.parse_qos_options` turns CLI strings into `rclpy` QoS profiles.
- `cosmo_slam_ros2.sim_time.configure_sim_time` applies `--use-sim-time` automatically
  when the flag/environment variable is set.
- `cosmo_slam_ros2.impair.ImpairmentPolicy` can throttle or drop ROS 2 messages based on
  specs supplied via `COSMO_IMPAIR` or `COSMO_IMPAIR_FILE`, and writes robustness
  summaries alongside KPI metrics.

## Experiment automation & analysis

- `tools/ros2_factor_publisher.py` replays `.jrl` factors over ROS 2 with pacing controls
  (`--time-scale`, `--max-sleep`) and completion signalling.
- `tools/orchestrate_kpi.py` launches coordinated centralised and decentralised KPI runs
  (ROS 2 required), applies optional impairments, and summarises statistics with
  confidence intervals.
- `tools/export_kpis.py` computes derived KPIs per run and produces long-format CSVs for
  loop-closure correction, interface correction, and convergence times.
- `tools/compare_backends.py` aggregates latency/bandwidth/resource metrics across output
  directories and plots comparisons (Matplotlib optional).
- `tools/plot_scaling.py` and the `scaling_trends/` artifacts illustrate how to
  post-process KPI CSVs from batch experiments.

## Datasets

Sample COSMO-bench datasets are checked in under `datasets/COSMO/`:

- `datasets/COSMO/proradio/*.jrl` – protonet radio logs.
- `datasets/COSMO/wifi/*.jrl` – Wi-Fi logs.

File names encode the site, robots, and modality. Runs expect the `.jrl` schema described
by COSMO-bench; the loader handles per-robot lists, flat lists, and minor schema variants,
and can include potential outliers when `--include-potential-outliers` is set.

## Testing

```bash
pytest -q
```

- Tests that require GTSAM auto-skip when the bindings are unavailable.
- ROS 2-dependent tests (`tests/test_source_ros2.py`, `tests/test_ros2_peer_bus.py`)
  expect `rclpy` to import and will skip otherwise.

## Environment knobs

- `COSMO_RUN_SEED`: seed the random number generators used for reproducibility in both
  backends.
- `COSMO_RESOURCE_INTERVAL`: override the resource sampling period (seconds).
- `COSMO_RESOURCE_PIDS`: comma-separated list of extra PIDs whose CPU/RSS usage should be
  aggregated.
- `COSMO_USE_SIM_TIME`: honour ROS 2 simulation time (also set automatically by
  `--use-sim-time`).
- `COSMO_IMPAIR` / `COSMO_IMPAIR_FILE`: JSON spec or path configuring ROS 2 network
  impairments.
- `COSMO_IMPAIR_OUT_DIR`: where impairment summaries (`robustness_metrics.json`) are
  written.
- `ROS_DOMAIN_ID`: recorded in KPI metadata when ROS 2 transports are used.
- `COSMO_LOG`: logging level consumed by sub-processes spawned in decentralised
  multi-process runs.

## Current status & limitations

- The repository is not yet packaged for PyPI; run the tools from the checkout.
- Centralised ROS 2 ingest currently supports only `--solver isam2`.
- Decentralised multi-process runs reconstruct a merged snapshot from per-agent CSVs; the
  intermediate per-iteration values are not persisted.
- ROS 2 impairments are opt-in and only affect transports that go through `Ros2PeerBus`
  or the factor batch subscriber.
- KPI derivations assume monotonic timestamps; ensure system clocks are steady when
  running across multiple machines.
