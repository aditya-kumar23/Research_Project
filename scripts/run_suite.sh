#!/usr/bin/env bash
set -euo pipefail

# Comparative suite: centralised and decentralised ROS 2 runs
# across R3/R4/R5 Wi‑Fi and Pro‑Radio datasets, with 5 replicas,
# fixed seed, sim time enabled, and 10 Hz resource sampling.

# Optional: set your ROS domain to match your environment
# export ROS_DOMAIN_ID=23

REPLICAS=1
SEED_BASE=1000
RES_INTERVAL=0.1

# Common orchestrator flags
COMMON_FLAGS=(
  --replicas "$REPLICAS"
  --seed-base "$SEED_BASE"
  --use-sim-time
  --resource-interval "$RES_INTERVAL"
  --eval-gt
)

# Datasets
R3_WIFI="datasets/COSMO/wifi/ntu_r3_01_n04_n08_n13_wifi.jrl"
R3_PRO="datasets/COSMO/proradio/ntu_r3_01_n04_n08_n13_proradio.jrl"

R4_WIFI="datasets/COSMO/wifi/ntu_r4_00_d01_n04_n08_n13_wifi.jrl"
R4_PRO="datasets/COSMO/proradio/ntu_r4_00_d01_n04_n08_n13_proradio.jrl"

R5_WIFI="datasets/COSMO/wifi/ntu_r5_00_d01_d02_n04_n08_n13_wifi.jrl"
R5_PRO="datasets/COSMO/proradio/ntu_r5_00_d01_d02_n04_n08_n13_proradio.jrl"

# Impairments for bandwidth-cap runs (no random loss/bursts, just caps)
IMPAIR_CAP10='{"seed":12345,"warmup_s":0,"random_loss_p":0.0,"burst_period_s":0,"burst_duration_s":0,"bw_caps_mbps":{"default":10}}'
IMPAIR_CAP5='{"seed":12345,"warmup_s":0,"random_loss_p":0.0,"burst_period_s":0,"burst_duration_s":0,"bw_caps_mbps":{"default":5}}'

run_case() {
  local jrl="$1"
  local out_base="$2"
  shift 2
  local impair_opts=("$@")
  echo "=== Running: $jrl -> $out_base ==="
  python3 tools/orchestrate_kpi.py --jrl "$jrl" --out-base "$out_base" "${COMMON_FLAGS[@]}" "${impair_opts[@]}"
}

# Baseline (no impairments)
run_case "$R3_WIFI" "output_kpi_runs_r3_wifi_base"
run_case "$R3_PRO"  "output_kpi_runs_r3_proradio_base"
run_case "$R4_WIFI" "output_kpi_runs_r4_wifi_base"
run_case "$R4_PRO"  "output_kpi_runs_r4_proradio_base"
run_case "$R5_WIFI" "output_kpi_runs_r5_wifi_base"
run_case "$R5_PRO"  "output_kpi_runs_r5_proradio_base"

# 10 Mbps cap (per agent)
run_case "$R3_WIFI" "output_kpi_runs_r3_wifi_cap10"      --impair-json "$IMPAIR_CAP10"
run_case "$R3_PRO"  "output_kpi_runs_r3_proradio_cap10"  --impair-json "$IMPAIR_CAP10"
run_case "$R4_WIFI" "output_kpi_runs_r4_wifi_cap10"      --impair-json "$IMPAIR_CAP10"
run_case "$R4_PRO"  "output_kpi_runs_r4_proradio_cap10"  --impair-json "$IMPAIR_CAP10"
run_case "$R5_WIFI" "output_kpi_runs_r5_wifi_cap10"      --impair-json "$IMPAIR_CAP10"
run_case "$R5_PRO"  "output_kpi_runs_r5_proradio_cap10"  --impair-json "$IMPAIR_CAP10"

# 5 Mbps cap (per agent)
run_case "$R3_WIFI" "output_kpi_runs_r3_wifi_cap5"      --impair-json "$IMPAIR_CAP5"
run_case "$R3_PRO"  "output_kpi_runs_r3_proradio_cap5"  --impair-json "$IMPAIR_CAP5"
run_case "$R4_WIFI" "output_kpi_runs_r4_wifi_cap5"      --impair-json "$IMPAIR_CAP5"
run_case "$R4_PRO"  "output_kpi_runs_r4_proradio_cap5"  --impair-json "$IMPAIR_CAP5"
run_case "$R5_WIFI" "output_kpi_runs_r5_wifi_cap5"      --impair-json "$IMPAIR_CAP5"
run_case "$R5_PRO"  "output_kpi_runs_r5_proradio_cap5"  --impair-json "$IMPAIR_CAP5"

echo "All runs completed."

