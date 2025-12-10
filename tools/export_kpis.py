#!/usr/bin/env python3
"""Compute derived KPIs and export long-format CSVs.

Outputs per run (under <run>/kpi_metrics/):
- derived_kpis.json

Outputs under base directory (if --base provided):
- long/loop_closure_correction_events.csv (one row per event)
- long/convergence_times.csv (one row per run)

Examples:
  python tools/export_kpis.py --base output_kpi_runs
  python tools/export_kpis.py --run output_kpi_runs/central_ros2 --run output_kpi_runs/ddf_ros2_mp_paced
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

from cosmo_slam_centralised.kpi_derive import (
    derive_kpis_for_run,
    extract_loop_closure_correction_times,
    extract_loop_closure_correction_stabilised,
    extract_interface_correction_times,
    extract_interface_correction_stabilised,
    time_to_global_convergence,
    write_json,
    _read_json,  # type: ignore
    _load_robot_map_for_run,  # type: ignore
    _read_kpi_events,  # type: ignore
)


def _detect_system(run_dir: str) -> str:
    # Prefer explicit decentralised marker
    if os.path.exists(os.path.join(run_dir, "decentralised_stats.json")):
        return "D"
    name = os.path.basename(os.path.normpath(run_dir)).lower()
    if "ddf" in name or "decent" in name:
        return "D"
    return "C"


def _runs_from_base(base: str) -> List[str]:
    out: List[str] = []
    try:
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if os.path.isdir(p):
                out.append(p)
    except Exception:
        pass
    return sorted(out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute derived KPIs and export long-format CSVs.")
    ap.add_argument("--base", help="Base directory containing one or more run subdirectories")
    ap.add_argument("--run", action="append", help="Run directory (can be specified multiple times)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    runs: List[str] = []
    if args.run:
        runs.extend(args.run)
    if args.base:
        runs.extend(_runs_from_base(args.base))
    runs = [r for r in runs if r and os.path.isdir(r)]
    if not runs:
        print("No runs found. Provide --base or at least one --run.")
        return 2

    # Per-base long CSV writers
    base = args.base or os.path.dirname(os.path.commonpath(runs)) or os.getcwd()
    long_dir = os.path.join(base, "long")
    os.makedirs(long_dir, exist_ok=True)
    lcct_path = os.path.join(long_dir, "loop_closure_correction_events.csv")
    lcct_stab_path = os.path.join(long_dir, "loop_closure_correction_stabilised_events.csv")
    iface_path = os.path.join(long_dir, "interface_correction_events.csv")
    iface_stab_path = os.path.join(long_dir, "interface_correction_stabilised_events.csv")
    conv_path = os.path.join(long_dir, "convergence_times.csv")

    with open(lcct_path, "w", newline="", encoding="utf-8") as f_lcct, \
         open(lcct_stab_path, "w", newline="", encoding="utf-8") as f_lcct_stab, \
         open(iface_path, "w", newline="", encoding="utf-8") as f_iface, \
         open(iface_stab_path, "w", newline="", encoding="utf-8") as f_iface_stab, \
         open(conv_path, "w", newline="", encoding="utf-8") as f_conv:
        lcct_writer = csv.writer(f_lcct)
        lcct_stab_writer = csv.writer(f_lcct_stab)
        iface_writer = csv.writer(f_iface)
        iface_stab_writer = csv.writer(f_iface_stab)
        conv_writer = csv.writer(f_conv)
        lcct_writer.writerow(["run", "system", "value_s"])  # long format (one value per row)
        lcct_stab_writer.writerow(["run", "system", "value_s"])  # stabilised TLC per event
        iface_writer.writerow(["run", "system", "value_s"])  # decentralised analogue per event
        iface_stab_writer.writerow(["run", "system", "value_s"])  # stabilised decentralised analogue per event
        conv_writer.writerow(["run", "system", "time_to_convergence_s", "method"])  # one row per run

        for run in runs:
            system = _detect_system(run)
            # 1) Derive KPIs and persist JSON next to existing metrics
            derived = derive_kpis_for_run(run)
            out_json = os.path.join(run, "kpi_metrics", "derived_kpis.json")
            try:
                write_json(out_json, derived)
                print(f"Wrote: {out_json}")
            except Exception as exc:
                print(f"WARN: failed to write {out_json}: {exc}")

            # 2) Long rows for LCCT (event-level)
            try:
                lat = _read_json(os.path.join(run, "kpi_metrics", "latency_metrics.json"))  # type: ignore
                robot_map = _load_robot_map_for_run(run)
                for v in extract_loop_closure_correction_times(lat, robot_map=robot_map):
                    lcct_writer.writerow([run, system, f"{float(v):.9g}"])
                # Stabilised TLC values (requires KPI events)
                kpi_events = _read_kpi_events(os.path.join(run, "kpi_metrics", "kpi_events.jsonl"))  # type: ignore
                if kpi_events:
                    for v in extract_loop_closure_correction_stabilised(lat, kpi_events, robot_map=robot_map):
                        lcct_stab_writer.writerow([run, system, f"{float(v):.9g}"])
                # Decentralised analogue: interface correction time (if present)
                for v in extract_interface_correction_times(lat):
                    iface_writer.writerow([run, system, f"{float(v):.9g}"])
                # Stabilised analogue: requires KPI events
                if kpi_events:
                    for v in extract_interface_correction_stabilised(lat, kpi_events):
                        iface_stab_writer.writerow([run, system, f"{float(v):.9g}"])
            except Exception:
                # If missing or malformed, skip
                pass

            # 3) Long rows for time-to-convergence
            try:
                conv = time_to_global_convergence(run)
                conv_writer.writerow([
                    run,
                    system,
                    ("" if conv.seconds is None else f"{float(conv.seconds):.9g}"),
                    conv.method,
                ])
            except Exception:
                conv_writer.writerow([run, system, "", "error"])

    print(f"Wrote: {lcct_path}")
    print(f"Wrote: {iface_path}")
    print(f"Wrote: {lcct_stab_path}")
    print(f"Wrote: {iface_stab_path}")
    print(f"Wrote: {conv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
