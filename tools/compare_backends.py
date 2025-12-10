#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Iterable

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def read_json(path: Path) -> Any:
    with open(path, 'r') as f:
        return json.load(f)


def safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_json_if_exists(dir_path: Path, relative: Iterable[str]) -> Dict[str, Any]:
    if isinstance(relative, (str, Path)):
        relative = [relative]
    p = dir_path
    for part in relative:
        p = p / part
    if p.exists():
        return read_json(p)
    return {}


def load_gt_metrics(dir_path: Path) -> Dict[str, Any]:
    return load_json_if_exists(dir_path, 'gt_metrics.json')


def load_latency(dir_path: Path) -> Dict[str, Any]:
    return load_json_if_exists(dir_path, ('kpi_metrics', 'latency_metrics.json'))


def load_bandwidth(dir_path: Path) -> Dict[str, Any]:
    return load_json_if_exists(dir_path, ('kpi_metrics', 'bandwidth_stats.json'))


def load_resource(dir_path: Path) -> Dict[str, Any]:
    return load_json_if_exists(dir_path, ('kpi_metrics', 'resource_profile.json'))


def load_decentralised_stats(dir_path: Path) -> Dict[str, Any]:
    return load_json_if_exists(dir_path, 'decentralised_stats.json')


def load_graph_stats(dir_path: Path) -> Dict[str, Any]:
    return load_json_if_exists(dir_path, 'graph_stats.json')


def load_events_duration(dir_path: Path) -> Tuple[int, float]:
    p = dir_path / 'kpi_metrics' / 'kpi_events.jsonl'
    if not p.exists():
        return 0, 0.0
    first = None
    last = None
    n = 0
    with open(p, 'r') as f:
        for line in f:
            n += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ts = obj.get('ts')
            if ts is None:
                continue
            if first is None:
                first = ts
            last = ts
    dur = (last - first) if (first is not None and last is not None) else 0.0
    return n, float(dur)


def aggregate_bandwidth(bw: Dict[str, Any]) -> Dict[str, Any]:
    uplink = bw.get('uplink', {}) if isinstance(bw, dict) else {}
    downlink = bw.get('downlink', {}) if isinstance(bw, dict) else {}

    def sum_msgs_bytes(bucket: Dict[str, Any], key_prefix: str = None):
        msgs = 0
        bytes_ = 0
        for k, v in bucket.items():
            if key_prefix is not None and not k.startswith(key_prefix):
                continue
            if isinstance(v, dict):
                msgs += int(v.get('messages', 0))
                bytes_ += int(v.get('bytes', 0))
        return msgs, bytes_

    up_msgs, up_bytes = sum_msgs_bytes(uplink)
    down_msgs, down_bytes = sum_msgs_bytes(downlink)

    # Specific categories
    iface_msgs, iface_bytes = sum_msgs_bytes(uplink, 'iface/')
    between_msgs, between_bytes = sum_msgs_bytes(uplink, 'between/')
    prior_msgs, prior_bytes = sum_msgs_bytes(uplink, 'prior/')

    map_broadcast_msgs = 0
    map_broadcast_bytes = 0
    for name, bucket in downlink.items():
        if name.startswith('map_broadcast'):
            if isinstance(bucket, dict):
                map_broadcast_msgs += int(bucket.get('messages', 0))
                map_broadcast_bytes += int(bucket.get('bytes', 0))

    return {
        'uplink_messages': up_msgs,
        'uplink_bytes': up_bytes,
        'downlink_messages': down_msgs,
        'downlink_bytes': down_bytes,
        'iface_messages': iface_msgs,
        'iface_bytes': iface_bytes,
        'between_messages': between_msgs,
        'between_bytes': between_bytes,
        'prior_messages': prior_msgs,
        'prior_bytes': prior_bytes,
        'map_broadcast_messages': map_broadcast_msgs,
        'map_broadcast_bytes': map_broadcast_bytes,
    }


def summarise_gt(gt: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise per-robot GT metrics (rmse, matches) for any robot keys present."""
    out = {}
    rmses = []
    weighted_sum = 0.0
    weight_total = 0.0
    # Detect robot IDs dynamically from keys in gt (e.g., 'a','b','c','d', ...)
    robot_keys = [k for k, v in gt.items() if isinstance(v, dict)] if isinstance(gt, dict) else []
    for robot in sorted(robot_keys):
        r = gt.get(robot)
        if not isinstance(r, dict):
            continue
        rmse = r.get('rmse')
        matches = r.get('matches', 0)
        if rmse is not None:
            out[f'rmse_{robot}'] = float(rmse)
            rmses.append(float(rmse))
            weighted_sum += float(rmse) * float(matches or 0)
            weight_total += float(matches or 0)
        if matches is not None:
            try:
                out[f'matches_{robot}'] = int(matches)
            except Exception:
                # Fallback if matches is non-integer numeric
                out[f'matches_{robot}'] = int(float(matches) if matches is not None else 0)
    if rmses:
        out['rmse_mean'] = sum(rmses) / len(rmses)
    if weight_total > 0:
        out['rmse_weighted'] = weighted_sum / weight_total
    return out


def summarise_latency(lat: Dict[str, Any]) -> Dict[str, Any]:
    s = lat.get('summary', {}) if isinstance(lat, dict) else {}
    out = {}
    events = s.get('events')
    if events is not None:
        out['latency_events'] = int(events)
    for name, bucket in s.items():
        if name == 'events':
            continue
        if not isinstance(bucket, dict):
            continue
        for stat, val in bucket.items():
            key = f'{name}_{stat}'
            if stat == 'count':
                try:
                    out[key] = int(val)
                except Exception:
                    out[key] = float(val)
            else:
                out[key] = float(val)
    return out


def summarise_resource(res: Dict[str, Any]) -> Dict[str, Any]:
    s = res.get('summary', {}) if isinstance(res, dict) else {}
    out = {}
    for metric in ['cpu_process_pct', 'cpu_system_pct', 'rss_bytes']:
        bucket = s.get(metric, {})
        if not isinstance(bucket, dict):
            continue
        for stat in ['mean', 'median', 'p90', 'p95', 'max']:
            if stat in bucket:
                out[f'{metric}_{stat}'] = float(bucket[stat])
    meta = res.get('metadata', {}) if isinstance(res, dict) else {}
    for mk in ['solver', 'ddf_rounds', 'ddf_convergence', 'ddf_rot_convergence', 'ddf_local_iters', 'batch_size', 'relinearize_skip', 'robots', 'multiprocess', 'transport', 'ddf_transport']:
        if mk in meta:
            out[f'meta_{mk}'] = meta[mk]
    return out


def summarise_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(graph, dict):
        return out
    factors = graph.get('factors')
    if isinstance(factors, dict):
        for k, v in factors.items():
            out[f'factors_{k}'] = v
    if 'final_error' in graph:
        out['final_error'] = graph['final_error']
    return out


def summarise_decentralised(stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(stats, dict):
        return out
    for key in ['iterations', 'converged', 'messages_delivered', 'kpi_event_count']:
        if key in stats:
            out[f'ddf_{key}'] = stats[key]
    agents = stats.get('agents', {})
    if isinstance(agents, dict):
        total_poses = 0
        for rid, info in agents.items():
            if not isinstance(info, dict):
                continue
            poses = info.get('poses')
            if poses is not None:
                total_poses += int(poses)
                out[f'ddf_agent_{rid}_poses'] = int(poses)
            iters = info.get('iterations')
            if iters is not None:
                out[f'ddf_agent_{rid}_iterations'] = int(iters)
            conv = info.get('converged')
            if conv is not None:
                out[f'ddf_agent_{rid}_converged'] = bool(conv)
        out['ddf_total_poses'] = total_poses
    return out


def compare_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out = {}
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        diff = None
        ratio = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            diff = vb - va
            ratio = (vb / va) if va not in (0, 0.0) else None
        out[k] = {
            'centralised': va,
            'decentralised': vb,
            'delta': diff,
            'ratio': ratio,
        }
    return out


def write_csv_from_comparison(comp: Dict[str, Dict[str, Any]], out_csv: Path) -> None:
    # Flatten into rows: metric, centralised, decentralised, delta, ratio
    with open(out_csv, 'w') as f:
        f.write('metric,centralised,decentralised,delta,ratio\n')
        for metric, vals in comp.items():
            cen = vals.get('centralised')
            dec = vals.get('decentralised')
            delta = vals.get('delta')
            ratio = vals.get('ratio')
            def fm(v):
                if v is None:
                    return ''
                if isinstance(v, float):
                    return f'{v:.6g}'
                return str(v)
            f.write(f'{metric},{fm(cen)},{fm(dec)},{fm(delta)},{fm(ratio)}\n')


def plot_simple_bars(central: Dict[str, Any], decentral: Dict[str, Any], out_dir: Path) -> None:
    if not HAVE_MPL:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize(seq):
        out = []
        for v in seq:
            if isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, bool):
                out.append(float(v))
            elif v is None:
                out.append(0.0)
            else:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(0.0)
        return out

    # 1) ATE RMSE per robot (detect robots dynamically from rmse_* keys)
    def rmse_robot_ids(d: Dict[str, Any]):
        return {k.split('_', 1)[1] for k in d.keys() if isinstance(k, str) and k.startswith('rmse_') and len(k.split('_', 1)) == 2}
    robots = sorted(rmse_robot_ids(central) | rmse_robot_ids(decentral))
    if robots:
        cen_vals = _sanitize([central.get(f'rmse_{r}') for r in robots])
        dec_vals = _sanitize([decentral.get(f'rmse_{r}') for r in robots])
        if any(v is not None for v in cen_vals + dec_vals):
            x = list(range(len(robots)))
            w = 0.35
            plt.figure(figsize=(max(6, 1.5*len(robots)), 4))
            plt.bar([i - w/2 for i in x], cen_vals, width=w, label='Centralised')
            plt.bar([i + w/2 for i in x], dec_vals, width=w, label='Decentralised')
            plt.xticks(x, robots)
            plt.ylabel('ATE RMSE')
            plt.title('ATE RMSE per robot')
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / 'rmse_per_robot.png', dpi=160)
            plt.close()

    # 2) Latency means
    lat_metrics = ['ingest_to_opt_mean', 'ingest_to_broadcast_mean', 'e2e_send_to_ingest_mean', 'e2e_send_to_use_mean']
    labels = ['ingest→opt', 'ingest→broadcast', 'send→ingest', 'send→use']
    cen_lat = _sanitize([central.get(m) for m in lat_metrics])
    dec_lat = _sanitize([decentral.get(m) for m in lat_metrics])
    if any(v is not None for v in cen_lat + dec_lat):
        x = list(range(len(labels)))
        w = 0.35
        plt.figure(figsize=(6,4))
        plt.bar([i - w/2 for i in x], cen_lat, width=w, label='Centralised')
        plt.bar([i + w/2 for i in x], dec_lat, width=w, label='Decentralised')
        plt.xticks(x, labels)
        plt.ylabel('Latency (s)')
        plt.title('Latency (mean)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'latency_means.png', dpi=160)
        plt.close()

    # 3) Bandwidth totals
    bw_labels = ['uplink_bytes', 'downlink_bytes', 'iface_bytes', 'between_bytes', 'prior_bytes']
    cen_bw = _sanitize([central.get(l) for l in bw_labels])
    dec_bw = _sanitize([decentral.get(l) for l in bw_labels])
    if any(v is not None for v in cen_bw + dec_bw):
        x = list(range(len(bw_labels)))
        w = 0.35
        plt.figure(figsize=(8,4))
        plt.bar([i - w/2 for i in x], cen_bw, width=w, label='Centralised')
        plt.bar([i + w/2 for i in x], dec_bw, width=w, label='Decentralised')
        plt.xticks(x, [s.replace('_bytes','') for s in bw_labels], rotation=20)
        plt.ylabel('Bytes')
        plt.title('Bandwidth totals')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'bandwidth_totals.png', dpi=160)
        plt.close()

    # 4) CPU / RSS
    cpu_labels = ['cpu_process_pct_mean', 'cpu_system_pct_mean']
    cen_cpu = _sanitize([central.get(m) for m in cpu_labels])
    dec_cpu = _sanitize([decentral.get(m) for m in cpu_labels])
    if any(v is not None for v in cen_cpu + dec_cpu):
        x = list(range(len(cpu_labels)))
        w = 0.35
        plt.figure(figsize=(6,4))
        plt.bar([i - w/2 for i in x], cen_cpu, width=w, label='Centralised')
        plt.bar([i + w/2 for i in x], dec_cpu, width=w, label='Decentralised')
        plt.xticks(x, ['proc_cpu%', 'sys_cpu%'])
        plt.ylabel('Percent')
        plt.title('CPU (mean)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'cpu_means.png', dpi=160)
        plt.close()

    rss_labels = ['rss_bytes_mean']
    cen_rss = _sanitize([central.get(m) for m in rss_labels])
    dec_rss = _sanitize([decentral.get(m) for m in rss_labels])
    if any(v is not None for v in cen_rss + dec_rss):
        x = [0]
        w = 0.35
        plt.figure(figsize=(5,4))
        plt.bar([i - w/2 for i in x], cen_rss, width=w, label='Centralised')
        plt.bar([i + w/2 for i in x], dec_rss, width=w, label='Decentralised')
        plt.xticks(x, ['rss_mean'])
        plt.ylabel('Bytes')
        plt.title('RSS (mean)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'rss_mean.png', dpi=160)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description='Compare KPI outputs between centralised and decentralised backends.')
    ap.add_argument('--base', default='/home/gabruuu/cosmo_slam/output_kpi_runs', help='Base directory containing centralised/decentralised runs')
    ap.add_argument('--centralised', help='Path to centralised output folder (default: <base>/central_ros2)')
    ap.add_argument('--decentralised', help='Path to decentralised output folder (default: <base>/ddf_ros2_mp_paced)')
    ap.add_argument('--out', help='Directory to write comparison results (default: <base>/comparison)')
    args = ap.parse_args()

    base_dir = Path(args.base).expanduser() if args.base else Path.cwd()
    cen_dir = Path(args.centralised).expanduser() if args.centralised else base_dir / 'central_ros2'
    dec_dir = Path(args.decentralised).expanduser() if args.decentralised else base_dir / 'ddf_ros2_mp_paced'
    out_dir = Path(args.out).expanduser() if args.out else base_dir / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    cen_gt = load_gt_metrics(cen_dir)
    dec_gt = load_gt_metrics(dec_dir)
    cen_lat = load_latency(cen_dir)
    dec_lat = load_latency(dec_dir)
    cen_bw = load_bandwidth(cen_dir)
    dec_bw = load_bandwidth(dec_dir)
    cen_res = load_resource(cen_dir)
    dec_res = load_resource(dec_dir)
    cen_events, cen_dur = load_events_duration(cen_dir)
    dec_events, dec_dur = load_events_duration(dec_dir)
    cen_graph = load_graph_stats(cen_dir)
    dec_ddf_stats = load_decentralised_stats(dec_dir)

    # Summarise
    cen_gt_s = summarise_gt(cen_gt)
    dec_gt_s = summarise_gt(dec_gt)
    cen_lat_s = summarise_latency(cen_lat)
    dec_lat_s = summarise_latency(dec_lat)
    cen_bw_s = aggregate_bandwidth(cen_bw)
    dec_bw_s = aggregate_bandwidth(dec_bw)
    cen_res_s = summarise_resource(cen_res)
    dec_res_s = summarise_resource(dec_res)
    cen_graph_s = summarise_graph(cen_graph)
    dec_ddf_s = summarise_decentralised(dec_ddf_stats)

    # Add events/duration/throughput
    cen_misc = {
        'events_count': cen_events,
        'duration_seconds': cen_dur,
    }
    dec_misc = {
        'events_count': dec_events,
        'duration_seconds': dec_dur,
    }
    # Bytes per second (uplink + downlink) if duration > 0
    for tgt, bw_s, misc in ((cen_dir, cen_bw_s, cen_misc), (dec_dir, dec_bw_s, dec_misc)):
        dur = misc['duration_seconds']
        total_bytes = (bw_s.get('uplink_bytes', 0) or 0) + (bw_s.get('downlink_bytes', 0) or 0)
        misc['throughput_bytes_per_s'] = (total_bytes / dur) if dur > 0 else None

    # Merge summaries under namespaces
    def prefixed(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {f'{prefix}{k}': v for k, v in d.items()}

    # Build comparison per metric key (no prefix) — align dictionaries
    # Gather centralised metrics into one flat dict
    cen_all = {}
    dec_all = {}
    for dd, src in ((cen_all, cen_gt_s), (cen_all, cen_lat_s), (cen_all, cen_bw_s), (cen_all, cen_res_s), (cen_all, cen_misc), (cen_all, cen_graph_s)):
        dd.update(src)
    for dd, src in ((dec_all, dec_gt_s), (dec_all, dec_lat_s), (dec_all, dec_bw_s), (dec_all, dec_res_s), (dec_all, dec_misc), (dec_all, dec_ddf_s)):
        dd.update(src)

    comparison = compare_dicts(cen_all, dec_all)

    # Write outputs
    with open(out_dir / 'comparison_summary.json', 'w') as f:
        json.dump({
            'centralised': cen_all,
            'decentralised': dec_all,
            'comparison': comparison,
        }, f, indent=2)

    write_csv_from_comparison(comparison, out_dir / 'comparison_summary.csv')

    # A few quick plots
    plot_simple_bars(cen_all, dec_all, out_dir)

    # Also write a short human-readable summary
    lines = []
    def fmt(v):
        if v is None:
            return 'n/a'
        if isinstance(v, float):
            return f'{v:.4g}'
        return str(v)
    lines.append('KPI Comparison (centralised vs decentralised)')
    lines.append('')
    lines.append(f"ATE RMSE mean: {fmt(cen_all.get('rmse_mean'))} vs {fmt(dec_all.get('rmse_mean'))}")
    lines.append(f"Latency ingest→opt mean: {fmt(cen_all.get('ingest_to_opt_mean'))} s vs {fmt(dec_all.get('ingest_to_opt_mean'))} s")
    lines.append(f"Latency ingest→broadcast mean: {fmt(cen_all.get('ingest_to_broadcast_mean'))} s vs {fmt(dec_all.get('ingest_to_broadcast_mean'))} s")
    lines.append(f"E2E send→ingest mean: {fmt(cen_all.get('e2e_send_to_ingest_mean'))} s vs {fmt(dec_all.get('e2e_send_to_ingest_mean'))} s")
    lines.append(f"E2E send→use mean: {fmt(cen_all.get('e2e_send_to_use_mean'))} s vs {fmt(dec_all.get('e2e_send_to_use_mean'))} s")
    lines.append(f"Uplink bytes: {fmt(cen_all.get('uplink_bytes'))} vs {fmt(dec_all.get('uplink_bytes'))}")
    lines.append(f"Downlink bytes: {fmt(cen_all.get('downlink_bytes'))} vs {fmt(dec_all.get('downlink_bytes'))}")
    lines.append(f"Iface bytes: {fmt(cen_all.get('iface_bytes'))} vs {fmt(dec_all.get('iface_bytes'))}")
    lines.append(f"Map broadcast bytes: {fmt(cen_all.get('map_broadcast_bytes'))} vs {fmt(dec_all.get('map_broadcast_bytes'))}")
    lines.append(f"Throughput (bytes/s): {fmt(cen_all.get('throughput_bytes_per_s'))} vs {fmt(dec_all.get('throughput_bytes_per_s'))}")
    lines.append(f"CPU process % (mean): {fmt(cen_all.get('cpu_process_pct_mean'))} vs {fmt(dec_all.get('cpu_process_pct_mean'))}")
    lines.append(f"RSS mean (bytes): {fmt(cen_all.get('rss_bytes_mean'))} vs {fmt(dec_all.get('rss_bytes_mean'))}")
    if dec_all.get('ddf_messages_delivered') is not None:
        lines.append(f"DDF messages delivered: {fmt(dec_all.get('ddf_messages_delivered'))}")
    if dec_all.get('ddf_iterations') is not None:
        lines.append(f"DDF iterations (global): {fmt(dec_all.get('ddf_iterations'))}")
    with open(out_dir / 'summary.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Wrote: {out_dir}/comparison_summary.json, comparison_summary.csv, summary.txt')
    if HAVE_MPL:
        print('Plots:', 'rmse_per_robot.png, latency_means.png, bandwidth_totals.png, cpu_means.png, rss_mean.png')
    else:
        print('matplotlib not available; skipped plots')


if __name__ == '__main__':
    main()
