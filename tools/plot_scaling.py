import json
import os
from typing import Dict, Any, List, Tuple

import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DATASETS = [
    (3, os.path.join(BASE_DIR, 'comparison_r3_C_vs_D', 'comparison_summary.json')),
    (4, os.path.join(BASE_DIR, 'comparison_r4_C_vs_D', 'comparison_summary.json')),
    (5, os.path.join(BASE_DIR, 'comparison_r5_C_vs_D', 'comparison_summary.json')),
]

OUT_DIR = os.path.join(BASE_DIR, 'scaling_trends')


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def per_robot_rmses(d: Dict[str, Any]) -> List[float]:
    vals = []
    for k, v in d.items():
        if k.startswith('rmse_') and k not in ('rmse_mean', 'rmse_weighted'):
            # ensure it's a per-robot key like rmse_a/b/c...
            if isinstance(v, (int, float)):
                vals.append(float(v))
    return vals


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def collect_metrics() -> Dict[int, Dict[str, Dict[str, float]]]:
    metrics: Dict[int, Dict[str, Dict[str, float]]] = {}
    for robots, path in DATASETS:
        js = load_json(path)
        C = js['centralised']
        D = js['decentralised']

        def robust(d: Dict[str, Any]) -> Tuple[float, float]:
            vals = per_robot_rmses(d)
            if not vals:
                return (float('nan'), float('nan'))
            mx = max(vals)
            mean = sum(vals)/len(vals)
            var = sum((x-mean)**2 for x in vals)/len(vals)
            stdev = math.sqrt(var)
            return (mx, stdev)

        c_max, c_std = robust(C)
        d_max, d_std = robust(D)

        metrics[robots] = {
            'rmse_mean': {
                'C': C.get('rmse_mean'),
                'D': D.get('rmse_mean'),
            },
            'rmse_max': {
                'C': c_max,
                'D': d_max,
            },
            'rmse_stdev': {
                'C': c_std,
                'D': d_std,
            },
            'latency_mean_s': {
                'C': C.get('ingest_to_opt_mean'),
                'D': D.get('ingest_to_opt_mean'),
            },
            'cpu_process_mean': {
                'C': C.get('cpu_process_pct_mean'),
                'D': D.get('cpu_process_pct_mean'),
            },
            'cpu_process_p95': {
                'C': C.get('cpu_process_pct_p95'),
                'D': D.get('cpu_process_pct_p95'),
            },
            'rss_mean_bytes': {
                'C': C.get('rss_bytes_mean'),
                'D': D.get('rss_bytes_mean'),
            },
            'throughput_bytes_per_s': {
                'C': C.get('throughput_bytes_per_s'),
                'D': D.get('throughput_bytes_per_s'),
            },
            'uplink_bytes': {
                'C': C.get('uplink_bytes'),
                'D': D.get('uplink_bytes'),
            },
            'downlink_bytes': {
                'C': C.get('downlink_bytes'),
                'D': D.get('downlink_bytes'),
            },
            'iface_bytes': {
                'C': C.get('iface_bytes'),
                'D': D.get('iface_bytes'),
            },
        }
    return metrics


def to_series(metrics: Dict[int, Dict[str, Dict[str, float]]], key: str, system: str) -> Tuple[List[int], List[float]]:
    xs = sorted(metrics.keys())
    ys = [metrics[n][key][system] for n in xs]
    return xs, ys


def plot_lines(xs: List[int], ys_dict: Dict[str, List[float]], title: str, ylabel: str, filename: str, logy: bool = False):
    plt.figure(figsize=(7, 4))
    for label, ys in ys_dict.items():
        plt.plot(xs, ys, marker='o', label=label)
    plt.title(title)
    plt.xlabel('Robots (N)')
    plt.ylabel(ylabel)
    if logy:
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def write_kpi_table(metrics: Dict[int, Dict[str, Dict[str, float]]]):
    # CSV summary
    csv_path = os.path.join(OUT_DIR, 'kpi_summary.csv')
    header = [
        'robots',
        'rmse_mean_C','rmse_mean_D',
        'rmse_max_C','rmse_max_D',
        'latency_mean_s_C','latency_mean_s_D',
        'comm_total_bytes_C','comm_total_bytes_D',
        'uplink_C','downlink_C','uplink_D','iface_D',
        'cpu_process_mean_C','cpu_process_mean_D',
        'cpu_process_p95_C','cpu_process_p95_D',
        'rss_mean_bytes_C','rss_mean_bytes_D',
        'throughput_Bps_C','throughput_Bps_D',
    ]
    with open(csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for n in sorted(metrics.keys()):
            m = metrics[n]
            comm_total_C = float(m['uplink_bytes']['C'] or 0) + float(m['downlink_bytes']['C'] or 0)
            comm_total_D = float(m['uplink_bytes']['D'] or 0) + float(m['iface_bytes']['D'] or 0)
            row = [
                n,
                m['rmse_mean']['C'], m['rmse_mean']['D'],
                m['rmse_max']['C'], m['rmse_max']['D'],
                m['latency_mean_s']['C'], m['latency_mean_s']['D'],
                comm_total_C, comm_total_D,
                m['uplink_bytes']['C'], m['downlink_bytes']['C'],
                m['uplink_bytes']['D'], m['iface_bytes']['D'],
                m['cpu_process_mean']['C'], m['cpu_process_mean']['D'],
                m['cpu_process_p95']['C'], m['cpu_process_p95']['D'],
                m['rss_mean_bytes']['C'], m['rss_mean_bytes']['D'],
                m['throughput_bytes_per_s']['C'], m['throughput_bytes_per_s']['D'],
            ]
            f.write(','.join(str(x) for x in row) + '\n')

    # Markdown table (compact, selected KPIs)
    md_path = os.path.join(OUT_DIR, 'kpi_summary.md')
    cols = [
        ('RMSE mean', 'rmse_mean'),
        ('Robust max RMSE', 'rmse_max'),
        ('Latency mean (s)', 'latency_mean_s'),
        ('Comm total bytes', None),
        ('CPU proc p95 (%)', 'cpu_process_p95'),
        ('RSS mean (bytes)', 'rss_mean_bytes'),
    ]
    with open(md_path, 'w') as f:
        f.write('| KPI \\ N | r3 C | r3 D | r4 C | r4 D | r5 C | r5 D |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|\n')
        for label, key in cols:
            if key is not None:
                vals = []
                for n in [3,4,5]:
                    vals.append(metrics[n][key]['C'])
                    vals.append(metrics[n][key]['D'])
            else:
                vals = []
                for n in [3,4,5]:
                    m = metrics[n]
                    c_total = float(m['uplink_bytes']['C'] or 0) + float(m['downlink_bytes']['C'] or 0)
                    d_total = float(m['uplink_bytes']['D'] or 0) + float(m['iface_bytes']['D'] or 0)
                    vals.append(c_total)
                    vals.append(d_total)
            f.write('| {} | {} | {} | {} | {} | {} | {} |\n'.format(
                label,
                *[f'{v:.3g}' if isinstance(v, (int, float)) else str(v) for v in vals]
            ))
    return csv_path, md_path


def main():
    ensure_out_dir()
    metrics = collect_metrics()

    xs, rmse_mean_C = to_series(metrics, 'rmse_mean', 'C')
    _, rmse_mean_D = to_series(metrics, 'rmse_mean', 'D')
    plot_lines(xs, {
        'C rmse_mean': rmse_mean_C,
        'D rmse_mean': rmse_mean_D,
    }, 'RMSE Mean vs Robots', 'ATE RMSE (m)', 'plot_rmse_mean.png', logy=False)

    xs, rmse_max_C = to_series(metrics, 'rmse_max', 'C')
    _, rmse_max_D = to_series(metrics, 'rmse_max', 'D')
    plot_lines(xs, {
        'C rmse_max': rmse_max_C,
        'D rmse_max': rmse_max_D,
    }, 'Robustness: Max Per-Robot RMSE', 'Max ATE RMSE (m)', 'plot_rmse_max.png', logy=False)

    xs, lat_C = to_series(metrics, 'latency_mean_s', 'C')
    _, lat_D = to_series(metrics, 'latency_mean_s', 'D')
    plot_lines(xs, {
        'C latency_mean': lat_C,
        'D latency_mean': lat_D,
    }, 'Latency Mean (ingest→opt)', 'Seconds (log scale)', 'plot_latency_mean_log.png', logy=True)

    # Communication components
    xs = sorted(metrics.keys())
    C_up = [metrics[n]['uplink_bytes']['C'] for n in xs]
    C_down = [metrics[n]['downlink_bytes']['C'] for n in xs]
    D_up = [metrics[n]['uplink_bytes']['D'] for n in xs]
    D_iface = [metrics[n]['iface_bytes']['D'] for n in xs]
    plot_lines(xs, {
        'C uplink': C_up,
        'C downlink': C_down,
        'D uplink': D_up,
        'D iface': D_iface,
    }, 'Communication Components vs Robots', 'Bytes', 'plot_comm_components.png', logy=False)

    # Communication totals
    C_total = [float(metrics[n]['uplink_bytes']['C'] or 0) + float(metrics[n]['downlink_bytes']['C'] or 0) for n in xs]
    D_total = [float(metrics[n]['uplink_bytes']['D'] or 0) + float(metrics[n]['iface_bytes']['D'] or 0) for n in xs]
    plot_lines(xs, {
        'C total': C_total,
        'D total': D_total,
    }, 'Total Communication vs Robots', 'Bytes', 'plot_comm_total.png', logy=False)

    # CPU process
    xs, cpu_mean_C = to_series(metrics, 'cpu_process_mean', 'C')
    _, cpu_mean_D = to_series(metrics, 'cpu_process_mean', 'D')
    xs, cpu_p95_C = to_series(metrics, 'cpu_process_p95', 'C')
    _, cpu_p95_D = to_series(metrics, 'cpu_process_p95', 'D')
    plot_lines(xs, {
        'C process mean': cpu_mean_C,
        'D process mean': cpu_mean_D,
        'C process p95': cpu_p95_C,
        'D process p95': cpu_p95_D,
    }, 'CPU Process Utilization', 'Percent (%)', 'plot_cpu_process.png', logy=False)

    # RSS memory
    xs, rss_C = to_series(metrics, 'rss_mean_bytes', 'C')
    _, rss_D = to_series(metrics, 'rss_mean_bytes', 'D')
    plot_lines(xs, {
        'C RSS mean': rss_C,
        'D RSS mean': rss_D,
    }, 'Memory (RSS Mean)', 'Bytes', 'plot_rss_mean.png', logy=False)

    # Throughput
    xs, thr_C = to_series(metrics, 'throughput_bytes_per_s', 'C')
    _, thr_D = to_series(metrics, 'throughput_bytes_per_s', 'D')
    plot_lines(xs, {
        'C throughput': thr_C,
        'D throughput': thr_D,
    }, 'Data Throughput', 'Bytes/s', 'plot_throughput.png', logy=False)

    csv_path, md_path = write_kpi_table(metrics)
    print('Wrote:', csv_path)
    print('Wrote:', md_path)


if __name__ == '__main__':
    main()

