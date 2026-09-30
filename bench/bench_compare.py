#!/usr/bin/env python3
"""
Compare OpenMP and CUDA using the results already produced by bench_omp.py
and bench_cuda.py (their metrics.txt files). Nothing is run.

Usage:
  ./bench/bench_compare.py OMP_DIR CUDA_DIR [--out-dir DIR]

  OMP_DIR    folder with the OpenMP metrics.txt  (e.g. results/omp)
  CUDA_DIR   folder with the CUDA metrics.txt    (e.g. results/cuda)
  --out-dir  where to save compare_metrics.txt and the plots
             (default: bench/results/compare)

Only the N present in both results are compared. For each N, OpenMP is
taken with p = 1, 12, 24 threads and with the best p (the fastest T(p)).
If the best p is the same reference for every N (e.g. always 24), no
separate line is drawn: the legend of that reference says "(= best)".

Metrics, for each N and each OpenMP reference p:
  CUDA speedup                 = T_omp(N, p) / T_cuda(N)
  CUDA speedup incl. overhead  = T_omp(N, p) / (T_cuda(N) + overhead_cuda(N))
  overhead: OpenMP estimated   = T_omp(N, p) - T_omp(N, 1) / p
            CUDA measured      = "Overhead time" (context, alloc, copies)
  overhead share: OpenMP       = overhead / T_omp(N, p)
                  CUDA         = overhead / (T_cuda + overhead)

Plots (x axis: number of points N):
  compare_speedup.png          CUDA speedup vs each OpenMP reference
  compare_speedup_total.png    same, with the CUDA overhead included
  compare_overhead.png         overhead [s]: CUDA vs OpenMP references
  compare_overhead_frac.png    overhead as a share of each version's time

Example:
  ./bench/bench_compare.py results/omp results/cuda --out-dir results/compare
"""

import argparse
import re
from pathlib import Path

REF_P = [1, 12, 24]                               # OpenMP reference thread counts
REF_COLORS = {1: "#2a78d6", 12: "#1baf7a", 24: "#eda100", "best": "#e87ba4"}
CUDA_COLOR = "#eb6834"
INK2 = "#52514e"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"


# ---------------------------------------------------------------------------
# reading the metrics
# ---------------------------------------------------------------------------

def read_omp(path):
    """{N: {p: {"time": T(p), "overhead": O(p)}}} from the OpenMP metrics.txt."""
    data, n = {}, None
    for line in open(path):
        m = re.search(r"\(N\s*=\s*([\d,]+)\)", line)
        if m:
            n = int(m.group(1).replace(",", ""))
            data[n] = {}
            continue
        cells = line.split()
        if n is not None and len(cells) == 5 and cells[0].isdigit():
            p, t, _speedup, _eff, ovh = cells
            data[n][int(p)] = {"time": float(t), "overhead": float(ovh)}
    if not data:
        raise SystemExit(f"no OpenMP results found in {path}")
    return data


def read_cuda(path):
    """{N: {"time", "overhead", "total"}} from the CUDA metrics.txt."""
    data = {}
    for line in open(path):
        cells = line.split()
        if len(cells) >= 6 and re.fullmatch(r"[\d,]+", cells[0]) and cells[1].isdigit():
            num = lambda s: None if s == "NA" else float(s)
            data[int(cells[0].replace(",", ""))] = {
                "time": num(cells[3]), "overhead": num(cells[4]), "total": num(cells[5])}
    if not data:
        raise SystemExit(f"no CUDA results found in {path}")
    return data


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare(omp, cuda):
    rows = []
    for n in sorted(set(omp) & set(cuda)):
        o, c = omp[n], cuda[n]
        best_p = min(o, key=lambda p: o[p]["time"])
        refs = {p: o[p] for p in REF_P if p in o}
        refs["best"] = o[best_p]
        row = {"n": n, "best_p": best_p, "cuda_time": c["time"],
               "cuda_overhead": c["overhead"], "cuda_total": c["total"],
               "cuda_overhead_frac": c["overhead"] / c["total"] if c["total"] else None}
        for key, r in refs.items():
            row[f"omp_time_{key}"] = r["time"]
            row[f"speedup_{key}"] = r["time"] / c["time"] if c["time"] else None
            row[f"speedup_total_{key}"] = r["time"] / c["total"] if c["total"] else None
            row[f"omp_overhead_{key}"] = r["overhead"]
            row[f"omp_overhead_frac_{key}"] = r["overhead"] / r["time"] if r["time"] else None
        rows.append(row)
    if not rows:
        raise SystemExit("no N in common between the OpenMP and CUDA results")
    return rows


def references(rows):
    """[(key, label)] of the OpenMP references to draw. If the best p is the
    same reference for every N, it is merged into that reference's label."""
    present = [p for p in REF_P if all(f"omp_time_{p}" in r for r in rows)]
    best = {r["best_p"] for r in rows}
    merged = best.pop() if len(best) == 1 and next(iter(best)) in present else None
    refs = [(p, f"OpenMP {p} thread{'s' if p > 1 else ''}" + (" (= best)" if p == merged else ""))
            for p in present]
    if merged is None:
        per_n = ", ".join(str(r["best_p"]) for r in rows)      # in the order of the x axis
        refs.append(("best", f"OpenMP best p (per p: {per_n})"))
    return refs


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def report(rows, refs):
    f = lambda v, spec: "NA" if v is None else format(v, spec)
    keys = [k for k, _ in refs] + (["best"] if "best" not in [k for k, _ in refs] else [])
    lines = ["CUDA speedup = T_omp(p) / T_cuda;  incl. overhead = T_omp(p) / (T_cuda + overhead_cuda)",
             "overhead: OpenMP = T(p) - T(1)/p (estimated), CUDA = measured", ""]
    head = f"{'N':>10} {'best p':>6} {'T_cuda':>8} {'ovh_cuda':>9}"
    for k in keys:
        head += f" {'T_omp(' + str(k) + ')':>12} {'S(' + str(k) + ')':>9} {'S_tot(' + str(k) + ')':>11}"
    lines.append(head)
    for r in rows:
        line = (f"{r['n']:>10,} {r['best_p']:>6} {f(r['cuda_time'], '.4f'):>8} "
                f"{f(r['cuda_overhead'], '.4f'):>9}")
        for k in keys:
            line += (f" {f(r[f'omp_time_{k}'], '.4f'):>12} {f(r[f'speedup_{k}'], '.2f'):>9}"
                     f" {f(r[f'speedup_total_{k}'], '.2f'):>11}")
        lines.append(line)
    lines += ["", "overhead [s] and share of the run time",
              f"{'N':>10} {'CUDA':>8} {'CUDA %':>7}" +
              "".join(f" {'OMP(' + str(k) + ')':>9} {'%':>6}" for k in keys if k != 1)]
    for r in rows:
        line = f"{r['n']:>10,} {f(r['cuda_overhead'], '.4f'):>8} {f(100 * r['cuda_overhead_frac'], '.1f'):>7}"
        for k in keys:
            if k != 1:
                line += (f" {f(r[f'omp_overhead_{k}'], '.4f'):>9}"
                         f" {f(100 * r[f'omp_overhead_frac_{k}'], '.1f'):>6}")
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot(rows, refs, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("plots skipped: pip install --user matplotlib")
        return
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10})
    x = [r["n"] for r in rows]

    def fig(name, ylabel, series, hline=None, log_y=False, y_top=None):
        f, ax = plt.subplots(figsize=(7.5, 4.6))
        f.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("number of points N", color=INK2, labelpad=6)
        ax.set_ylabel(ylabel, color=INK2, labelpad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n // 1000:,}k" for n in x])
        ax.tick_params(colors=INK2, length=3)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        if hline is not None:
            ax.axhline(hline[0], color=INK2, linestyle="--", linewidth=1.2, label=hline[1])
        for label, key, color in series:
            if key.endswith("_best"):           # dashed on top: the reference it may overlap stays visible
                ax.plot(x, [r[key] for r in rows], "--o", color=color, linewidth=1.6, markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1, zorder=4, label=label)
            else:
                ax.plot(x, [r[key] for r in rows], "-o", color=color, linewidth=2.2, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        if log_y:                               # speedups span 1..~200: log scale keeps all readable
            from matplotlib.ticker import FuncFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_ylim(bottom=0.8)
            ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.08))
        else:
            ax.set_ylim(0, y_top)
            ax.legend(frameon=False, loc="best")
        f.tight_layout()
        f.savefig(out_dir / f"{name}.png", dpi=150)
        plt.close(f)
        print(f"plot: {out_dir / (name + '.png')}")

    even = (1, "CUDA = OpenMP")
    fig("compare_speedup", "CUDA speedup (log scale)",
        [(label, f"speedup_{k}", REF_COLORS[k]) for k, label in refs], hline=even, log_y=True)
    fig("compare_speedup_total", "CUDA speedup incl. overhead (log scale)",
        [(label, f"speedup_total_{k}", REF_COLORS[k]) for k, label in refs], hline=even, log_y=True)
    omp_refs = [(k, label) for k, label in refs if k != 1]    # O(1) = 0 by definition
    fig("compare_overhead", "overhead [s]",
        [("CUDA (measured)", "cuda_overhead", CUDA_COLOR)] +
        [(label, f"omp_overhead_{k}", REF_COLORS[k]) for k, label in omp_refs])
    fig("compare_overhead_frac", "overhead / run time",
        [("CUDA (measured)", "cuda_overhead_frac", CUDA_COLOR)] +
        [(label, f"omp_overhead_frac_{k}", REF_COLORS[k]) for k, label in omp_refs], y_top=1.0)


def main():
    ap = argparse.ArgumentParser(description="OpenMP vs CUDA comparison",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("omp_dir", help="folder with the OpenMP metrics.txt")
    ap.add_argument("cuda_dir", help="folder with the CUDA metrics.txt")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results" / "compare"),
                    help="where to save compare_metrics.txt and the plots")
    args = ap.parse_args()

    omp = read_omp(Path(args.omp_dir) / "metrics.txt")
    cuda = read_cuda(Path(args.cuda_dir) / "metrics.txt")
    rows = compare(omp, cuda)
    refs = references(rows)

    only = sorted((set(omp) ^ set(cuda)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = (f"OpenMP vs CUDA: {args.omp_dir} vs {args.cuda_dir}\n"
            + (f"(N present in only one of the two, ignored: {', '.join(f'{n:,}' for n in only)})\n"
               if only else "")
            + "\n" + report(rows, refs) + "\n")
    (out_dir / "compare_metrics.txt").write_text(text)
    print(text + f"\nmetrics: {out_dir / 'compare_metrics.txt'}")
    plot(rows, refs, out_dir)


if __name__ == "__main__":
    main()
