#!/usr/bin/env python3
"""
Plot the final result of omp-k-means or cuda-k-means.

Both programs write the same output format:

    # Data points: N
    # Dimensions: D
    # Clusters: K
    # Centroids:
    #
    #   0 : c00 c01 ... c0(D-1)
    #   ...
    #
    x0 x1 ... x(D-1) cluster_id
    ...

Usage:
    ./plot_result.py OUTPUT_FILE [-o IMAGE.png] [--dims I J] [--no-pca] [--show]

If the data have 2 dimensions they are plotted directly. With D > 2 the
points and centroids are projected onto the first two principal
components (the default), onto two chosen dimensions (--dims), or onto
dimensions 0 and 1 (--no-pca).

Requires numpy and matplotlib:  pip install --user numpy matplotlib
"""

import argparse
import re
import sys

import numpy as np
import matplotlib

CENTROID_RE = re.compile(r"^#\s*\d+\s*:(.*)$")


def read_output(path):
    """Return (points, labels, centroids) read from a k-means output file."""
    centroids, rows = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                m = CENTROID_RE.match(line)
                if m:
                    centroids.append([float(v) for v in m.group(1).split()])
                continue
            rows.append([float(v) for v in line.split()])
    if not rows:
        sys.exit(f"FATAL: no data points found in {path}")
    data = np.array(rows)
    return data[:, :-1], data[:, -1].astype(int), np.array(centroids)


def project(points, centroids, dims, use_pca):
    """Map points and centroids to 2-D; return them plus the axis labels."""
    d = points.shape[1]
    if d == 1:
        zeros = np.zeros((points.shape[0], 1))
        return (np.hstack([points, zeros]),
                np.hstack([centroids, np.zeros((centroids.shape[0], 1))]),
                ("dim 0", ""))
    if d == 2 and dims is None:
        return points, centroids, ("dim 0", "dim 1")
    if dims is not None:
        i, j = dims
        if not (0 <= i < d and 0 <= j < d):
            sys.exit(f"FATAL: --dims must be in 0..{d - 1}")
        return points[:, [i, j]], centroids[:, [i, j]], (f"dim {i}", f"dim {j}")
    if use_pca:
        mean = points.mean(axis=0)
        # principal axes from the SVD of the centered data
        _, s, vt = np.linalg.svd(points - mean, full_matrices=False)
        axes = vt[:2].T
        var = s**2 / np.sum(s**2)
        return ((points - mean) @ axes, (centroids - mean) @ axes,
                (f"PC1 ({var[0]:.0%} var.)", f"PC2 ({var[1]:.0%} var.)"))
    return points[:, :2], centroids[:, :2], ("dim 0", "dim 1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("output_file", help="file written by omp-k-means or cuda-k-means")
    ap.add_argument("-o", "--image", help="image to write (default: OUTPUT_FILE.png)")
    ap.add_argument("--dims", type=int, nargs=2, metavar=("I", "J"),
                    help="plot dimensions I and J instead of the PCA projection")
    ap.add_argument("--no-pca", dest="pca", action="store_false",
                    help="with D > 2, plot dims 0 and 1 instead of PCA")
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    ap.add_argument("--max-points", type=int, default=50000,
                    help="plot at most this many points (random sample); default 50000")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")           # no display needed
    import matplotlib.pyplot as plt

    points, labels, centroids = read_output(args.output_file)
    n, d = points.shape
    k = len(centroids)

    if n > args.max_points:             # keep big files readable and fast
        idx = np.random.default_rng(0).choice(n, args.max_points, replace=False)
        points, labels = points[idx], labels[idx]

    p2, c2, (xlab, ylab) = project(points, centroids, args.dims, args.pca)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(p2[:, 0], p2[:, 1], c=labels, cmap="tab20" if k <= 20 else "viridis",
               s=4, alpha=0.6, linewidths=0)
    ax.scatter(c2[:, 0], c2[:, 1], marker="X", s=180, c="black",
               edgecolors="white", linewidths=1.5, label="centroids")
    ax.set_title(f"{args.output_file}  (N={n}, D={d}, K={k})")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend(loc="best")
    fig.tight_layout()

    image = args.image or args.output_file + ".png"
    fig.savefig(image, dpi=120)
    print(f"Saved {image}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
