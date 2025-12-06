#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute average cell visits & coverage over time for Easy and Hard mazes,
and mark average human-detection times (per maze).

Assumes:
- ./Output/<user>/<Easy|Esay|easy|Hard|hard>/rosbag2_*_tf.csv
- For each TF file, a corresponding:
    rosbag2_*_tf_cell_visits_over_time.csv
  produced by your existing script, with columns:
    timestamp_s, total_cell_visits, unique_cell_visits

- human_detection_location_all_users.csv in the root dir with columns:
    user_name, maze, bag_timestamp_ns, point.x, point.y, point.z

Outputs (in the root dir):
- easy_maze_average_cell_visits_over_time.csv
- easy_maze_average_cell_visits_over_time.png
- hard_maze_average_cell_visits_over_time.csv
- hard_maze_average_cell_visits_over_time.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================

ROOT = Path("./Output")  # folder with user subfolders
DETECTION_CSV = Path("human_detection_location_all_users.csv")

# Total walkable cells (from your previous script)
WALKABLE_CELLS = {
    "easy": 31,
    "hard": 200,
}

# Time limits (seconds) for each maze.
# Adjust "easy" if its limit is different.
TIME_LIMITS = {
    "easy": 200.0,  # e.g., 10 minutes
    "hard": 600.0,  # 10 minutes
}

BIN_SIZE_S = 10.0  # average every 10 seconds


# ======================================================
# HELPERS
# ======================================================

def collect_runs(root: Path, maze_type: str):
    """
    Scan ./Output for TF runs of the given maze_type ("easy" or "hard").

    Returns:
        runs: list of dicts:
          {
            "username": <str>,
            "tf_csv": Path(...),
            "t0_ns": <int>,
            "t_end_ns": <int>,
          }
    """
    runs = []

    if not root.exists():
        print(f"[ERROR] ROOT directory not found: {root.resolve()}")
        return runs

    maze_type = maze_type.lower()
    if maze_type == "easy":
        level_names = ("Easy", "Esay", "easy")
    elif maze_type == "hard":
        level_names = ("Hard", "hard")
    else:
        print(f"[ERROR] Unknown maze_type: {maze_type}")
        return runs

    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        username = user_dir.name

        # Look for the appropriate level folders
        for level_name in level_names:
            level_dir = user_dir / level_name
            if not level_dir.is_dir():
                continue

            tf_files = sorted(level_dir.glob("rosbag2_*_tf.csv"))
            if not tf_files:
                continue

            for tf_csv in tf_files:
                # Only need min/max bag_timestamp_ns for mapping detections
                try:
                    df_meta = pd.read_csv(tf_csv, usecols=["bag_timestamp_ns"])
                except Exception as e:
                    print(f"[WARN] Failed to read {tf_csv}: {e}")
                    continue

                if df_meta.empty:
                    continue

                t0_ns = int(df_meta["bag_timestamp_ns"].min())
                t_end_ns = int(df_meta["bag_timestamp_ns"].max())

                runs.append(
                    {
                        "username": username,
                        "tf_csv": tf_csv,
                        "t0_ns": t0_ns,
                        "t_end_ns": t_end_ns,
                    }
                )

    print(f"[INFO] Found {len(runs)} {maze_type.capitalize()} runs.")
    return runs


def build_average_cell_curves(runs, maze_type: str):
    """
    For each run of a given maze_type, load its *_cell_visits_over_time.csv and
    build step-wise series for:
        - total_cell_visits
        - unique_cell_visits
        - coverage_pct (unique / WALKABLE_CELLS[maze_type] * 100)
    sampled at t = 0, 10, 20, ..., TIME_LIMITS[maze_type].

    Returns:
        grid_s:      (T,) array of time points [s]
        avg_total:   (T,) average total cell visits
        avg_unique:  (T,) average unique cell visits
        avg_cov_pct: (T,) average coverage percentage
    """
    maze_type = maze_type.lower()
    time_limit_s = TIME_LIMITS[maze_type]
    total_cells = WALKABLE_CELLS[maze_type]

    grid_s = np.arange(0.0, time_limit_s + BIN_SIZE_S, BIN_SIZE_S)

    series_total = []
    series_unique = []
    series_cov = []

    for r in runs:
        tf_csv = r["tf_csv"]
        summary_csv = tf_csv.with_name(tf_csv.stem + "_cell_visits_over_time.csv")

        if not summary_csv.exists():
            print(f"[WARN] Summary CSV not found for {tf_csv.name}, skipping this run.")
            continue

        try:
            df = pd.read_csv(summary_csv)
        except Exception as e:
            print(f"[WARN] Failed to read {summary_csv}: {e}")
            continue

        if df.empty:
            print(f"[WARN] Empty summary CSV: {summary_csv}, skipping.")
            continue

        required_cols = {"timestamp_s", "total_cell_visits", "unique_cell_visits"}
        if not required_cols.issubset(df.columns):
            print(f"[WARN] Missing required columns in {summary_csv}, skipping.")
            continue

        times = df["timestamp_s"].to_numpy(dtype=float)
        total = df["total_cell_visits"].to_numpy(dtype=float)
        unique = df["unique_cell_visits"].to_numpy(dtype=float)

        # Sort (just in case)
        order = np.argsort(times)
        times = times[order]
        total = total[order]
        unique = unique[order]

        total_grid = np.zeros_like(grid_s)
        unique_grid = np.zeros_like(grid_s)

        for i, t in enumerate(grid_s):
            # index of last event time <= t
            idx = np.searchsorted(times, t, side="right") - 1
            if idx >= 0:
                total_grid[i] = total[idx]
                unique_grid[i] = unique[idx]
            else:
                total_grid[i] = 0.0
                unique_grid[i] = 0.0

        cov_grid = (unique_grid / float(total_cells)) * 100.0

        series_total.append(total_grid)
        series_unique.append(unique_grid)
        series_cov.append(cov_grid)

    if not series_total:
        raise RuntimeError(
            f"No valid {maze_type.capitalize()} runs with *_cell_visits_over_time.csv found."
        )

    total_arr = np.vstack(series_total)   # shape: (N_runs, T)
    unique_arr = np.vstack(series_unique)
    cov_arr = np.vstack(series_cov)

    avg_total = total_arr.mean(axis=0)
    avg_unique = unique_arr.mean(axis=0)
    avg_cov = cov_arr.mean(axis=0)

    print(f"[INFO] Averaged over {total_arr.shape[0]} {maze_type.capitalize()} runs.")
    return grid_s, avg_total, avg_unique, avg_cov


def compute_average_detection_times(runs, maze_type: str):
    """
    Map human detections in the given maze_type ("easy"/"hard")
    to individual runs and compute average detection times
    for the 1st..8th detected humans.

    Returns:
        avg_times: dict {k: mean_time_s} for k in [1..8]
        std_times: dict {k: std_time_s} for k in [1..8]
        count_by_k: dict {k: N_runs_with_kth_detection}
    """
    maze_type = maze_type.lower()

    if not DETECTION_CSV.exists():
        print(f"[WARN] Detection CSV not found: {DETECTION_CSV.resolve()}")
        return {}, {}, {}

    try:
        det = pd.read_csv(DETECTION_CSV)
    except Exception as e:
        print(f"[WARN] Failed to read detection CSV: {e}")
        return {}, {}, {}

    required_cols = {"user_name", "maze", "bag_timestamp_ns"}
    if not required_cols.issubset(det.columns):
        print(f"[WARN] Detection CSV missing columns {required_cols}, skipping.")
        return {}, {}, {}

    det["maze"] = det["maze"].astype(str).str.strip().str.lower()
    det["user_name"] = det["user_name"].astype(str).str.strip()

    det_mz = det[det["maze"] == maze_type].copy()
    if det_mz.empty:
        print(f"[WARN] No {maze_type.capitalize()}-maze detections found.")
        return {}, {}, {}

    # Index runs by username
    runs_by_user = {}
    for r in runs:
        runs_by_user.setdefault(r["username"], []).append(r)

    detections_by_run = {}  # key: tf_csv Path, value: list of t_rel_s
    unmatched = 0

    time_limit_s = TIME_LIMITS[maze_type]

    for _, row in det_mz.iterrows():
        uname = row["user_name"]
        ts_ns = int(row["bag_timestamp_ns"])

        user_runs = runs_by_user.get(uname, [])
        match = None
        for r in user_runs:
            if r["t0_ns"] <= ts_ns <= r["t_end_ns"]:
                match = r
                break

        if match is None:
            unmatched += 1
            continue

        run_key = match["tf_csv"]
        t_rel_s = (ts_ns - match["t0_ns"]) / 1e9
        if t_rel_s < 0:
            continue

        detections_by_run.setdefault(run_key, []).append(t_rel_s)

    if unmatched > 0:
        print(
            f"[WARN] {unmatched} {maze_type.capitalize()} detection rows "
            f"could not be mapped to any run."
        )

    # For each run, sort detection times and keep up to 8
    times_by_k = {k: [] for k in range(1, 9)}

    for run_key, times_list in detections_by_run.items():
        # Keep reasonable times (e.g., within 1.5 * time limit)
        times_sorted = sorted(
            t for t in times_list if 0.0 <= t <= time_limit_s * 1.5
        )
        for idx, t in enumerate(times_sorted):
            k = idx + 1
            if k > 8:
                break
            times_by_k[k].append(t)

    avg_times = {}
    std_times = {}
    count_by_k = {}

    for k in range(1, 9):
        vals = np.array(times_by_k[k], dtype=float)
        if len(vals) == 0:
            continue
        avg_times[k] = float(vals.mean())
        std_times[k] = float(vals.std(ddof=0))
        count_by_k[k] = int(len(vals))

    return avg_times, std_times, count_by_k


def plot_average_with_detections(
    grid_s,
    avg_total,
    avg_unique,
    avg_cov_pct,
    avg_det_times,
    maze_type: str,
    out_path: Path,
):
    """
    Plot average cell visits & coverage over time for a given maze_type,
    plus vertical red dotted lines at average detection times.
    """
    maze_label = maze_type.capitalize()

    fig, ax_left = plt.subplots(figsize=(10, 6))

    # ============================
    #  Left Y-axis (Cell counts)
    # ============================

    ax_left.plot(
        grid_s,
        avg_total,
        linewidth=4.0,              # thicker line
        label="Avg total cell visits",
    )
    ax_left.plot(
        grid_s,
        avg_unique,
        linewidth=4.0,              # thicker line
        label="Avg unique cell visits",
    )

    ax_left.set_xlabel("Time (s)", fontsize=18)
    ax_left.set_ylabel("Number of cells visited", fontsize=18)
    ax_left.grid(True, linestyle="--", alpha=0.4)
    ax_left.tick_params(axis='both', labelsize=18)

    # ============================
    #  Right Y-axis (Coverage %)
    # ============================

    ax_right = ax_left.twinx()
    coverage_color = "#99d8c9"

    ax_right.plot(
        grid_s,
        avg_cov_pct,
        linewidth=4.0,              # thicker coverage line
        alpha=0.6,
        color=coverage_color,
        label="Avg unique coverage (%)",
    )

    ax_right.fill_between(
        grid_s,
        avg_cov_pct,
        0.0,
        color=coverage_color,
        alpha=0.25,
    )

    ax_right.set_ylim(0.0, 100.0)
    ax_right.set_ylabel("Unique cell coverage (%)", fontsize=18)
    ax_right.tick_params(axis='y', labelsize=18)

    # ============================
    #  Detection Time Lines
    # ============================

    from matplotlib.lines import Line2D

    if avg_det_times:
        for k, t in sorted(avg_det_times.items()):
            ax_left.axvline(
                x=t,
                color="red",
                linestyle=":",
                linewidth=2.0,       # thicker vertical line
                alpha=0.9,
            )

        # Legend entry for detection lines
        det_line = Line2D(
            [0], [0],
            color="red",
            linestyle=":",
            linewidth=3.0,
            label="Avg human detection times",
        )

        # Merge all legend items
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()

        ax_left.legend(
            lines_left + [det_line] + lines_right,
            labels_left + ["Avg human detection times"] + labels_right,
            loc="upper left",
            fontsize=20,            #  Increase legend font size here
        )
    else:
        # No detection times → merge typical legends
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()

        ax_left.legend(
            lines_left + lines_right,
            labels_left + labels_right,
            loc="upper left",
            fontsize=20,            #  Increase legend font size
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {maze_label} maze plot: {out_path}")

# ======================================================
# MAIN
# ======================================================

def main():
    for maze_type in ("easy", "hard"):
        print(f"\n========== {maze_type.upper()} MAZE ==========")

        # 1) Find all runs for this maze
        runs = collect_runs(ROOT, maze_type)
        if not runs:
            print(f"[WARN] No runs found for {maze_type} maze. Skipping.")
            continue

        # 2) Build average cell-visit curves
        try:
            grid_s, avg_total, avg_unique, avg_cov_pct = build_average_cell_curves(
                runs, maze_type
            )
        except RuntimeError as e:
            print(f"[WARN] {e}")
            continue

        # 3) Compute average human-detection times
        avg_det_times, std_det_times, count_by_k = compute_average_detection_times(
            runs, maze_type
        )

        if avg_det_times:
            print(f"\n[INFO] Average {maze_type.capitalize()}-maze human detection times:")
            for k in sorted(avg_det_times.keys()):
                mean_t = avg_det_times[k]
                std_t = std_det_times.get(k, float("nan"))
                n = count_by_k.get(k, 0)
                print(
                    f"  {k}th human: mean = {mean_t:.2f} s, "
                    f"std = {std_t:.2f} s, N = {n}"
                )
        else:
            print(f"\n[WARN] No detection statistics available for {maze_type} maze.")

        # 4) Save averaged series as CSV
        out_csv = Path(f"{maze_type}_maze_average_cell_visits_over_time.csv")
        df_out = pd.DataFrame(
            {
                "timestamp_s": grid_s,
                "avg_total_cell_visits": avg_total,
                "avg_unique_cell_visits": avg_unique,
                "avg_unique_coverage_pct": avg_cov_pct,
            }
        )
        df_out.to_csv(out_csv, index=False)
        print(f"[OK] Saved CSV: {out_csv}")

        # 5) Plot with vertical detection lines
        out_png = Path(f"{maze_type}_maze_average_cell_visits_over_time.png")
        plot_average_with_detections(
            grid_s,
            avg_total,
            avg_unique,
            avg_cov_pct,
            avg_det_times,
            maze_type,
            out_png,
        )


if __name__ == "__main__":
    main()
