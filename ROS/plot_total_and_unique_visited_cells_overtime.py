#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For each user and each Easy/Hard level in ./Output:

- Read rosbag2_*_tf.csv
- Keep only rows for the human avatar:
      parent_frame == "human/odom"
      child_frame  == "human/base_footprint"
- Map (x, y) positions to maze cells using maze_info.csv (centres of 7x7 cells).
- Compute, over time:
      * total_cell_visits: number of cell transitions (compressed, not every frame)
      * unique_cell_visits: number of distinct cells ever visited so far
- Save 2 plots and 1 CSV next to each rosbag2_*_tf.csv:
      *_total_cells_over_time.png
      *_unique_cells_over_time.png
      *_cell_visits_over_time.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================

ROOT = Path("./Output")          # folder with user subfolders
MAZE_INFO_CSV = Path("maze_info.csv")  # in the root directory

CELL_SIZE = 7.0
CELL_HALF_SIZE = CELL_SIZE / 2.0  # radius threshold for assigning a point to a cell

PARENT_FRAME = "human/odom"
CHILD_FRAME = "human/base_footprint"

WALKABLE_CELLS = {
    "easy": 31,
    "hard": 200,
}


# ======================================================
# HELPERS
# ======================================================

def load_maze_cells(maze_info_path: Path):
    """
    Load maze_info.csv and return dict:
        cells["easy"] -> np.array of shape (N_easy, 2) with columns [X, Y]
        cells["hard"] -> np.array of shape (N_hard, 2)
    """
    df = pd.read_csv(maze_info_path)
    df["maze"] = df["maze"].str.strip().str.lower()

    cells = {}
    for maze_type in ("easy", "hard"):
        sub = df[df["maze"] == maze_type]
        if sub.empty:
            print(f"[WARN] No cells found for maze='{maze_type}' in {maze_info_path}")
            cells[maze_type] = np.zeros((0, 2), dtype=float)
        else:
            cells[maze_type] = sub[["X", "Y"]].to_numpy(dtype=float)
    return cells


def assign_cells_to_positions(xy: np.ndarray,
                              cell_centers: np.ndarray,
                              max_radius: float) -> np.ndarray:
    """
    For each (x, y) in xy, assign index of nearest cell centre.
    If nearest centre is farther than max_radius, return -1 for that position.

    Args:
        xy:           (N, 2) array of positions
        cell_centers: (M, 2) array of cell centres
        max_radius:   maximum distance from cell centre to accept it

    Returns:
        cell_ids: (N,) int array; -1 means "no cell"
    """
    if xy.size == 0 or cell_centers.size == 0:
        return np.full(len(xy), -1, dtype=int)

    # Broadcasting to compute distances
    diff = xy[:, None, :] - cell_centers[None, :, :]   # (N, M, 2)
    sqdist = (diff ** 2).sum(axis=2)                  # (N, M)
    nearest_idx = sqdist.argmin(axis=1)               # (N,)
    nearest_dist = np.sqrt(sqdist[np.arange(len(xy)), nearest_idx])

    cell_ids = nearest_idx.copy()
    cell_ids[nearest_dist > max_radius] = -1
    return cell_ids


def compute_cell_visit_series(times_s: np.ndarray,
                              cell_ids: np.ndarray):
    """
    Given times (seconds) and cell_ids per frame, compute a compressed series
    of cell visits over time.

    - Only counts transitions when cell_ids change (ignores consecutive duplicates).
    - Ignores rows where cell_id == -1.

    Returns:
        visit_times_s:      list of times (seconds) at each counted visit
        total_counts:       list of total_cell_visits at each time
        unique_counts:      list of unique_cell_visits at each time
    """
    visit_times = []
    total_counts = []
    unique_counts = []

    visited_cells = set()
    total = 0
    prev_cell = None

    for t, c in zip(times_s, cell_ids):
        if c == -1:
            continue  # not in any cell

        if prev_cell is None or c != prev_cell:
            # New visit (enter a cell or switch cell)
            total += 1
            visited_cells.add(c)

            visit_times.append(t)
            total_counts.append(total)
            unique_counts.append(len(visited_cells))

            prev_cell = c

    return visit_times, total_counts, unique_counts



def plot_combined_cell_visits(summary_df, title_base, out_path: Path, total_walkable_cells: int):
    """
    Plot:
      - Total cell visits (left Y)
      - Unique cell visits (left Y)
      - % unique cell coverage (right Y, light line + filled area)
    """
    times = summary_df["timestamp_s"].to_numpy()
    total_visits = summary_df["total_cell_visits"].to_numpy()
    unique_visits = summary_df["unique_cell_visits"].to_numpy()

    # Percentage of unique cell coverage over time
    coverage_pct = (unique_visits / float(total_walkable_cells)) * 100.0

    fig, ax_left = plt.subplots()

    # Left Y-axis: counts
    ax_left.plot(times, total_visits, linewidth=1.5, label="Total cell visits")
    ax_left.plot(times, unique_visits, linewidth=1.5, label="Unique cell visits")
    ax_left.set_xlabel("Time (s)")
    ax_left.set_ylabel("Number of cells visited")
    ax_left.grid(True, linestyle="--", alpha=0.4)

    # Right Y-axis: percentage coverage
    ax_right = ax_left.twinx()
    coverage_color = "#99d8c9"  # very light green/teal
    ax_right.plot(
        times,
        coverage_pct,
        linewidth=2.0,
        alpha=0.5,
        color=coverage_color,
        label="Unique cell coverage (%)",
    )
    ax_right.fill_between(
        times,
        coverage_pct,
        0,
        color=coverage_color,
        alpha=0.2,  # make the area very light
    )
    ax_right.set_ylim(0, 100)
    ax_right.set_ylabel("Unique cell coverage (%)")

    # Title
    ax_left.set_title(f"{title_base} - Cell Visits & Coverage Over Time")

    # Combined legend from both axes
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Saved combined plot: {out_path}")



def plot_series(x, y, title, xlabel, ylabel, out_path: Path):
    """Simple helper to make and save a line plot."""
    plt.figure()
    plt.plot(x, y, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


# ======================================================
# MAIN
# ======================================================

def main():
    if not ROOT.exists():
        print(f"[ERROR] ROOT directory not found: {ROOT.resolve()}")
        return

    if not MAZE_INFO_CSV.exists():
        print(f"[ERROR] maze_info.csv not found: {MAZE_INFO_CSV.resolve()}")
        return

    maze_cells = load_maze_cells(MAZE_INFO_CSV)

    # Loop over users
    for user_dir in sorted(ROOT.iterdir()):
        if not user_dir.is_dir():
            continue

        username = user_dir.name
        print(f"\n=== User: {username} ===")

        # Handle both "Easy"/"Esay" just in case
        level_dirs = []
        for name in ("Easy", "Esay", "easy", "Hard", "hard"):
            d = user_dir / name
            if d.is_dir():
                level_dirs.append(d)

        if not level_dirs:
            print(f"[WARN] No level folders found in {user_dir}")
            continue

        for level_dir in level_dirs:
            level_name = level_dir.name
            level_lower = level_name.lower()

            if "easy" in level_lower:
                maze_type = "easy"
            elif "hard" in level_lower:
                maze_type = "hard"
            else:
                print(f"[WARN] Unknown level folder name: {level_dir}")
                continue

            print(f"  - Level: {level_name} (maze='{maze_type}')")

            cell_centers = maze_cells.get(maze_type, np.zeros((0, 2)))
            if cell_centers.size == 0:
                print(f"    [WARN] No cell centres for maze '{maze_type}', skipping.")
                continue

            tf_files = sorted(level_dir.glob("rosbag2_*_tf.csv"))
            if not tf_files:
                print(f"    [WARN] No rosbag2_*_tf.csv in {level_dir}")
                continue

            for tf_csv in tf_files:
                print(f"    Processing {tf_csv.name} ...")

                # --- Load CSV ---
                df = pd.read_csv(tf_csv)

                # --- Filter frames for the human avatar ---
                df["parent_frame"] = df["parent_frame"].astype(str).str.strip()
                df["child_frame"] = df["child_frame"].astype(str).str.strip()

                mask = (df["parent_frame"] == PARENT_FRAME) & \
                       (df["child_frame"] == CHILD_FRAME)
                df = df[mask].copy()

                if df.empty:
                    print(f"      [WARN] No rows with parent={PARENT_FRAME} and child={CHILD_FRAME}.")
                    continue

                # Sort by ROS bag timestamp
                df = df.sort_values("bag_timestamp_ns").reset_index(drop=True)

                # Relative time in seconds
                t0 = df["bag_timestamp_ns"].iloc[0]
                df["time_s"] = (df["bag_timestamp_ns"] - t0) / 1e9

                # Position array
                xy = df[["x", "y"]].to_numpy(float)

                # Assign each position to a cell
                cell_ids = assign_cells_to_positions(xy, cell_centers, CELL_HALF_SIZE)

                # Compute visit series
                visit_times_s, total_counts, unique_counts = compute_cell_visit_series(
                    df["time_s"].to_numpy(), cell_ids
                )

                if not visit_times_s:
                    print("      [WARN] No valid cell visits detected.")
                    continue

                # Build summary DataFrame
                summary_df = pd.DataFrame({
                    "timestamp_s": visit_times_s,
                    "total_cell_visits": total_counts,
                    "unique_cell_visits": unique_counts,
                })

                # --- Save CSV ---
                out_csv = tf_csv.with_name(tf_csv.stem + "_cell_visits_over_time.csv")
                summary_df.to_csv(out_csv, index=False)
                print(f"      [OK] Saved summary CSV: {out_csv}")

                # --- Plots ---
                title_base = f"{username} - {level_name}"

                # Decide number of walkable cells based on maze type
                total_walkable = WALKABLE_CELLS.get(maze_type)
                if total_walkable is None or total_walkable <= 0:
                    print(f"      [WARN] Unknown total walkable cells for maze='{maze_type}', skipping plot.")
                    continue

                out_png_combined = tf_csv.with_name(tf_csv.stem + "_cells_over_time_combined.png")
                plot_combined_cell_visits(summary_df, title_base, out_png_combined, total_walkable)        


if __name__ == "__main__":
    main()
