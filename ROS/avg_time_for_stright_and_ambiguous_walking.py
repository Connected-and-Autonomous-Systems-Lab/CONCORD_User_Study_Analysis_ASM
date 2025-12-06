#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute average time users spend walking across:

1) Vertical path between cells (7, -133) and (7, -77)
2) Horizontal path between cells (35, -7) and (91, -7)

We look at all users in ./Output/<user>/Hard/ and all files named
    rosbag2_*_tf.csv      (but ignore *_cell_visits_over_time.csv)

Each TF CSV has columns:
    bag_timestamp_ns, header_sec, header_nanosec,
    parent_frame, child_frame, x, y, z, qx, qy, qz, qw

We only consider rows with:
    parent_frame == "human/odom"
    child_frame  == "human/base_footprint"
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
MAZE_INFO_CSV = Path("maze_info.csv")
OUTPUT_ROOT = Path("./Output")

# how close cell X needs to be (for robustness to tiny float errors)
X_TOL = 1e-6


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_hard_cells(maze_csv: Path):
    """Return arrays of (x, y) for Hard-maze cells."""
    df = pd.read_csv(maze_csv)
    hard = df[df["maze"] == "hard"].reset_index(drop=True)
    cell_x = hard["X"].to_numpy(dtype=float)
    cell_y = hard["Y"].to_numpy(dtype=float)
    return cell_x, cell_y


def build_path_indices(cell_x, cell_y):
    """
    Build ordered indices for:
    - vertical path from (7, -133) to (7, -77), ordered by Y
    - horizontal path from (35, -7) to (91, -7), ordered by X
    """
    # Vertical: X == 7, Y in [-133, -77]
    v_mask = (np.isclose(cell_x, 7.0, atol=X_TOL) &
              (cell_y >= -133.0 - 1e-6) &
              (cell_y <= -77.0 + 1e-6))
    v_indices = np.where(v_mask)[0]
    v_indices = v_indices[np.argsort(cell_y[v_indices])]  # bottom -> top

    # Horizontal: Y == -7, X in [35, 91]
    h_mask = (np.isclose(cell_y, -7.0, atol=1e-6) &
              (cell_x >= 35.0 - 1e-6) &
              (cell_x <= 91.0 + 1e-6))
    h_indices = np.where(h_mask)[0]
    h_indices = h_indices[np.argsort(cell_x[h_indices])]  # left -> right

    if len(v_indices) == 0:
        print("[WARN] No cells found for vertical path (7,-133)-(7,-77).")
    if len(h_indices) == 0:
        print("[WARN] No cells found for horizontal path (35,-7)-(91,-7).")

    return v_indices, h_indices


def map_positions_to_cells(xs, ys, cell_x, cell_y):
    """
    Map each (x, y) to index of nearest maze cell using brute-force distances.
    xs, ys: 1D numpy arrays
    Returns: 1D array of cell indices (0 .. n_cells-1)
    """
    points = np.column_stack([xs, ys])          # (N, 2)
    centers = np.column_stack([cell_x, cell_y])  # (M, 2)

    diff = points[:, None, :] - centers[None, :, :]   # (N, M, 2)
    dist2 = (diff ** 2).sum(axis=2)                   # (N, M)
    nearest_idx = dist2.argmin(axis=1)                # (N,)
    return nearest_idx


def extract_crossing_durations_ns(cell_idx, timestamps_ns, path_indices):
    """
    Generic function: given sequence of cell_idx (per frame) and timestamps, and
    an ordered list of path_indices (path_indices[0] is one endpoint,
    path_indices[-1] is the other), compute durations (in ns) of *full crossings*
    from one endpoint to the other while staying on the path.

    Counts both directions.
    """
    if len(path_indices) < 2:
        return []

    # map global cell index -> position along path (0..L-1)
    index_to_pos = {idx: pos for pos, idx in enumerate(path_indices)}
    first_pos = 0
    last_pos = len(path_indices) - 1

    durations = []
    state = "outside"
    start_t = None
    start_pos = None

    for t, c in zip(timestamps_ns, cell_idx):
        pos = index_to_pos.get(int(c), None)  # None if not on this path

        if state == "outside":
            # we start counting only if we enter at an endpoint
            if pos in (first_pos, last_pos):
                state = "in_path"
                start_t = t
                start_pos = pos

        else:  # state == "in_path"
            if pos is None:
                # left the path before reaching the opposite endpoint
                state = "outside"
                start_t = None
                start_pos = None
            else:
                # still on path
                if start_pos == first_pos and pos == last_pos:
                    durations.append(int(t - start_t))
                    state = "outside"
                    start_t = None
                    start_pos = None
                elif start_pos == last_pos and pos == first_pos:
                    durations.append(int(t - start_t))
                    state = "outside"
                    start_t = None
                    start_pos = None
                # else: still moving along path, but not yet at far end

    return durations


def process_tf_csv(
    csv_path: Path,
    cell_x,
    cell_y,
    vertical_indices,
    horizontal_indices,
):
    """
    For a single rosbag2_*_tf.csv, return lists:
        vertical_durs_ns, horizontal_durs_ns
    (each a list of crossing durations in nanoseconds)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not read {csv_path}: {e}")
        return [], []

    mask = (
        (df["parent_frame"] == "human/odom")
        & (df["child_frame"] == "human/base_footprint")
    )
    df = df.loc[mask, ["bag_timestamp_ns", "x", "y"]].copy()
    if df.empty:
        return [], []

    df = df.sort_values("bag_timestamp_ns")
    t_ns = df["bag_timestamp_ns"].to_numpy(dtype="int64")
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)

    # Map every position to nearest Hard-maze cell
    cell_idx = map_positions_to_cells(xs, ys, cell_x, cell_y)

    v_durs = extract_crossing_durations_ns(cell_idx, t_ns, vertical_indices)
    h_durs = extract_crossing_durations_ns(cell_idx, t_ns, horizontal_indices)
    return v_durs, h_durs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # Load Hard-maze cells and determine path cells
    cell_x, cell_y = load_hard_cells(MAZE_INFO_CSV)
    vertical_indices, horizontal_indices = build_path_indices(cell_x, cell_y)

    all_vertical_ns = []
    all_horizontal_ns = []

    if not OUTPUT_ROOT.exists():
        print(f"Output folder not found: {OUTPUT_ROOT.resolve()}")
        return

    # Walk over users and their Hard/rosbag2_*_tf.csv files
    for user_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not user_dir.is_dir():
            continue

        hard_dir = user_dir / "Hard"
        if not hard_dir.is_dir():
            continue

        for csv_path in hard_dir.glob("rosbag2_*_tf.csv"):
            # skip *_cell_visits_over_time.csv etc.
            if "cell_visits" in csv_path.name:
                continue

            v_durs, h_durs = process_tf_csv(
                csv_path, cell_x, cell_y, vertical_indices, horizontal_indices
            )
            all_vertical_ns.extend(v_durs)
            all_horizontal_ns.extend(h_durs)

    # -----------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------
    if all_vertical_ns:
        v_sec = np.array(all_vertical_ns, dtype="float64") / 1e9
        print(
            f"Vertical path (7,-133) <-> (7,-77): "
            f"{len(v_sec)} full crossings, "
            f"average time = {v_sec.mean():.3f} s"
        )
    else:
        print("No full crossings detected on vertical path (7,-133)-(7,-77).")

    if all_horizontal_ns:
        h_sec = np.array(all_horizontal_ns, dtype="float64") / 1e9
        print(
            f"Horizontal path (35,-7) <-> (91,-7): "
            f"{len(h_sec)} full crossings, "
            f"average time = {h_sec.mean():.3f} s"
        )
    else:
        print("No full crossings detected on horizontal path (35,-7)-(91,-7).")


if __name__ == "__main__":
    main()
