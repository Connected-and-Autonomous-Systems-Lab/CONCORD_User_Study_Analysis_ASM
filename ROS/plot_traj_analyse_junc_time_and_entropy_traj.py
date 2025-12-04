#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traverse ./Output, plot human avatar trajectories, and analyze junction behavior.

Outputs:
- Per-run trajectory plots saved next to each *_tf.csv
- decision_times_summary.csv   (per user / level / run / junction / visit)
- explorativeness_summary.csv  (per user / level / junction, with entropy)
"""

import math
from pathlib import Path
from collections import deque  # === NEW ===

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================
OUTPUT_ROOT = Path("./Output")

# Only consider CSVs created by your TF export:
CSV_PATTERN = "*_tf.csv"

# Frames to keep (human avatar)
PARENT_FRAME = "human/odom"
CHILD_FRAME = "human/base_footprint"

# Visibility / junction thresholds
VISIBLE_RADIUS = 5.0       # user can "see" junction from this distance (units)
JUNCTION_BOX_HALF = 3.0    # |x - jx| <= 3 and |y - jy| <= 3 => close enough to count as being at junction

# === Maze geometry info ===
MAZE_INFO = {
    "Easy": {
        "start": (20.3, -8.2),
        # One 3-way junction at (21, -35)
        "junctions": [
            # (junction_id, x, y)
            ("EJ1", 21.0, -35.0),
        ],
    },
    "Hard": {
        "start": (91.5, -20.3),
        # 13 junction coordinates provided
        "junctions": [
            ("HJ1", 132.0, -20.0),
            ("HJ2", 120.0, -77.0),
            ("HJ3", 105.0, -75.0),
            ("HJ4",  20.0, -130.0),
            ("HJ5",  63.0, -118.0),
            ("HJ6",  90.0, -119.0),
            ("HJ7",  90.0, -132.0),
            ("HJ8",  20.0, -76.0),
            ("HJ9", 63.0, -62.0),
            ("HJ10", 78.0, -35.0),
            ("HJ11", 77.0, -7.5),
            ("HJ12", 35.0, -35.0),
            ("HJ13", 35.0, -7.0),
        ],
    },
}

# ======================================================
# TREE DEFINITIONS (from your textual description)
# ======================================================
# === NEW: abstract tree structure of the mazes ===
MAZE_TREES = {
    "Easy": {
        "root": "start",
        "children": {
            "start": ["EJ1", "DE_E1"],   # deadend branch not tracked as junction
            "EJ1": ["DE_E2", "DE_E3"],
        },
    },
    "Hard": {
        "root": "start",
        "children": {
            "start": ["HJ1", "HJ11"],        # from start, two branches
            "HJ1": ["HJ2", "DE_H1"],
            "HJ2": ["HJ3", "DE_H2"],
            "HJ3": ["DE_H3", "DE_H4"],

            "HJ4": ["HJ5", "DE_H5"],
            "HJ5": ["HJ6", "DE_H6"],
            "HJ6": ["HJ7", "DE_H7"],
            "HJ7": ["DE_H8", "DE_H9"],

            "HJ8": ["HJ4", "DE_H10"],
            "HJ9": ["HJ8", "DE_H11"],
            "HJ10": ["HJ9", "DE_H12"],

            "HJ11": ["HJ10", "HJ12"],
            "HJ12": ["HJ13", "DE_H13"],
            "HJ13": ["DE_H14", "DE_H15"],
        },
    },
}


# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def classify_branch(dx: float, dy: float) -> str:
    """
    Classify the direction of leaving a junction into one of {N, S, E, W}.
    dx, dy are from junction center to a point AFTER leaving the junction area.
    """
    if abs(dx) >= abs(dy):
        # More horizontal than vertical
        return "E" if dx > 0 else "W"
    else:
        # More vertical than horizontal
        return "N" if dy > 0 else "S"


def get_user_level_from_path(csv_path: Path):
    """
    Assuming structure: Output/<User>/<Level>/<something>_tf.csv
    Return (user_name, level_name) or (None, None) if cannot parse.
    """
    try:
        level = csv_path.parent.name
        user = csv_path.parent.parent.name
        return user, level
    except Exception:
        return None, None


def compute_time_column(df: pd.DataFrame) -> pd.Series:
    """
    Compute relative time (seconds) from bag_timestamp_ns.
    """
    ts0 = df["bag_timestamp_ns"].iloc[0]
    return (df["bag_timestamp_ns"] - ts0) / 1e9


# ======================================================
# TREE HELPERS (NEW)
# ======================================================
# === NEW: generic helpers to work with the MAZE_TREES ===

def build_parent_map(children: dict, root: str) -> dict:
    parent = {}

    def dfs(node: str):
        for ch in children.get(node, []):
            parent[ch] = node
            dfs(ch)

    dfs(root)
    return parent


def bfs_traversal(children: dict, root: str) -> list:
    """Return BFS order starting at root (including root)."""
    order = []
    q = deque([root])
    visited = {root}
    while q:
        node = q.popleft()
        order.append(node)
        for ch in children.get(node, []):
            if ch not in visited:
                visited.add(ch)
                q.append(ch)
    return order


def dfs_preorder(children: dict, root: str) -> list:
    """Return DFS pre-order starting at root (including root)."""
    order = []

    def _dfs(node: str):
        order.append(node)
        for ch in children.get(node, []):
            _dfs(ch)

    _dfs(root)
    return order


def compute_order_similarity(ref_order: list, actual_order: list) -> float:
    """
    Compare two permutations (or partial permutations) of the same node set
    using pairwise ordering agreement (0..1).
    """
    idx_ref = {n: i for i, n in enumerate(ref_order)}
    idx_act = {n: i for i, n in enumerate(actual_order)}
    common = list(set(idx_ref) & set(idx_act))
    if len(common) <= 1:
        return 1.0  # trivial / no comparison possible

    agree = 0
    total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a = common[i]
            b = common[j]
            total += 1
            if (idx_ref[a] - idx_ref[b]) * (idx_act[a] - idx_act[b]) > 0:
                agree += 1
    return agree / total if total > 0 else 1.0


def build_tree_layout(children: dict, root: str) -> dict:
    """
    Simple tree layout for plotting:
    Returns dict: node -> (x, y) coordinates.
    """
    positions = {}
    x_counter = 0

    def dfs(node: str, depth: int):
        nonlocal x_counter
        chs = children.get(node, [])
        if not chs:
            # Leaf
            positions[node] = (x_counter, -depth)
            x_counter += 1
        else:
            child_xs = []
            for ch in chs:
                dfs(ch, depth + 1)
                child_xs.append(positions[ch][0])
            # place node at mean of children
            positions[node] = (sum(child_xs) / len(child_xs), -depth)

    dfs(root, 0)
    return positions


def plot_tree_with_path(level: str,
                        out_path: Path,
                        tree_def: dict,
                        maze_info_level: dict,
                        visited_junction_order: list):
    """
    Plot the maze tree and highlight the path implied by visited_junction_order.
    """
    children = tree_def["children"]
    root = tree_def["root"]

    positions = build_tree_layout(children, root)
    parent_map = build_parent_map(children, root)

    # junction nodes = ones that appear in MAZE_INFO for this level
    junction_ids = {j_id for (j_id, _x, _y) in maze_info_level["junctions"]}

    # set of nodes to highlight along root->junction paths
    highlight_edges = set()

    for j in visited_junction_order:
        if j not in junction_ids:
            continue
        node = j
        while node in parent_map:
            p = parent_map[node]
            edge = tuple(sorted((p, node)))
            highlight_edges.add(edge)
            node = p

    plt.figure(figsize=(6, 5))

    # Draw all edges (light)
    for node, chs in children.items():
        for ch in chs:
            x = [positions[node][0], positions[ch][0]]
            y = [positions[node][1], positions[ch][1]]
            plt.plot(x, y, linewidth=1, alpha=0.4, color="black")

    # Highlight path edges (thicker)
    for (a, b) in highlight_edges:
        x = [positions[a][0], positions[b][0]]
        y = [positions[a][1], positions[b][1]]
        plt.plot(x, y, linewidth=3, alpha=0.9, color="black")

    # Draw nodes
    all_nodes = set(positions.keys())
    visited_set = set(visited_junction_order) | {"start"}

    for node in all_nodes:
        x, y = positions[node]
        if node == "start":
            plt.scatter([x], [y], s=80, marker="*", zorder=3)
        elif node in junction_ids:
            # junctions
            if node in visited_set:
                plt.scatter([x], [y], s=50, marker="o", zorder=3)
            else:
                plt.scatter([x], [y], s=30, marker="o", facecolors="none", edgecolors="black", zorder=2)
        else:
            # deadends
            plt.scatter([x], [y], s=20, marker="x", alpha=0.6, zorder=1)

        plt.text(x + 0.05, y + 0.05, node, fontsize=7)

    plt.title(f"{level} maze tree (highlighted visited path)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ======================================================
# JUNCTION VISIT DETECTION
# ======================================================
def find_junction_visits(df: pd.DataFrame,
                         time_s: pd.Series,
                         jx: float,
                         jy: float,
                         user: str,
                         level: str,
                         run_name: str,
                         junction_id: str):
    """
    Given trajectory df with columns x, y, and time_s, find all visits to the
    junction at (jx, jy).

    A *visit* is a contiguous segment where:
      - distance to junction <= VISIBLE_RADIUS, and
      - during that segment, the avatar comes within the inner junction box
        |x-jx| <= JUNCTION_BOX_HALF and |y-jy| <= JUNCTION_BOX_HALF.

    For each visit we also infer the exit branch direction.

    Returns list of dicts (one per visit).
    """
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    t = time_s.to_numpy()

    dx = x - jx
    dy = y - jy
    dist = np.sqrt(dx**2 + dy**2)

    visible = dist <= VISIBLE_RADIUS
    in_box = (np.abs(dx) <= JUNCTION_BOX_HALF) & (np.abs(dy) <= JUNCTION_BOX_HALF)

    visits = []
    n = len(df)
    i = 0
    visit_idx = 0

    while i < n:
        if not visible[i]:
            i += 1
            continue

        # Start of visible segment
        start_idx = i
        while i < n and visible[i]:
            i += 1
        end_idx = i - 1  # last visible index

        # Check if this segment actually passes through the junction box
        if not in_box[start_idx : end_idx + 1].any():
            continue

        visit_idx += 1

        # Decision time: duration of this visible segment
        t_start = t[start_idx]
        t_end = t[end_idx]
        decision_time = t_end - t_start

        # Closest approach to junction within this visit
        visit_dist = dist[start_idx : end_idx + 1]
        min_dist = float(visit_dist.min())
        min_idx_rel = int(visit_dist.argmin())
        min_idx = start_idx + min_idx_rel

        # Determine exit direction:
        # Take position at end of segment (the moment they finally leave visibility)
        dx_leave = dx[end_idx]
        dy_leave = dy[end_idx]
        branch_dir = classify_branch(dx_leave, dy_leave)

        visits.append({
            "user": user,
            "level": level,
            "run": run_name,
            "junction_id": junction_id,
            "junction_x": jx,
            "junction_y": jy,
            "visit_index": visit_idx,
            "t_start_s": float(t_start),
            "t_end_s": float(t_end),
            "decision_time_s": float(decision_time),
            "min_distance": float(min_dist),
            "branch": branch_dir,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
        })

    return visits


# ======================================================
# PLOTTING
# ======================================================
def plot_trajectory_xy(df: pd.DataFrame,
                       user: str,
                       level: str,
                       run_name: str,
                       out_path: Path,
                       maze_info_level: dict | None):
    """
    Plot 2D trajectory (x-y) and save to out_path.
    If maze_info_level is provided, mark start + junctions.
    """
    plt.figure()
    plt.plot(df["x"], df["y"], linewidth=1)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"{user} - {level} - {run_name} (x-y)")

    # Equal aspect ratio helps see maze layout
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)

    if maze_info_level is not None:
        # Mark start
        sx, sy = maze_info_level["start"]
        plt.scatter([sx], [sy], marker="*", s=80)
        plt.text(sx, sy, " start", fontsize=8)

        # Mark junctions
        for j_id, jx, jy in maze_info_level["junctions"]:
            plt.scatter([jx], [jy], marker="o", s=40)
            plt.text(jx, jy, f" {j_id}", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ======================================================
# MAIN
# ======================================================
def main():
    all_visits = []  # for decision_times_summary
    # We'll accumulate per-visit data; explorativeness will be derived from it.

    if not OUTPUT_ROOT.exists():
        print(f"[ERROR] Output root not found: {OUTPUT_ROOT.resolve()}")
        return

    # Traverse all *_tf.csv files
    for csv_path in sorted(OUTPUT_ROOT.rglob(CSV_PATTERN)):
        user, level = get_user_level_from_path(csv_path)
        if user is None or level is None:
            print(f"[WARN] Could not infer user/level from path: {csv_path}")
            continue

        print(f"\n=== User: {user} | Level: {level} ===")
        print(f"Processing CSV: {csv_path}")

        # Only analyze levels for which we have maze info
        maze_info_level = MAZE_INFO.get(level)
        if maze_info_level is None:
            print(f"  [INFO] No maze info for level '{level}', skipping analysis.")
        # We'll still plot trajectory even if no maze info.

        # Load CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [ERROR] Failed to read {csv_path}: {e}")
            continue

        # Filter for human avatar frames
        mask = (
            (df["parent_frame"] == PARENT_FRAME) &
            (df["child_frame"] == CHILD_FRAME)
        )
        hdf = df[mask].copy()
        if hdf.empty:
            print("  [WARN] No human/odom -> human/base_footprint rows; skipping.")
            continue

        hdf = hdf.sort_values("bag_timestamp_ns").reset_index(drop=True)
        hdf["time_s"] = compute_time_column(hdf)

        # Run name from filename (drop suffix "_tf" if present)
        run_name = csv_path.stem
        if run_name.endswith("_tf"):
            run_name = run_name[:-3]

        # ---- Plot trajectory and save next to CSV ----
        traj_png = csv_path.with_name(csv_path.stem + "_traj_xy.png")
        plot_trajectory_xy(hdf, user, level, run_name, traj_png, maze_info_level)
        print(f"  [OK] Saved trajectory plot to {traj_png}")

        # ---- Junction analysis (only if we know maze layout for this level) ----
        # === NEW: per-run visits list for tree/path analysis ===
        run_visits = []

        if maze_info_level is None:
            continue

        for j_id, jx, jy in maze_info_level["junctions"]:
            visits = find_junction_visits(
                hdf,
                hdf["time_s"],
                jx,
                jy,
                user,
                level,
                run_name,
                j_id,
            )
            if visits:
                print(f"  [INFO] Found {len(visits)} visit(s) to junction {j_id} at ({jx}, {jy})")
                all_visits.extend(visits)
                run_visits.extend(visits)   # === NEW ===
            else:
                print(f"  [INFO] No visits detected for junction {j_id} at ({jx}, {jy})")

        # === NEW: tree-based analysis & plotting, per run ===
        tree_def = MAZE_TREES.get(level)
        if tree_def is not None:
            tree_png = csv_path.with_name(csv_path.stem + "_tree.png")

            # get order of first visits to each junction for this run
            if run_visits:
                rv_df = pd.DataFrame(run_visits)
                first_visits = (
                    rv_df.groupby("junction_id")["t_start_s"]
                    .min()
                    .reset_index()
                    .sort_values("t_start_s")
                )
                visited_junction_order = first_visits["junction_id"].tolist()
            else:
                visited_junction_order = []

            # Plot tree with highlighted path
            plot_tree_with_path(
                level=level,
                out_path=tree_png,
                tree_def=tree_def,
                maze_info_level=maze_info_level,
                visited_junction_order=visited_junction_order,
            )
            print(f"  [OK] Saved tree plot to {tree_png}")

            # Compare with BFS / DFS traversals
            children = tree_def["children"]
            root = tree_def["root"]
            bfs_order_full = bfs_traversal(children, root)
            dfs_order_full = dfs_preorder(children, root)

            junction_ids = {j_id for (j_id, _x, _y) in maze_info_level["junctions"]}
            visited_set = set(visited_junction_order)

            # restrict BFS/DFS to junctions visited in this run
            bfs_junction_order = [n for n in bfs_order_full if n in junction_ids and n in visited_set]
            dfs_junction_order = [n for n in dfs_order_full if n in junction_ids and n in visited_set]
            actual_order = [n for n in visited_junction_order if n in junction_ids]

            if actual_order:
                sim_bfs = compute_order_similarity(bfs_junction_order, actual_order)
                sim_dfs = compute_order_similarity(dfs_junction_order, actual_order)

                if sim_bfs > sim_dfs:
                    pattern = "BFS-like"
                elif sim_dfs > sim_bfs:
                    pattern = "DFS-like"
                else:
                    pattern = "mixed/ambiguous"

                print(f"  [TREE] Run={run_name}  pattern={pattern} "
                      f"(sim_BFS={sim_bfs:.3f}, sim_DFS={sim_dfs:.3f})")
            else:
                print("  [TREE] No junction visits in this run for traversal comparison.")

    # ==================================================
    # SAVE SUMMARY CSVs
    # ==================================================
    if not all_visits:
        print("\n[INFO] No junction visits detected in any run; no summary CSVs created.")
        return

    visits_df = pd.DataFrame(all_visits)

    # 1) Decision times
    decision_csv = Path("decision_times_summary.csv")
    visits_df.to_csv(decision_csv, index=False)
    print(f"\n[OK] Wrote decision time summary to {decision_csv.resolve()}")

    # 2) Explorativess summary (per user, level, junction)
    #    We treat each visit's 'branch' as a choice. The more diverse the branches,
    #    the more exploratory the user is at that junction.
    groups = visits_df.groupby(["user", "level", "junction_id", "junction_x", "junction_y"])

    exp_rows = []
    for (user, level, j_id, jx, jy), g in groups:
        total_visits = len(g)
        branches = g["branch"].tolist()

        counts = {b: branches.count(b) for b in ["N", "S", "E", "W"]}
        unique_branches = sum(c > 0 for c in counts.values())

        # Entropy over branches
        probs = [c / total_visits for c in counts.values() if c > 0]
        entropy_bits = -sum(p * math.log2(p) for p in probs) if probs else 0.0

        exp_rows.append({
            "user": user,
            "level": level,
            "junction_id": j_id,
            "junction_x": jx,
            "junction_y": jy,
            "total_visits": total_visits,
            "unique_branches": unique_branches,
            "count_N": counts["N"],
            "count_S": counts["S"],
            "count_E": counts["E"],
            "count_W": counts["W"],
            "branch_entropy_bits": entropy_bits,
        })

    explor_df = pd.DataFrame(exp_rows)
    explor_csv = Path("explorativeness_summary.csv")
    explor_df.to_csv(explor_csv, index=False)
    print(f"[OK] Wrote explorativeness summary to {explor_csv.resolve()}")


if __name__ == "__main__":
    main()
