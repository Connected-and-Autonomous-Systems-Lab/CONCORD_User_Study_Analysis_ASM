#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute average time spent walking between junctions HJ4 and HJ8 (both directions)
for users: Alireza, Mohammad, Raj, Subal.

Folder structure:
    ./Output/
        Alireza/
            Easy/
            Hard/
                rosbag2_<timestamp>_tf.csv
                rosbag2_*_tf_cell_visits_over_time.csv   (ignored)
        ...

Each rosbag2_*_tf.csv has columns:
    bag_timestamp_ns, header_sec, header_nanosec,
    parent_frame, child_frame, x, y, z, qx, qy, qz, qw

We only consider rows with:
    parent_frame == "human/odom" AND child_frame == "human/base_footprint"

We detect when the human walks from HJ4 to HJ8 or from HJ8 to HJ4.
Time between the two junctions is measured from the moment they leave
one junction (enter the corridor) until they enter the other one.

All times are printed in seconds.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ======================================================
# CONFIG
# ======================================================

OUTPUT_ROOT = Path("./Output")

TARGET_USERS = {"Alireza", "Mohammad", "Raj", "Subal"}

# Junction positions (Hard maze)
HJ4_X, HJ4_Y = 20.0, -130.0
HJ8_X, HJ8_Y = 20.0, -76.0

# How close you need to be to count as "at" a junction (box around junction)
JUNCTION_BOX_HALF = 3.0  # |x - jx| <= 3 and |y - jy| <= 3


def classify_zone(x: float, y: float) -> str | None:
    """
    Return 'HJ4', 'HJ8', or None depending on (x, y) position.
    """
    if abs(x - HJ4_X) <= JUNCTION_BOX_HALF and abs(y - HJ4_Y) <= JUNCTION_BOX_HALF:
        return "HJ4"
    if abs(x - HJ8_X) <= JUNCTION_BOX_HALF and abs(y - HJ8_Y) <= JUNCTION_BOX_HALF:
        return "HJ8"
    return None


def extract_travel_durations_between_HJ4_HJ8(csv_path: Path) -> list[int]:
    """
    For one rosbag2_*_tf.csv file, return a list of travel times (in nanoseconds)
    between HJ4 and HJ8 (both directions).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not read {csv_path}: {e}")
        return []

    # Keep only human avatar frames
    mask = (
        (df["parent_frame"] == "human/odom")
        & (df["child_frame"] == "human/base_footprint")
    )
    df = df.loc[mask, ["bag_timestamp_ns", "x", "y"]].copy()
    if df.empty:
        return []

    # Sort by time just in case
    df = df.sort_values("bag_timestamp_ns")

    # Make sure timestamps are integers
    df["bag_timestamp_ns"] = df["bag_timestamp_ns"].astype("int64")

    # Classify each row as being at HJ4, HJ8, or neither
    df["zone"] = [
        classify_zone(x, y) for x, y in zip(df["x"].to_numpy(), df["y"].to_numpy())
    ]

    durations_ns: list[int] = []

    prev_zone = None
    start_t = None          # time when leaving a junction into the corridor
    source_zone = None      # 'HJ4' or 'HJ8'

    for t, zone in zip(df["bag_timestamp_ns"], df["zone"]):

        # Detect changes in zone
        if zone != prev_zone:
            # Case 1: leaving a junction into corridor -> start measuring
            if prev_zone in ("HJ4", "HJ8") and zone is None:
                start_t = t
                source_zone = prev_zone

            # Case 2: coming from corridor into a junction -> maybe end segment
            elif prev_zone is None and zone in ("HJ4", "HJ8") and start_t is not None:
                # we finished a corridor segment; check if it ended at the *other* junction
                if source_zone is not None and zone != source_zone:
                    durations_ns.append(int(t - start_t))
                # reset for next possible segment
                start_t = None
                source_zone = None

        prev_zone = zone

    return durations_ns


def main():
    all_durations_ns: list[int] = []
    per_user: dict[str, list[int]] = {u: [] for u in TARGET_USERS}

    for user_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not user_dir.is_dir():
            continue

        user = user_dir.name
        if user not in TARGET_USERS:
            continue

        hard_dir = user_dir / "Hard"
        if not hard_dir.is_dir():
            continue

        print(f"Processing user '{user}' ...")

        for csv_path in hard_dir.glob("rosbag2_*_tf.csv"):
            # Ignore any *_tf_cell_visits_over_time.csv etc.
            if "cell_visits" in csv_path.name:
                continue

            durs = extract_travel_durations_between_HJ4_HJ8(csv_path)
            if durs:
                per_user[user].extend(durs)
                all_durations_ns.extend(durs)

    # ==========================
    # Print results
    # ==========================
    if not all_durations_ns:
        print("No HJ4 <-> HJ8 travel segments found.")
        return

    all_durations_sec = np.array(all_durations_ns, dtype="float64") / 1e9
    print(
        f"\nOverall: {len(all_durations_sec)} segments "
        f"between HJ4 and HJ8 (both directions)."
    )
    print(f"Average time: {all_durations_sec.mean():.3f} s")
    print(f"STD time:     {all_durations_sec.std(ddof=1):.3f} s\n")

    for user, durs_ns in per_user.items():
        if not durs_ns:
            print(f"{user}: no HJ4<->HJ8 segments detected.")
            continue
        durs_sec = np.array(durs_ns, dtype='float64') / 1e9
        print(
            f"{user}: {len(durs_sec)} segments, "
            f"avg = {durs_sec.mean():.3f} s, std = {durs_sec.std(ddof=1):.3f} s"
        )


if __name__ == "__main__":
    main()
