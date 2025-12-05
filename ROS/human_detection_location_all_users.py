#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk over ../../Collaboration_user_study, read /human/detection_location
from all rosbag2_* folders, and save everything into a SINGLE CSV:

    human_detection_location_all_users.csv

Columns:
- user_name
- maze   ("Easy" or "Hard")
- bag_timestamp_ns   (rosbags Reader timestamp, ns)
- point.x
- point.y
- point.z

At the end, prints:
- Average time (seconds) users spent identifying humans in Easy maze
  (per user time span between first and last detection, averaged)
- Average number of humans identified by users in Hard maze
  (per user count of detection messages, averaged)
"""

import csv
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping, Sequence

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# ======================================================
# CONFIG
# ======================================================
DATASET_ROOT = Path("../../Collaboration_user_study")
# CSV will be created in the same directory as this script
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "human_detection_location_all_users.csv"

DETECTION_TOPIC_NAME = "/human/detection_location"
ROS_DISTRO_STORE = Stores.ROS2_HUMBLE


# ======================================================
# HELPER: CONVERT MESSAGE TO PRIMITIVE PYTHON STRUCTURES
# ======================================================
def to_primitive(obj):
    """
    Convert rosbags message (namedtuple/dataclass/etc.) into
    plain Python types: dicts, lists, ints, floats, strings, etc.
    """
    # Basic scalar types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Mapping-like
    if isinstance(obj, Mapping):
        return {k: to_primitive(v) for k, v in obj.items()}

    # Messages that support _asdict() (many ROS msg types in rosbags do)
    if hasattr(obj, "_asdict"):
        return {k: to_primitive(v) for k, v in obj._asdict().items()}

    # Sequences (lists, tuples, arrays) but not bytes/str
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray, str)):
        return [to_primitive(v) for v in obj]

    # Objects with __dict__
    if hasattr(obj, "__dict__"):
        return {k: to_primitive(v) for k, v in vars(obj).items()}

    # Fallback to string
    return str(obj)


# ======================================================
# HELPER: FLATTEN NESTED DICTS/LISTS INTO A SINGLE-LEVEL DICT
# ======================================================
def flatten(prefix, value, out_dict):
    """
    Recursively flatten `value` into `out_dict` using keys like:
        header.stamp.sec
        point.x
        point.y
        etc.
    """
    if isinstance(value, Mapping):
        for k, v in value.items():
            new_key = f"{prefix}.{k}" if prefix else k
            flatten(new_key, v, out_dict)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for i, v in enumerate(value):
            new_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            flatten(new_key, v, out_dict)
    else:
        # Primitive or fallback
        out_dict[prefix] = value


# ======================================================
# EXPORT FUNCTION FOR A SINGLE BAG
# ======================================================
def collect_detection_for_bag(bag_dir: Path, user_name: str, maze: str,
                              deserialize_cdr, all_rows):
    """
    Read /human/detection_location from a single rosbag2_* directory
    and append rows with the required columns into all_rows.
    """
    print(f"  -> Processing bag: {bag_dir}")

    if not bag_dir.is_dir():
        print(f"    [ERROR] Bag directory not found: {bag_dir}")
        return

    try:
        with Reader(bag_dir) as reader:
            # Find connections for the detection topic
            det_connections = [
                conn for conn in reader.connections
                if conn.topic == DETECTION_TOPIC_NAME
            ]

            if not det_connections:
                print(f"    [INFO] Topic '{DETECTION_TOPIC_NAME}' not found in {bag_dir}, skipping.")
                return

            msg_count = 0

            # Iterate over all messages on that topic
            for connection, timestamp, rawdata in reader.messages(connections=det_connections):
                msg = deserialize_cdr(rawdata, connection.msgtype)

                # Convert to primitive Python types, then flatten
                primitive = to_primitive(msg)
                flat = {}

                if isinstance(primitive, Mapping):
                    for key, val in primitive.items():
                        flatten(key, val, flat)
                else:
                    # If top-level isn't a dict, treat as a single field "value"
                    flatten("value", primitive, flat)

                # Build row with ONLY required columns
                row = {
                    "user_name": user_name,
                    "maze": maze,
                    "bag_timestamp_ns": int(timestamp),
                    "point.x": flat.get("point.x"),
                    "point.y": flat.get("point.y"),
                    "point.z": flat.get("point.z"),
                }

                all_rows.append(row)
                msg_count += 1

        print(f"    [OK] Collected {msg_count} messages from {DETECTION_TOPIC_NAME} in {bag_dir}")

    except Exception as e:
        print(f"    [ERROR] Failed to process {bag_dir}: {e}")


# ======================================================
# STATS: EASY TIME + HARD COUNT
# ======================================================
def compute_and_print_stats(all_rows):
    """
    Computes:
    - Average time (seconds) users spent identifying humans in Easy maze:
      For each user: (max_ts - min_ts) over all Easy rows -> duration,
      then average these durations across users.

    - Average number of humans identified by the users in Hard maze:
      For each user: number of rows in Hard -> count,
      then average these counts across users.
    """
    # --- Easy maze: per-user time span ---
    easy_min_ts = {}
    easy_max_ts = {}

    # --- Hard maze: per-user detection count ---
    hard_counts = defaultdict(int)

    for row in all_rows:
        user = row["user_name"]
        maze = row["maze"]
        ts_ns = row["bag_timestamp_ns"]

        if ts_ns is None:
            continue

        if maze == "Easy":
            if user not in easy_min_ts or ts_ns < easy_min_ts[user]:
                easy_min_ts[user] = ts_ns
            if user not in easy_max_ts or ts_ns > easy_max_ts[user]:
                easy_max_ts[user] = ts_ns

        elif maze == "Hard":
            hard_counts[user] += 1

    # Compute Easy durations
    easy_durations_sec = []
    for user in easy_min_ts:
        span_ns = easy_max_ts[user] - easy_min_ts[user]
        if span_ns > 0:
            easy_durations_sec.append(span_ns / 1e9)

    avg_easy_time_sec = sum(easy_durations_sec) / len(easy_durations_sec) if easy_durations_sec else 0.0

    # Compute Hard average detections per user
    hard_counts_list = list(hard_counts.values())
    avg_hard_detections = sum(hard_counts_list) / len(hard_counts_list) if hard_counts_list else 0.0

    print("\n================= SUMMARY STATS =================")
    if easy_durations_sec:
        print(f"Average time users spent identifying humans in Easy maze: {avg_easy_time_sec:.2f} seconds")
    else:
        print("No Easy maze detection data found for computing time span.")

    if hard_counts_list:
        print(f"Average number of humans identified by users in Hard maze: {avg_hard_detections:.2f}")
    else:
        print("No Hard maze detection data found for computing counts.")
    print("=================================================\n")


# ======================================================
# WALK ENTIRE DATASET AND COLLECT INTO ONE CSV
# ======================================================
def main():
    if not DATASET_ROOT.exists():
        print(f"Dataset root not found: {DATASET_ROOT.resolve()}")
        return

    print(f"Dataset root: {DATASET_ROOT.resolve()}")
    print(f"Output CSV will be: {OUTPUT_CSV}")

    # Prepare typestore once
    typestore = get_typestore(ROS_DISTRO_STORE)
    deserialize_cdr = typestore.deserialize_cdr

    all_rows = []  # list[dict]

    # Walk users
    for user_dir in sorted(DATASET_ROOT.iterdir()):
        if not user_dir.is_dir():
            continue

        user_name = user_dir.name
        print(f"\n=== User: {user_name} ===")

        # Easy / Hard levels
        for maze in ("Easy", "Hard"):
            level_dir = user_dir / maze
            if not level_dir.exists() or not level_dir.is_dir():
                print(f"  [INFO] {level_dir} not found, skipping.")
                continue

            # All rosbag2_* dirs under this level
            bag_dirs = [
                d for d in level_dir.iterdir()
                if d.is_dir() and d.name.startswith("rosbag2_")
            ]

            if not bag_dirs:
                print(f"  [INFO] No rosbag2_* dirs in {level_dir}, skipping.")
                continue

            for bag_dir in sorted(bag_dirs):
                collect_detection_for_bag(
                    bag_dir, user_name, maze,
                    deserialize_cdr, all_rows
                )

    # If no data, nothing to write
    if not all_rows:
        print("\n[WARN] No /human/detection_location messages found in the dataset.")
        return

    # Write single CSV with ONLY the requested columns
    print(f"\nWriting {len(all_rows)} rows to {OUTPUT_CSV} ...")

    columns = ["user_name", "maze", "bag_timestamp_ns", "point.x", "point.y", "point.z"]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({col: row.get(col, None) for col in columns})

    print(f"[DONE] CSV saved at: {OUTPUT_CSV}")

    # Compute and print stats
    compute_and_print_stats(all_rows)


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    main()
