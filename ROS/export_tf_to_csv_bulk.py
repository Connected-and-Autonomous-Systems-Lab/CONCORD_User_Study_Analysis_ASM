#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# export_tf_to_csv_bulk.py

import csv
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# ======================================================
# CONFIG
# ======================================================
DATASET_ROOT = Path("../../Collaboration_user_study")
OUTPUT_ROOT = Path("./Output")

TF_TOPIC_NAME = "/tf"  # topic to read
# IMPORTANT: set this to match your ROS 2 distro
ROS_DISTRO_STORE = Stores.ROS2_HUMBLE


# ======================================================
# SINGLE-BAG EXPORT USING ROSBAGS
# ======================================================
def export_tf_for_bag(bag_dir: Path, out_csv: Path):
    """
    Export TF from a single rosbag2_* directory to out_csv using rosbags.
    """
    print(f"  -> Processing {bag_dir}")

    if not bag_dir.is_dir():
        print(f"    [ERROR] Bag directory not found: {bag_dir}")
        return

    try:
        # Get typestore and its deserialize function
        typestore = get_typestore(ROS_DISTRO_STORE)
        deserialize_cdr = typestore.deserialize_cdr

        with Reader(bag_dir) as reader:
            # Find /tf connections
            tf_connections = [
                conn for conn in reader.connections
                if conn.topic == TF_TOPIC_NAME
            ]

            if not tf_connections:
                print(f"    [ERROR] Topic '{TF_TOPIC_NAME}' not found in {bag_dir}")
                return

            # Ensure output directory exists
            out_csv.parent.mkdir(parents=True, exist_ok=True)

            with open(out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "bag_timestamp_ns",      # Reader timestamp (ns since epoch)
                    "header_sec",            # header stamp seconds
                    "header_nanosec",        # header stamp nanoseconds
                    "parent_frame",
                    "child_frame",
                    "x", "y", "z",
                    "qx", "qy", "qz", "qw",
                ])

                msg_count = 0

                # Iterate over all /tf messages
                for connection, timestamp, rawdata in reader.messages(connections=tf_connections):
                    # Decode TFMessage using typestore
                    msg = deserialize_cdr(rawdata, connection.msgtype)

                    # msg.transforms is a list of geometry_msgs/TransformStamped
                    for transform in msg.transforms:
                        header = transform.header
                        t = transform.transform.translation
                        r = transform.transform.rotation

                        parent = header.frame_id
                        child = transform.child_frame_id

                        writer.writerow([
                            timestamp,
                            header.stamp.sec,
                            header.stamp.nanosec,
                            parent,
                            child,
                            t.x, t.y, t.z,
                            r.x, r.y, r.z, r.w,
                        ])

                    msg_count += 1

        print(f"    [OK] Exported TF from topic '{TF_TOPIC_NAME}' to: {out_csv}")

    except Exception as e:
        print(f"    [ERROR] Failed to process {bag_dir}: {e}")


# ======================================================
# WALK ENTIRE DATASET AND MIRROR STRUCTURE
# ======================================================
def main():
    if not DATASET_ROOT.exists():
        print(f"Dataset root not found: {DATASET_ROOT.resolve()}")
        return

    for user_dir in sorted(DATASET_ROOT.iterdir()):
        if not user_dir.is_dir():
            continue

        user_name = user_dir.name
        print(f"\n=== User: {user_name} ===")

        for level in ("Easy", "Hard"):
            level_dir = user_dir / level
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

            for bag_dir in bag_dirs:
                # Output path: ./Output/<User>/<Level>/<bag_dir_name>_tf.csv
                out_dir = OUTPUT_ROOT / user_name / level
                out_csv = out_dir / f"{bag_dir.name}_tf.csv"
                export_tf_for_bag(bag_dir, out_csv)


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    main()
