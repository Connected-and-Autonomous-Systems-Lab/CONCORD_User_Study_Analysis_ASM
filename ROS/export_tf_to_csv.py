#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sqlite3
import yaml
import struct
from pathlib import Path


# ======================================================
# CONFIG
# ======================================================
BAG_DIR = r"../../Collaboration_user_study/Mugunthan/Easy/rosbag2_2025_11_26-18_13_26"
OUTPUT_CSV = "tf_data.csv"
TF_TOPIC_TYPE = "tf2_msgs/msg/TFMessage"


# ======================================================
# MINIMAL CDR READER (same as in your main script)
# ======================================================
class CDRReader:

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def skip_encapsulation_header(self):
        if len(self.data) >= 4:
            self.offset = 4

    def align(self, n: int):
        if n <= 1:
            return
        o = self.offset
        self.offset = ((o + (n - 1)) // n) * n

    def read_uint32(self):
        self.align(4)
        v = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return v

    def read_int32(self):
        self.align(4)
        v = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return v

    def read_float64(self):
        self.align(8)
        v = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return v

    def read_string(self):
        length = self.read_uint32()
        if length == 0:
            return ""
        start = self.offset
        end = start + length
        raw = self.data[start:end]
        self.offset = end
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        return raw.decode("utf-8", errors="ignore")

    # ----------------------------
    # Sub-structures
    # ----------------------------
    def read_time(self):
        sec = self.read_int32()
        nsec = self.read_uint32()
        return {"sec": sec, "nanosec": nsec}

    def read_header(self):
        return {"stamp": self.read_time(), "frame_id": self.read_string()}

    def read_vector3(self):
        x = self.read_float64()
        y = self.read_float64()
        z = self.read_float64()
        return {"x": x, "y": y, "z": z}

    def read_quaternion(self):
        x = self.read_float64()
        y = self.read_float64()
        z = self.read_float64()
        w = self.read_float64()
        return {"x": x, "y": y, "z": z, "w": w}

    # ----------------------------
    # TFMessage decoder (1st transform only)
    # ----------------------------
    def decode_tf_message(self):
        num = self.read_uint32()
        if num == 0:
            return None

        self.align(4)

        header = self.read_header()
        child_frame_id = self.read_string()
        translation = self.read_vector3()
        rotation = self.read_quaternion()

        return {
            "num_transforms": num,
            "header": header,
            "child_frame_id": child_frame_id,
            "translation": translation,
            "rotation": rotation,
        }


# ======================================================
# METADATA LOAD
# ======================================================
def load_metadata(bag_dir: Path):
    with open(bag_dir / "metadata.yaml", "r") as f:
        meta = yaml.safe_load(f)
    return meta["rosbag2_bagfile_information"]


def find_db3_file(bag_dir: Path, metadata: dict) -> Path:
    rel = metadata["relative_file_paths"][0]
    return bag_dir / rel


def get_tf_topic_id(metadata):
    for i, t in enumerate(metadata["topics_with_message_count"]):
        if t["topic_metadata"]["type"] == TF_TOPIC_TYPE:
            return i
    return None


# ======================================================
# MAIN EXPORT FUNCTION
# ======================================================
def export_tf_to_csv():
    bag_dir = Path(BAG_DIR)
    metadata = load_metadata(bag_dir)
    db_path = find_db3_file(bag_dir, metadata)

    # Find TF topic ID from DB3 topics table
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM topics;")
    topics = cur.fetchall()

    tf_topic_id = None
    for t in topics:
        if t["type"] == TF_TOPIC_TYPE:
            tf_topic_id = t["id"]
            break

    if tf_topic_id is None:
        print("ERROR: No TFMessage topic found.")
        return

    # Read TF messages
    cur.execute(
        """
        SELECT timestamp, data FROM messages
        WHERE topic_id=?
        ORDER BY timestamp ASC;
        """,
        (tf_topic_id,),
    )

    rows = cur.fetchall()
    conn.close()

    # Open CSV writer
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "parent_frame",
            "child_frame",
            "x", "y", "z",
            "qx", "qy", "qz", "qw",
        ])

        for row in rows:
            ts = row["timestamp"]
            data = row["data"]

            reader = CDRReader(data)
            reader.skip_encapsulation_header()
            decoded = reader.decode_tf_message()

            if decoded is None:
                continue

            parent = decoded["header"]["frame_id"]
            child = decoded["child_frame_id"]

            t = decoded["translation"]
            r = decoded["rotation"]

            writer.writerow([
                ts,
                parent,
                child,
                t["x"], t["y"], t["z"],
                r["x"], r["y"], r["z"], r["w"],
            ])

    print(f"TF data exported successfully to: {OUTPUT_CSV}")


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    export_tf_to_csv()
