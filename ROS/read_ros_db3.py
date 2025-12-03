#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import yaml
import struct
from pathlib import Path
from datetime import datetime, timezone

# ===================================================
# HARD-CODED PARAMETERS
# ===================================================
BAG_DIR = r"../../Collaboration_user_study/Mugunthan/Easy/rosbag2_2025_11_26-18_13_26"
MESSAGE_LIMIT = 5
SHOW_ONLY_TOPIC_ID = None  # e.g. 3 to show only topic_id=3, or None for all


# ===================================================
# HELPER: TIME CONVERSION
# ===================================================
def ns_to_dt(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


# ===================================================
# CDR READER (very small subset, enough for our msgs)
# ===================================================
class CDRReader:
    """
    Minimal CDR (Common Data Representation) reader for ROS2 messages.
    Assumes:
      - Little-endian (typical on x86/Windows/Linux)
      - CDR encapsulation header is first 4 bytes (skipped)
      - Basic alignment rules for primitives
    This is NOT a full CDR implementation, just enough for common ROS2 types.
    """

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def skip_encapsulation_header(self):
        # ROS2 Fast-CDR adds a 4-byte encapsulation header at the front.
        if len(self.data) >= 4:
            self.offset = 4

    def align(self, n: int):
        """Align offset to n-byte boundary."""
        if n <= 1:
            return
        o = self.offset
        self.offset = ((o + (n - 1)) // n) * n

    def read_uint32(self):
        self.align(4)
        val = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return val

    def read_int32(self):
        self.align(4)
        val = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return val

    def read_float32(self):
        self.align(4)
        val = struct.unpack_from("<f", self.data, self.offset)[0]
        self.offset += 4
        return val

    def read_float64(self):
        self.align(8)
        val = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return val

    def read_int8_array(self, max_len=None):
        """
        Read sequence<int8>.
        If max_len is given, only return up to that many; still skip the rest.
        """
        length = self.read_uint32()
        start = self.offset
        end = start + length
        raw = self.data[start:end]
        self.offset = end

        if max_len is None or length <= max_len:
            return list(raw)
        else:
            return list(raw[:max_len])

    def read_float32_array(self, max_len=None):
        """
        Read sequence<float32>.
        If max_len is given, only return up to that many; still skip the rest.
        """
        length = self.read_uint32()
        vals = []
        for i in range(length):
            v = self.read_float32()
            if max_len is None or i < max_len:
                vals.append(v)
        return vals

    def read_string(self):
        """
        CDR string is encoded as:
          - uint32 length (including null terminator)
          - bytes[ length ]
        """
        length = self.read_uint32()
        if length == 0:
            return ""
        start = self.offset
        end = start + length
        raw = self.data[start:end]
        self.offset = end

        # Drop the trailing null terminator if present
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    # ---------- Common ROS2 substructures ----------

    def read_time(self):
        # builtin_interfaces/Time: int32 sec, uint32 nanosec
        sec = self.read_int32()
        nanosec = self.read_uint32()
        return {"sec": sec, "nanosec": nanosec}

    def read_header(self):
        # std_msgs/Header:
        #   builtin_interfaces/Time stamp
        #   string frame_id
        stamp = self.read_time()
        frame_id = self.read_string()
        return {"stamp": stamp, "frame_id": frame_id}

    def read_point(self):
        # geometry_msgs/Point: double x,y,z
        x = self.read_float64()
        y = self.read_float64()
        z = self.read_float64()
        return {"x": x, "y": y, "z": z}

    def read_quaternion(self):
        # geometry_msgs/Quaternion: double x,y,z,w
        x = self.read_float64()
        y = self.read_float64()
        z = self.read_float64()
        w = self.read_float64()
        return {"x": x, "y": y, "z": z, "w": w}

    def read_pose(self):
        # geometry_msgs/Pose:
        #   Point position
        #   Quaternion orientation
        position = self.read_point()
        orientation = self.read_quaternion()
        return {"position": position, "orientation": orientation}

    def read_vector3(self):
        x = self.read_float64()
        y = self.read_float64()
        z = self.read_float64()
        return {"x": x, "y": y, "z": z}

    # ---------- Message type-specific decoders ----------

    def decode_point_stamped(self):
        hdr = self.read_header()
        pt = self.read_point()
        return {
            "header": hdr,
            "point": pt,
        }

    def decode_pose_stamped(self):
        hdr = self.read_header()
        pose = self.read_pose()
        return {
            "header": hdr,
            "pose": pose,
        }

    def decode_pose_with_covariance_stamped(self):
        hdr = self.read_header()
        pose = self.read_pose()
        # covariance: double[36]
        cov = [self.read_float64() for _ in range(36)]
        return {"header": hdr, "pose": pose, "covariance": cov}

    def decode_laserscan(self, max_ranges_preview=10):
        hdr = self.read_header()
        angle_min = self.read_float32()
        angle_max = self.read_float32()
        angle_increment = self.read_float32()
        time_increment = self.read_float32()
        scan_time = self.read_float32()
        range_min = self.read_float32()
        range_max = self.read_float32()
        ranges = self.read_float32_array(max_len=max_ranges_preview)
        intensities = self.read_float32_array(max_len=0)  # we don't really need them

        return {
            "header": hdr,
            "angle_min": angle_min,
            "angle_max": angle_max,
            "angle_increment": angle_increment,
            "time_increment": time_increment,
            "scan_time": scan_time,
            "range_min": range_min,
            "range_max": range_max,
            "ranges_preview": ranges,
        }

    def decode_string_msg(self):
        # std_msgs/msg/String: string data
        data = self.read_string()
        return {"data": data}

    def decode_map_metadata(self):
        # nav_msgs/MapMetaData:
        #   builtin_interfaces/Time map_load_time
        #   float32 resolution
        #   uint32 width
        #   uint32 height
        #   geometry_msgs/Pose origin
        map_load_time = self.read_time()
        resolution = self.read_float32()
        width = self.read_uint32()
        height = self.read_uint32()
        origin = self.read_pose()
        return {
            "map_load_time": map_load_time,
            "resolution": resolution,
            "width": width,
            "height": height,
            "origin": origin,
        }

    def decode_occupancy_grid(self, max_data_preview=20):
        # nav_msgs/OccupancyGrid:
        #   std_msgs/Header header
        #   MapMetaData info
        #   int8[] data
        header = self.read_header()
        info = self.decode_map_metadata()
        data_preview = self.read_int8_array(max_len=max_data_preview)
        return {
            "header": header,
            "info": info,
            "data_preview": data_preview,
        }

    def decode_tf_message(self, max_transforms=1):
        # tf2_msgs/TFMessage:
        #   geometry_msgs/TransformStamped[] transforms
        #
        # We keep it simple and only decode the FIRST transform for preview,
        # ignoring the rest to avoid alignment issues.

        # Number of transforms in the sequence
        num = self.read_uint32()

        if num == 0:
            return {"num_transforms": 0, "transforms_preview": []}

        # Align before first TransformStamped
        self.align(4)

        # --- TransformStamped #1 ---
        # std_msgs/Header header
        header = self.read_header()
        # string child_frame_id
        child_frame_id = self.read_string()
        # geometry_msgs/Transform:
        #   Vector3 translation
        #   Quaternion rotation
        translation = self.read_vector3()
        rotation = self.read_quaternion()

        transform0 = {
            "header": header,
            "child_frame_id": child_frame_id,
            "translation": translation,
            "rotation": rotation,
        }

        # We DO NOT attempt to parse further transforms; we just report how many
        # there supposedly are and show this first one as a preview.
        return {
            "num_transforms": num,
            "transforms_preview": [transform0],
        }




# ===================================================
# METADATA + DB ACCESS
# ===================================================
def load_metadata(bag_dir: Path):
    meta_path = bag_dir / "metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found at: {meta_path}")

    with meta_path.open("r") as f:
        meta = yaml.safe_load(f)

    return meta["rosbag2_bagfile_information"]


def find_db3_file(bag_dir: Path, metadata: dict) -> Path:
    relative_files = metadata.get("relative_file_paths", [])
    if not relative_files:
        raise ValueError("No .db3 file listed in metadata.yaml")

    first_db = bag_dir / relative_files[0]
    if not first_db.exists():
        raise FileNotFoundError(f"Referenced db3 file not found: {first_db}")

    return first_db


def inspect_db_topics(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM topics;")
    topics = []
    for row in cur.fetchall():
        topics.append(
            {
                "id": row["id"],
                "name": row["name"],
                "type": row["type"],
                "serialization_format": row["serialization_format"],
            }
        )

    conn.close()
    return topics


# ===================================================
# DECODING DISPATCH
# ===================================================
def decode_by_type(topic_type: str, data: bytes):
    """
    Dispatch decoding based on the ROS2 message type string.
    Returns a small python dict with readable fields.
    """
    reader = CDRReader(data)
    reader.skip_encapsulation_header()

    try:
        if topic_type == "geometry_msgs/msg/PointStamped":
            return reader.decode_point_stamped()

        if topic_type == "geometry_msgs/msg/PoseStamped":
            return reader.decode_pose_stamped()

        if topic_type == "geometry_msgs/msg/PoseWithCovarianceStamped":
            return reader.decode_pose_with_covariance_stamped()

        if topic_type == "sensor_msgs/msg/LaserScan":
            return reader.decode_laserscan()

        if topic_type == "std_msgs/msg/String":
            return reader.decode_string_msg()

        if topic_type == "nav_msgs/msg/MapMetaData":
            return reader.decode_map_metadata()

        if topic_type == "nav_msgs/msg/OccupancyGrid":
            return reader.decode_occupancy_grid()

        if topic_type == "tf2_msgs/msg/TFMessage":
            return reader.decode_tf_message()

        # Other types (visualization_msgs, rcl_interfaces, etc.) are not decoded here
        return None
    except Exception as e:
        # If decoding fails, just return error + fallback
        return {"_decode_error": str(e)}


# ===================================================
# SHOW MESSAGES
# ===================================================
def show_messages(db_path: Path, topic_id: int, topic_type: str, limit: int):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print(f"\n--- Topic ID {topic_id} ({topic_type}) | Showing up to {limit} messages ---")

    cur.execute(
        """
        SELECT timestamp, data FROM messages
        WHERE topic_id=?
        ORDER BY timestamp ASC
        LIMIT ?;
    """,
        (topic_id, limit),
    )

    rows = cur.fetchall()
    if not rows:
        print("  (No messages)")
        conn.close()
        return

    for i, row in enumerate(rows, start=1):
        ts = row["timestamp"]
        data = row["data"]
        preview_hex = data[:32].hex()

        print(f"\n  [{i}] timestamp={ts} | {ns_to_dt(ts).isoformat()}")
        print(f"      raw_len={len(data)} bytes")

        decoded = decode_by_type(topic_type, data)
        if decoded is None:
            print("      decoded: (no decoder for this type, showing hex preview)")
            print(f"      data_preview_hex={preview_hex}")
        else:
            if "_decode_error" in decoded:
                print("      decoded: ERROR decoding:", decoded["_decode_error"])
                print(f"      data_preview_hex={preview_hex}")
            else:
                print("      decoded:", decoded)

    conn.close()


# ===================================================
# MAIN
# ===================================================
def main():
    bag_dir = Path(BAG_DIR)

    print(f"=== Inspecting bag: {bag_dir} ===")

    # Load metadata
    metadata = load_metadata(bag_dir)

    print("\n=== Basic Info ===")
    print("Storage:", metadata.get("storage_identifier"))
    print("Compression:", metadata.get("compression_format"))
    print("Duration (ns):", metadata.get("duration", {}).get("nanoseconds"))
    print("Messages per topic:")

    for t in metadata.get("topics_with_message_count", []):
        print(
            " -",
            t["topic_metadata"]["name"],
            "| type:",
            t["topic_metadata"]["type"],
            "| count:",
            t["message_count"],
        )

    # Locate DB3 file
    db3_path = find_db3_file(bag_dir, metadata)
    print("\nUsing DB3 file:", db3_path)

    # Load topics table from db3
    topics = inspect_db_topics(db3_path)

    print("\n=== Topics in DB3 ===")
    for t in topics:
        print(f"ID={t['id']} | {t['name']} | {t['type']} | {t['serialization_format']}")

    # Choose which topics to show
    if SHOW_ONLY_TOPIC_ID is not None:
        topic_ids = [SHOW_ONLY_TOPIC_ID]
    else:
        topic_ids = [t["id"] for t in topics]

    # Show messages
    for t in topics:
        if t["id"] in topic_ids:
            print(f"\n### Topic: {t['name']} (ID={t['id']}, type={t['type']}) ###")
            show_messages(db3_path, t["id"], t["type"], MESSAGE_LIMIT)


if __name__ == "__main__":
    main()



