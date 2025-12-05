#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
PREFAB_PATH = Path("Easy_10.prefab")
OUTPUT_CSV = Path("bright_floor_xy_transformed_sorted.csv")


def parse_unity_prefab_blocks(prefab_path: Path):
    with prefab_path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    blocks = []
    current = None
    header_re = re.compile(r"^--- !u!(\d+) &(\d+)")

    for line in lines:
        m = header_re.match(line)
        if m:
            if current is not None:
                blocks.append(current)
            class_id, file_id = m.groups()
            current = {"class_id": class_id, "file_id": file_id, "lines": []}
        else:
            if current is not None:
                current["lines"].append(line)

    if current is not None:
        blocks.append(current)

    return blocks


def find_dark_floor_gameobject_ids(blocks):
    dark_ids = set()
    for b in blocks:
        if b["class_id"] == "1":  # GameObject
            for line in b["lines"]:
                if "m_Name:" in line and "Bright Floor(Clone)" in line:
                    dark_ids.add(b["file_id"])
                    break
    return dark_ids


def extract_positions_for_dark_floors(blocks, dark_go_ids):
    positions = []
    for b in blocks:
        if b["class_id"] != "4":  # Transform
            continue

        body = "\n".join(b["lines"])

        m_go = re.search(r"m_GameObject:\s*{fileID:\s*(\d+)}", body)
        if not m_go:
            continue
        go_id = m_go.group(1)

        if go_id not in dark_go_ids:
            continue

        m_pos = re.search(
            r"m_LocalPosition:\s*{x:\s*([-\d.eE+]+),\s*y:\s*([-\d.eE+]+),\s*z:\s*([-\d.eE+]+)}",
            body,
        )
        if not m_pos:
            continue

        x, y, z = map(float, m_pos.groups())
        positions.append((x, y, z))

    return positions


def transform_positions_xy(positions):
    transformed = []
    for x, y, z in positions:
        new_x = -z
        new_y = x
        transformed.append((new_x, new_y))
    return transformed


def save_xy_to_csv_sorted(xy_list, csv_path: Path):
    # Sort ascending by X (primary), then Y (secondary)
    xy_list_sorted = sorted(xy_list, key=lambda t: (t[0], t[1]))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])
        writer.writerows(xy_list_sorted)


def main():
    if not PREFAB_PATH.exists():
        print(f"ERROR: Prefab file not found: {PREFAB_PATH}")
        return

    print(f"Reading prefab: {PREFAB_PATH}")
    blocks = parse_unity_prefab_blocks(PREFAB_PATH)

    dark_go_ids = find_dark_floor_gameobject_ids(blocks)
    print(f"Found {len(dark_go_ids)} Dark Floor(Clone) GameObjects")

    positions = extract_positions_for_dark_floors(blocks, dark_go_ids)
    print(f"Extracted {len(positions)} world positions")

    transformed_xy = transform_positions_xy(positions)

    save_xy_to_csv_sorted(transformed_xy, OUTPUT_CSV)
    print(f"Saved sorted transformed X,Y to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
