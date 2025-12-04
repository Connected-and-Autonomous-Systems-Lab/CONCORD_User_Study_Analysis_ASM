#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot maze path from dark_floor_xy_transformed_sorted.csv.

- Expects a CSV file with columns: x, y
- Each (x, y) is the center of a 7x7 floor tile in the maze.
- Produces an image 'maze_path.png' in the same folder as the script.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# CONFIG
# ======================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MAZE_FLOOR_CSV = SCRIPT_DIR / "dark_floor_xy_transformed_sorted.csv"
OUT_PNG = SCRIPT_DIR / "maze_path.png"


def main():
    if not MAZE_FLOOR_CSV.exists():
        print(f"[ERROR] Maze floor CSV not found: {MAZE_FLOOR_CSV}")
        return

    try:
        df = pd.read_csv(MAZE_FLOOR_CSV)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV {MAZE_FLOOR_CSV}: {e}")
        return

    if not {"X", "Y"}.issubset(df.columns):
        print(f"[ERROR] CSV must contain 'x' and 'y' columns. Found: {list(df.columns)}")
        return

    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    print(f"[INFO] Loaded {len(df)} tile centers from {MAZE_FLOOR_CSV}")

    plt.figure(figsize=(8, 8))

    # Scatter each tile center; use square marker to visually suggest tiles
    plt.scatter(x, y, s=650, marker="s", alpha=0.5, edgecolors="none")

    plt.title("Maze path (tile centers)")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Equal aspect so maze geometry is not distorted
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    print(f"[OK] Saved maze path plot to {OUT_PNG}")

    # If you also want an interactive window, uncomment this:
    # plt.show()

    plt.close()


if __name__ == "__main__":
    main()
