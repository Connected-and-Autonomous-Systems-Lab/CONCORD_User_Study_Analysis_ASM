#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot_trajectory.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("./Output/Mugunthan/Easy/rosbag2_2025_11_26-18_13_26_tf.csv")      # change if needed
OUT_FIG_2D = Path("human_traj_xy.png")
OUT_FIG_3D = Path("human_traj_xyz.png")

def main():
    # 1) Load CSV
    df = pd.read_csv(CSV_PATH)

    # 2) Keep only human odom -> base_footprint transforms
    mask = (
        (df["parent_frame"] == "human/odom") &
        (df["child_frame"] == "human/base_footprint")
    )
    hdf = df[mask].copy()

    if hdf.empty:
        print("No rows with human/odom -> human/base_footprint found!")
        return

    # 3) Create a time column in seconds (relative to first sample)
    hdf["time_s"] = (hdf["bag_timestamp_ns"] - hdf["bag_timestamp_ns"].iloc[0]) / 1e9

    # 4) 2D trajectory plot (x-y)
    plt.figure()
    plt.plot(hdf["x"], hdf["y"])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Human Avatar Trajectory (x-y)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_FIG_2D, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
