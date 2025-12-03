

import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

import rarfile




CANNY_LOW = 50
CANNY_HIGH = 150

NUM_RAYS = 31              # Number of vertical rays
OPEN_THRESHOLD = 0.30      # Opening threshold
LEFT_MAX = 0.33            # Normalized x for left boundary
RIGHT_MIN = 0.67           # Normalized x for right boundary




def parse_frame_id(filename: str) -> int:
    
    m = re.search(r"(\d+)_([0-9]+)", filename)
    if m:
        sec = int(m.group(1))
        nsec = int(m.group(2))
        return sec * 1_000_000_000 + nsec

    m2 = re.search(r"(\d+)", filename)
    if m2:
        return int(m2.group(1))

    return -1


def distance_to_edge(edges: np.ndarray, x: int) -> int:
    h, _ = edges.shape
    for y in range(h - 1, -1, -1):
        if edges[y, x] != 0:
            return (h - 1) - y
    return h - 1


def build_rays_and_openings(edges: np.ndarray):
    h, w = edges.shape
    xs = np.linspace(0, w - 1, NUM_RAYS).astype(int).tolist()

    dists = []
    open_mask = []
    for x in xs:
        d = distance_to_edge(edges, x)
        dists.append(d)
        open_mask.append((d / h) >= OPEN_THRESHOLD)

    # cluster consecutive open rays
    clusters = []
    in_cluster = False
    start = None

    for i, is_open in enumerate(open_mask):
        if is_open and not in_cluster:
            in_cluster = True
            start = i
        elif not is_open and in_cluster:
            clusters.append((start, i - 1))
            in_cluster = False
            start = None

    if in_cluster and start is not None:
        clusters.append((start, len(open_mask) - 1))

    cluster_records = []
    for a, b in clusters:
        center_idx = (a + b) // 2
        x_center = xs[center_idx]
        x_norm = x_center / (w - 1) if w > 1 else 0.5

        if x_norm <= LEFT_MAX:
            side = "left"
        elif x_norm >= RIGHT_MIN:
            side = "right"
        else:
            side = "forward"

        cluster_records.append(
            {
                "start_idx": a,
                "end_idx": b,
                "x_center": x_center,
                "side": side,
            }
        )

    rays_info = {
        "xs": xs,
        "dists": dists,
        "open_mask": open_mask,
        "h": h,
        "w": w,
    }

    return rays_info, cluster_records


def classify_scene(num_left, num_forward, num_right, num_total):

    open_left = num_left > 0
    open_forward = num_forward > 0
    open_right = num_right > 0

    n_open_dirs = int(open_left) + int(open_forward) + int(open_right)

    # Fine label logic
    if num_total <= 1:
        if open_forward:
            fine = "straight_corridor"
        elif open_left:
            fine = "side_open_left"
        elif open_right:
            fine = "side_open_right"
        else:
            fine = "blocked"
    else:
        if not open_forward and (open_left ^ open_right):
            fine = "corner_left" if open_left else "corner_right"
        elif not open_forward and open_left and open_right:
            fine = "t_junction"
        elif open_forward and open_left and open_right:
            fine = "three_way_or_four_way"
        elif open_forward and open_left:
            fine = "corridor_branch_left"
        elif open_forward and open_right:
            fine = "corridor_branch_right"
        else:
            fine = "complex"

    # coarse labels compatible with old version
    if n_open_dirs <= 1:
        coarse = "no_choice"
    elif n_open_dirs == 2:
        coarse = "junction_two_options"
    else:
        coarse = "corridor_multi_options"

    return fine, coarse, open_left, open_forward, open_right, n_open_dirs




def process_single_image(img_path: Path, debug_dir: Path | None):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot load image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(gray_blur, CANNY_LOW, CANNY_HIGH)

    rays_info, clusters = build_rays_and_openings(edges)

    num_left = sum(1 for c in clusters if c["side"] == "left")
    num_forward = sum(1 for c in clusters if c["side"] == "forward")
    num_right = sum(1 for c in clusters if c["side"] == "right")
    num_total = len(clusters)

    w = rays_info["w"]
    h = rays_info["h"]

    # legacy distance samples
    x_left = int(w * 0.25)
    x_center = int(w * 0.50)
    x_right = int(w * 0.75)
    d_left = distance_to_edge(edges, x_left)
    d_center = distance_to_edge(edges, x_center)
    d_right = distance_to_edge(edges, x_right)

    # classification
    fine_label, coarse_label, open_left, open_forward, open_right, n_open_dirs = \
        classify_scene(num_left, num_forward, num_right, num_total)

    # debug visualization
    if debug_dir is not None:
        vis = img.copy()
        for x, d, is_open in zip(rays_info["xs"], rays_info["dists"], rays_info["open_mask"]):
            color = (0, 255, 0) if is_open else (0, 0, 255)
            cv2.line(vis, (x, h - 1), (x, h - 1 - d), color, 1)

        txt = f"{fine_label} | L/F/R={num_left}/{num_forward}/{num_right}"
        cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imwrite(str(debug_dir / img_path.name), vis)

    return {
        "fine_label": fine_label,
        "coarse_label": coarse_label,
        "num_openings_total": num_total,
        "num_left_openings": num_left,
        "num_forward_openings": num_forward,
        "num_right_openings": num_right,
        "open_left": int(open_left),
        "open_forward": int(open_forward),
        "open_right": int(open_right),
        "n_open_directions": n_open_dirs,
        "d_left_px": d_left,
        "d_center_px": d_center,
        "d_right_px": d_right,
    }




def build_decision_segments(df_frames: pd.DataFrame):
    segs = []
    in_seg = False
    start = None
    fine_labels = []
    coarse_labels = []
    totals = []

    for idx, row in df_frames.iterrows():
        is_decision = row["num_openings_total"] >= 2

        if is_decision and not in_seg:
            in_seg = True
            start = idx
            fine_labels = [row["fine_label"]]
            coarse_labels = [row["coarse_label"]]
            totals = [row["num_openings_total"]]

        elif is_decision and in_seg:
            fine_labels.append(row["fine_label"])
            coarse_labels.append(row["coarse_label"])
            totals.append(row["num_openings_total"])

        elif not is_decision and in_seg:
            segs.append((start, idx - 1, fine_labels, coarse_labels, totals))
            in_seg = False

    if in_seg:
        segs.append((start, df_frames.index[-1], fine_labels, coarse_labels, totals))

    records = []
    for i, (s, e, fl, cl, tot) in enumerate(segs, start=1):
        first = df_frames.loc[s]
        last = df_frames.loc[e]

        records.append({
            "decision_id": i,
            "first_image": first["image"],
            "last_image": last["image"],
            "first_frame_id": first["frame_id"],
            "last_frame_id": last["frame_id"],
            "start_row_index": s,
            "end_row_index": e,
            "length_in_frames": e - s + 1,
            "majority_fine_label": pd.Series(fl).mode().iloc[0],
            "majority_coarse_label": pd.Series(cl).mode().iloc[0],
            "max_num_openings_total": int(max(tot)),
        })

    return pd.DataFrame.from_records(records)




def process_maze_folder(input_folder: str | Path, output_folder: str | Path, debug=False):
    
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_vis" if debug else None
    if debug:
        debug_dir.mkdir(exist_ok=True)

    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        img_paths.extend(list(input_dir.glob(ext)))

    if not img_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    # sort by timestamp or number inside filename
    img_paths = sorted(img_paths, key=lambda p: parse_frame_id(p.name))

    frames = []

    for img_path in img_paths:
        fname = img_path.name
        f_id = parse_frame_id(fname)
        print(f"Processing {fname} (frame_id={f_id}) ...")

        try:
            info = process_single_image(img_path, debug_dir)
        except Exception as e:
            print("Error:", e)
            info = {
                "fine_label": "error",
                "coarse_label": "error",
                "num_openings_total": -1,
                "num_left_openings": -1,
                "num_forward_openings": -1,
                "num_right_openings": -1,
                "open_left": 0,
                "open_forward": 0,
                "open_right": 0,
                "n_open_directions": -1,
                "d_left_px": -1,
                "d_center_px": -1,
                "d_right_px": -1,
            }

        frames.append({
            "image": fname,
            "frame_id": f_id,
            **info
        })

    df = pd.DataFrame(frames)
    df = df.sort_values("frame_id").reset_index(drop=True)

    # Save per-frame CSV
    frames_csv = output_dir / "maze_multi_option_results.csv"
    df.to_csv(frames_csv, index=False)
    print(f"\nSaved frame CSV to: {frames_csv}")

    # Save decision segments CSV
    seg_df = build_decision_segments(df)
    seg_csv = output_dir / "maze_decision_segments.csv"
    seg_df.to_csv(seg_csv, index=False)
    print(f"Saved decision segments CSV to: {seg_csv}")

    return df, seg_df
