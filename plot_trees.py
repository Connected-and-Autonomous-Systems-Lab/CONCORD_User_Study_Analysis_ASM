#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import matplotlib.pyplot as plt

MAZE_TREES = {
    "Easy": {
        "root": "start",
        "children": {
            "start": ["EJ1", "DE_E1"],
            "EJ1": ["DE_E2", "DE_E3"],
        },
    },
    "Hard": {
        "root": "start",
        "children": {
            "start": ["HJ1", "HJ11"],
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

# ---------------------------------------------------------------------
# Compute positions
# ---------------------------------------------------------------------
def compute_positions(children: dict, root: str):
    pos = {}
    x_counter = [0.0]

    def dfs(node, depth):
        child_list = children.get(node, [])
        if not child_list:
            x = x_counter[0]
            pos[node] = (x, -depth * 2.0)
            x_counter[0] += 2.0
        else:
            xs = []
            for ch in child_list:
                dfs(ch, depth + 1)
                xs.append(pos[ch][0])
            pos[node] = (sum(xs) / len(xs), -depth * 2.0)

    dfs(root, 0)
    return pos


# ---------------------------------------------------------------------
# Plot a tree
# ---------------------------------------------------------------------
def plot_tree(tree_info: dict, title: str, output_file: str):
    children = tree_info["children"]
    root = tree_info["root"]
    pos = compute_positions(children, root)

    # ------------------------------
    # Label offsets
    # ------------------------------
    if title.startswith("Easy"):
        circle_offset_y = 0.2   # ← move closer for Easy
        box_offset_x    = 0.2   # ← move closer for Easy
    else:
        circle_offset_y = 0.6   # Hard (original)
        box_offset_x    = 0.7   # Hard (original)

    plt.figure(figsize=(10, 8))

    # Draw edges
    for parent, child_list in children.items():
        x1, y1 = pos[parent]
        for ch in child_list:
            x2, y2 = pos[ch]
            plt.plot([x1, x2], [y1, y2], color="black", linewidth=2.8)

    # Draw nodes
    for node, (x, y) in pos.items():

        if node.startswith("DE_"):
            plt.scatter(x, y, s=900, marker="s",
                        color="#ffdddd", edgecolors="black", linewidth=1.6)

            # Dead-end label (to the right)
            plt.text(x + box_offset_x, y, node,
                     fontsize=15, ha="left", va="center")

        else:
            plt.scatter(x, y, s=900, marker="o",
                        color="#ddeaff", edgecolors="black", linewidth=1.6)

            # Junction label (above)
            plt.text(x, y + circle_offset_y, node,
                     fontsize=15, ha="center", va="bottom")

    # plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    plot_tree(MAZE_TREES["Easy"], "Easy Maze Tree", "Easy_Maze_Tree.png")
    plot_tree(MAZE_TREES["Hard"], "Hard Maze Tree", "Hard_Maze_Tree.png")


if __name__ == "__main__":
    main()
