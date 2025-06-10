import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def plot_inner_outer_pegs(num_pegs=16, i=2):
    angles = np.linspace(0, 2 * np.pi, num_pegs, endpoint=False)

    r_inner = 1
    r_outer = 2 * r_inner

    # peg coordinates
    inner_points = np.column_stack((r_inner * np.cos(angles), r_inner * np.sin(angles)))
    outer_points = np.column_stack((r_outer * np.cos(angles), r_outer * np.sin(angles)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.grid(True, zorder=0)
    ax.axis("on")

    ax.set_xlim(-r_outer * 1.1, r_outer * 1.1)
    ax.set_ylim(-r_outer * 1.1, r_outer * 1.1)

    # polygons
    ax.add_patch(Polygon(inner_points, closed=True, fill=False, edgecolor="black", linewidth=1, zorder=4))
    ax.add_patch(Polygon(outer_points, closed=True, fill=False, edgecolor="black", linewidth=1, zorder=4))

    # pegs
    ax.plot(inner_points[:, 0], inner_points[:, 1], "ko", markersize=3, zorder=5)
    ax.plot(outer_points[:, 0], outer_points[:, 1], "ko", markersize=3, zorder=5)
    ax.plot(0, 0, "ko", markersize=4, zorder=5)  # center point

    center = np.array([0, 0])
    i_mod = i % num_pegs
    i_next_mod = (i + 1) % num_pegs

    dir1 = inner_points[i_mod] / np.linalg.norm(inner_points[i_mod])
    dir2 = outer_points[i_next_mod] / np.linalg.norm(outer_points[i_next_mod])

    extension_length = r_outer * 10
    line1_end = center + dir1 * extension_length
    line2_end = center + dir2 * extension_length

    # lines
    ax.plot([center[0], line1_end[0]], [center[1], line1_end[1]], "k-", zorder=5)
    ax.plot([center[0], line2_end[0]], [center[1], line2_end[1]], "k-", zorder=5)

    # labels
    label_offset = 0.3
    label_pos1 = center + dir1 * (r_outer + label_offset)
    label_pos2 = center + dir2 * (r_outer + label_offset)
    ax.text(label_pos1[0] - 0.2, label_pos1[1], "A", fontsize=12, ha="center", va="center")
    ax.text(label_pos2[0] - 0.15, label_pos2[1], "B", fontsize=12, ha="center", va="center")

    # function to check if a point is between two vectors (CCW)
    def is_between(p, a, b):
        def ccw(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        return ccw(center, a, p) >= 0 and ccw(center, b, p) <= 0

    # inner circle
    inner_section_pts = [center]
    for pt in inner_points:
        if is_between(pt, inner_points[i_mod], outer_points[i_next_mod]):
            inner_section_pts.append(pt)
    inner_section_pts.append(center)
    inner_patch = Polygon(
        inner_section_pts,
        closed=True,
        fill=True,
        facecolor="#ff858d",
        edgecolor="black",
        linewidth=1,
        zorder=3,
        alpha=0.5,
    )
    ax.add_patch(inner_patch)

    # outer circle
    outer_section_pts = [center]
    for pt in outer_points:
        if is_between(pt, inner_points[i_mod], outer_points[i_next_mod]):
            outer_section_pts.append(pt)
    outer_section_pts.append(center)
    outer_patch = Polygon(
        outer_section_pts,
        closed=True,
        fill=True,
        facecolor="#66a8ff",
        edgecolor="black",
        linewidth=1,
        zorder=2,
        alpha=0.5,
    )
    ax.add_patch(outer_patch)

    plt.tight_layout(pad=0)
    plt.subplots_adjust(right=0.98)
    plt.savefig("../../docs/assets/upsampling.svg")
    plt.show()


if __name__ == "__main__":
    plot_inner_outer_pegs(num_pegs=24, i=2)
