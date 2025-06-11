import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

output_dir = "../../imgs/radon"

img_size = 225


def save_image(img, filename):
    plt.imsave(os.path.join(output_dir, filename), img, cmap="gray", vmin=0, vmax=255)


# 1. half black, half white
half_bw = np.zeros((img_size, img_size), dtype=np.uint8)
half_bw[:, img_size // 2 :] = 255
save_image(255 - half_bw, "half_black_white.png")

# 2. concave luna
luna = np.zeros((img_size, img_size), dtype=np.uint8)
yy, xx = np.ogrid[:img_size, :img_size]
center1 = (img_size // 2, img_size // 2)
center2 = (img_size // 2 + 30, img_size // 2)
radius = 80
mask1 = (xx - center1[0]) ** 2 + (yy - center1[1]) ** 2 <= radius**2
mask2 = (xx - center2[0]) ** 2 + (yy - center2[1]) ** 2 <= radius**2
luna[mask1] = 255
luna[mask2] = 0
save_image(255 - luna, "concave_luna.png")

# 3. blurred circle
blurred = np.zeros((img_size, img_size), dtype=np.float32)
radius = img_size // 6
circle_center = (img_size // 2, img_size // 2)
mask = (xx - circle_center[0]) ** 2 + (yy - circle_center[1]) ** 2 <= radius**2
blurred[mask] = 1.0
blurred = gaussian_filter(blurred, sigma=5)
blurred = (blurred / blurred.max()) * 255
save_image(255 - blurred.astype(np.uint8), "circle.png")

# 4. Vertical lines
lines = np.zeros((img_size, img_size), dtype=np.uint8)
for x in range(0, img_size, 20):
    lines[:, x : x + 10] = 255
save_image(255 - lines, "vertical_lines.png")

# 5. triangle outline (10px thick, no fill)
border_thickness = 10
image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

half_size = img_size // 2
center_x, center_y = img_size // 2, img_size // 2
triangle_height = int(np.sqrt(3) / 2 * half_size)

triangle = np.array(
    [
        [center_x, center_y - triangle_height // 2],  # top vertex
        [center_x - half_size // 2, center_y + triangle_height // 2],  # bottom left
        [center_x + half_size // 2, center_y + triangle_height // 2],  # bottom right
    ]
)


def draw_thick_line(img, pt1, pt2, thickness, color):
    Y, X = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")

    line_vec = pt2 - pt1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return
    line_dir = line_vec / line_len

    pt1_to_pixel = np.stack((X - pt1[0], Y - pt1[1]), axis=-1)

    proj_len = pt1_to_pixel[..., 0] * line_dir[0] + pt1_to_pixel[..., 1] * line_dir[1]

    perp_dist = np.linalg.norm(pt1_to_pixel - proj_len[..., None] * line_dir, axis=-1)

    mask = (proj_len >= 0) & (proj_len <= line_len) & (perp_dist <= thickness / 2)

    img[mask] = color


black = [0, 0, 0]
for i in range(3):
    draw_thick_line(image, triangle[i], triangle[(i + 1) % 3], border_thickness, black)

save_image(image, "triangle.png")
