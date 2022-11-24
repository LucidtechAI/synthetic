import itertools
import math
from math import sqrt

import numpy as np
import pyamg
import scipy.sparse
from PIL import Image, ImageDraw
from skimage import feature
from sklearn.cluster import KMeans


# the below poisson blending function is courtesy of
# https://github.com/parosky/poissonblending/blob/master/poissonblending.py
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i, j in itertools.product(range(mask.shape[0]), range(mask.shape[1])):
            result[i][j] = 1 if sum(mask[i][j]) > 0 else 0
        mask = result
    return mask


# the below poisson blending function is courtesy of
# https://github.com/parosky/poissonblending/blob/master/poissonblending.py
def blend(image_target, image_source, image_mask, offset=(0, 0)):
    offset_x, offset_y = offset

    region_source_xmin = max(-offset_x, 0)
    region_source_ymin = max(-offset_y, 0)
    region_source_xmax = min(image_target.shape[0] - offset_x, image_source.shape[0])
    region_source_ymax = min(image_target.shape[1] - offset_y, image_source.shape[1])

    region_target_xmin = max(offset_x, 0)
    region_target_ymin = max(offset_y, 0)
    region_target_xmax = min(image_target.shape[0], image_source.shape[0] + offset_x)
    region_target_ymax = min(image_target.shape[1], image_source.shape[1] + offset_y)

    region_size = (
        region_source_xmax - region_source_xmin,
        region_source_ymax - region_source_ymin,
    )

    image_mask = image_mask[region_source_xmin:region_source_xmax, region_source_ymin:region_source_ymax]
    image_mask = prepare_mask(image_mask)
    image_mask[image_mask == 0] = False
    image_mask[image_mask != False] = True

    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y, x in itertools.product(range(region_size[0]), range(region_size[1])):
        if image_mask[y, x]:
            index = x + y * region_size[1]
            A[index, index] = 4
            if index + 1 < np.prod(region_size):
                A[index, index + 1] = -1
            if index - 1 >= 0:
                A[index, index - 1] = -1
            if index + region_size[1] < np.prod(region_size):
                A[index, index + region_size[1]] = -1
            if index - region_size[1] >= 0:
                A[index, index - region_size[1]] = -1
    A = A.tocsr()
    P = pyamg.gallery.poisson(image_mask.shape)

    for num_layer in range(image_target.shape[2]):
        t = image_target[region_target_xmin:region_target_xmax, region_target_ymin:region_target_ymax, num_layer]
        s = image_source[region_source_xmin:region_source_xmax, region_source_ymin:region_source_ymax, num_layer]
        t = t.flatten()
        s = s.flatten()

        b = P * s
        for y, x in itertools.product(range(region_size[0]), range(region_size[1])):
            index = x + y * region_size[1]
            if not image_mask[y, x]:
                b[index] = t[index]

        x = pyamg.solve(A, b, verb=False, tol=1E-10)
        x = np.reshape(x, region_size)
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, image_target.dtype)
        image_target[region_target_xmin:region_target_xmax, region_target_ymin:region_target_ymax, num_layer] = x

    return image_target


def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)


# TODO: sort colors by how close they are to the mean boundary image color, and say the closest is bg
def calculate_background_and_foreground_colors(image_array):
    kmeans = KMeans(n_clusters=2).fit(image_array.reshape(-1, image_array.shape[-1]))
    colors = kmeans.cluster_centers_[:2].astype(int)

    boundary_pixels = np.vstack([
        image_array[:, 0, :],
        image_array[:, -1, :],
        image_array[0, :, :],
        image_array[-1, :, :],
    ])
    colorarrs = [np.tile(c[np.newaxis, :], (boundary_pixels.shape[0], 1)) for c in colors]
    colordevs = [np.mean(np.abs(colorarr - boundary_pixels)) for colorarr in colorarrs]
    if colordevs[0] > colordevs[1]:
        colors = colors[::-1]

    return map(tuple, colors)


def create_text_image(text, image_size, fonts_by_size, background_color, foreground_color, angle):
    """
        Calculate which font and size we want to use. Loop over all fonts/sizes that fits the region and pick the size
        corresponding to the middle point between the smallest and largest font size that fits. Return an image with
        text rendered.
    """
    text_image = Image.new(mode='RGB', size=image_size, color=background_color)
    draw = ImageDraw.Draw(text_image)

    c = 1.5
    valid_sizes = []
    while not valid_sizes and c < 100:
        image_size_min = image_size[0] / c, image_size[1] / c

        for font_size in fonts_by_size.keys():
            tx, ty = draw.textsize(text=text, font=fonts_by_size[font_size])
            if (tx < image_size[0]) and (image_size_min[1] < ty < image_size[1]):
                valid_sizes.append(font_size)

        c *= 1.5

    if not valid_sizes:
        raise Exception('Could not draw text on image')

    fontsize = (min(valid_sizes) + max(valid_sizes)) // 2
    font = fonts_by_size[fontsize]
    xtext, ytext = draw.textsize(text=text, font=font)
    xmean, ymean = [int(s / 2) for s in text_image.size]
    xmin, ymin = xmean - xtext / 2, ymean - ytext / 2
    draw.text((xmin, ymin), text, font=font, fill=foreground_color)
    return text_image.rotate(angle, fillcolor=background_color)


def add_text_to_image(text, image_array, bounding_box, angle, fonts_by_size):
    image_array = np.array(image_array)
    xmin, ymin, xmax, ymax = bounding_box
    dx, dy = xmax - xmin, ymax - ymin
    boundary_ratio = 0.005
    boundary_ratio_x, boundary_ratio_y = boundary_ratio * sqrt(dy / dx), boundary_ratio * sqrt(dx / dy)
    bx, by = max(int(dx * boundary_ratio_x), 1), max(int(dy * boundary_ratio_y), 1)
    xmin_b, ymin_b = [r - 2 * b for r, b in [(xmin, bx), (ymin, by)]]
    xmax_b, ymax_b = [r + 2 * b for r, b in [(xmax, bx), (ymax, by)]]
    xmin_b, xmax_b = [clip(v, 1, image_array.shape[1] - 1) for v in [xmin_b, xmax_b]]
    ymin_b, ymax_b = [clip(v, 1, image_array.shape[0] - 1) for v in [ymin_b, ymax_b]]

    image_mask = np.zeros((ymax_b - ymin_b, xmax_b - xmin_b), dtype=np.uint8)
    image_mask[by:-by, bx:-bx] = 1

    image_target = np.array(image_array[ymin_b:ymax_b, xmin_b:xmax_b, :])
    background_color, foreground_color = calculate_background_and_foreground_colors(image_target)
    image_source = create_text_image(
        text=text,
        image_size=(xmax_b - xmin_b, ymax_b - ymin_b),
        fonts_by_size=fonts_by_size,
        background_color=background_color,
        foreground_color=foreground_color,
        angle=angle,
    )
    image_ret = blend(image_target, np.asarray(image_source), image_mask)

    image_array[ymin_b:ymax_b, xmin_b:xmax_b, :] = image_ret
    return image_array


def adjust_box(edginess, bounding_box):
    """
    Find the best bounding box by minimizing the difference in edges on the boundary
    """
    r = 0.1
    xmin, ymin, xmax, ymax = bounding_box
    dx, dy = int(xmax - xmin), int(ymax - ymin)
    dxy = max(int(r * sqrt(dy * dx)), 5)
    dx = dy = dxy
    dxh = dyh = int(dxy / 3)

    min_val = 1E20
    best_bounding_box = None
    D = max(2, int(np.sqrt(dx * dy) / 10))
    for possible_box in itertools.product(
        range(xmin - dx, xmin + dxh + 1),
        range(ymin - dy, ymin + dyh + 1),
        range(xmax - dxh, xmax + dx + 1),
        range(ymax - dxh, ymax + dy + 1),
    ):
        xmin_b, ymin_b, xmax_b, ymax_b = possible_box
        e1 = edginess[ymin_b - D:ymax_b + D + 1, xmin_b - D:xmax_b + D + 1]
        e2 = edginess[ymin_b + D:ymax_b - D + 1, xmin_b + D:xmax_b - D + 1]
        E = np.sum(e1) - np.sum(e2)
        if E < min_val:
            min_val = E
            best_bounding_box = xmin_b, ymin_b, xmax_b, ymax_b

    return best_bounding_box


def rel_to_abs(bounding_box, image_size):
    return tuple(map(int, [
        bounding_box[0] * image_size[0],
        bounding_box[1] * image_size[1],
        bounding_box[2] * image_size[0],
        bounding_box[3] * image_size[1],
    ]))


# TODO: consider doing a check here to ensure boxes do not overlap
def blend_image(image: Image, bounding_boxes: list, fonts: dict, max_size: int):
    # Resize if image is too large. Large images will have a performance impact
    if image.size[0] * image.size[1] > max_size:
        s = math.sqrt(image.size[0] * image.size[1] / max_size)
        resized_image = image.resize(tuple(map(int, (image.size[0] / s, image.size[1] / s))))
        return blend_image(resized_image, bounding_boxes, fonts, max_size).resize(image.size)

    bounding_boxes = [(rel_to_abs(bounding_box, image.size), text) for bounding_box, text in bounding_boxes]
    image_binary_arr = np.asarray(image.convert('L'))
    scale = np.sqrt(image_binary_arr.size) / 900
    edginess = feature.canny(image_binary_arr, sigma=scale)
    adjusted_boxes = [(adjust_box(edginess, bounding_box), text) for bounding_box, text in bounding_boxes]

    image_array = np.asarray(image)
    for bounding_box, text in adjusted_boxes:
        image_array = add_text_to_image(text, image_array, bounding_box, 0, fonts)

    return Image.fromarray(np.uint8(image_array))
