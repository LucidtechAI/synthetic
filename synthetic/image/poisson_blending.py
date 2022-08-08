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
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask


# the below poisson blending function is courtesy of
# https://github.com/parosky/poissonblending/blob/master/poissonblending.py
def blend(image_target, image_source, image_mask, offset=(0, 0)):
    region_source = (
        max(-offset[0], 0),
        max(-offset[1], 0),
        min(image_target.shape[0] - offset[0], image_source.shape[0]),
        min(image_target.shape[1] - offset[1], image_source.shape[1]),
    )
    region_target = (
        max(offset[0], 0),
        max(offset[1], 0),
        min(image_target.shape[0], image_source.shape[0] + offset[0]),
        min(image_target.shape[1], image_source.shape[1] + offset[1]),
    )
    region_size = (
        region_source[2] - region_source[0],
        region_source[3] - region_source[1],
    )

    image_mask = image_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    image_mask = prepare_mask(image_mask)
    image_mask[image_mask == 0] = False
    image_mask[image_mask != False] = True

    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if image_mask[y, x]:
                index = x + y * region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index + 1] = -1
                if index-1 >= 0:
                    A[index, index - 1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index + region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index - region_size[1]] = -1
    A = A.tocsr()
    P = pyamg.gallery.poisson(image_mask.shape)

    for num_layer in range(image_target.shape[2]):
        t = image_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer]
        s = image_source[region_source[0]:region_source[2], region_source[1]:region_source[3], num_layer]
        t = t.flatten()
        s = s.flatten()

        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                index = x + y * region_size[1]
                if not image_mask[y, x]:
                    b[index] = t[index]

        x = pyamg.solve(A, b, verb=False, tol=1E-10)
        x = np.reshape(x, region_size)
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, image_target.dtype)
        image_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer] = x

    return image_target


def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)


# TODO: sort colors by how close they are to the mean boundary image color, and say the closest is bg
def get_bg_fg_cols(image_array):
    kmeans = KMeans(n_clusters=2).fit(image_array.reshape(-1, image_array.shape[-1]))
    colors = kmeans.cluster_centers_[:2]

    bdy_pixels = np.vstack([
        image_array[:, 0, :],
        image_array[:, -1, :],
        image_array[0, :, :],
        image_array[-1, :, :],
    ])
    colors = [np.array(list(map(int, c))) for c in [colors[0], colors[1]]]
    colorarrs = [np.tile(c[np.newaxis, :], (bdy_pixels.shape[0], 1)) for c in colors]
    colordevs = [np.mean(np.abs(colorarr - bdy_pixels)) for colorarr in colorarrs]
    if colordevs[0] > colordevs[1]:
        colors = colors[::-1]

    return map(tuple, colors)


def text_to_image(text, image_size, fonts_by_size, bg_color, fg_color, angle):
    testimage = Image.new(mode='RGB', size=image_size, color=bg_color)
    draw = ImageDraw.Draw(testimage)

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
    tx, ty = draw.textsize(text=text, font=font)
    xm, ym = [int(s / 2) for s in testimage.size]
    x0, y0 = xm - tx / 2, ym - ty / 2
    draw.text((x0, y0), text, font=font, fill=fg_color)
    return testimage.rotate(angle, fillcolor=bg_color)


def add_text_to_image(text, image_array, bounding_box, angle, fonts_by_size):
    image_array = np.array(image_array)
    xmin, ymin, xmax, ymax = bounding_box
    dx, dy = xmax - xmin, ymax - ymin
    bdy_ratio = 0.005
    bdy_ratio_x, bdy_ratio_y = bdy_ratio * sqrt(dy / dx), bdy_ratio * sqrt(dx / dy)
    bx, by = max(int(dx * bdy_ratio_x), 1), max(int(dy * bdy_ratio_y), 1)
    xmb, ymb = [r - 2 * b for r, b in [(xmin, bx), (ymin, by)]]
    xMb, yMb = [r + 2 * b for r, b in [(xmax, bx), (ymax, by)]]
    xmb, xMb = [clip(v, 1, image_array.shape[1] - 1) for v in [xmb, xMb]]
    ymb, yMb = [clip(v, 1, image_array.shape[0] - 1) for v in [ymb, yMb]]

    image_mask = np.zeros((yMb - ymb, xMb - xmb), dtype=np.uint8)
    image_mask[by:-by, bx:-bx] = 1

    image_target = np.array(image_array[ymb:yMb, xmb:xMb, :])
    bg_color, fg_color = get_bg_fg_cols(image_target)
    image_source = text_to_image(text, image_target.shape[:2][::-1], fonts_by_size, bg_color, fg_color, angle)
    image_ret = blend(image_target, np.asarray(image_source), image_mask)

    image_array[ymb:yMb, xmb:xMb, :] = image_ret
    return image_array


def adjust_box(edginess, bounding_box):
    r = 0.1
    xmin, ymin, xmax, ymax = bounding_box
    dx, dy = int(xmax - xmin), int(ymax - ymin)
    dx, dy = int(r * sqrt(dy * dx)), int(r * sqrt(dy * dx))
    dx, dy = max(dx, 5), max(dy, 5)
    dxh, dyh = int(dx / 3), int(dy / 3)

    min_val = 1E20
    best_bounding_box = None
    D = max(2, int(np.sqrt(dx * dy) / 10))
    for possible_box in itertools.product(
        range(xmin - dx, xmin + dxh + 1),
        range(ymin - dy, ymin + dyh + 1),
        range(xmax - dxh, xmax + dx + 1),
        range(ymax - dxh, ymax + dy + 1),
    ):
        xm, ym, xM, yM = possible_box
        e1 = edginess[ym - D:yM + D + 1, xm - D:xM + D + 1]
        e2 = edginess[ym + D:yM - D + 1, xm + D:xM - D + 1]
        E = np.sum(e1) - np.sum(e2)
        if E < min_val:
            min_val = E
            best_bounding_box = xm, ym, xM, yM

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
    image_bw_arr = np.asarray(image.convert('L'))
    scale = np.sqrt(image_bw_arr.size) / 900
    edginess = feature.canny(image_bw_arr, sigma=scale)
    adjusted_boxes = [(adjust_box(edginess, bounding_box), text) for bounding_box, text in bounding_boxes]

    image_array = np.asarray(image)
    for bounding_box, text in adjusted_boxes:
        image_array = add_text_to_image(text, image_array, bounding_box, 0, fonts)

    return Image.fromarray(np.uint8(image_array))
