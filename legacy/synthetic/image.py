import math
import random
import glob
import numpy as np
import PIL.Image
import pathlib
import logging
import itertools
import io

from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from collections import defaultdict
from math import sqrt
from skimage import feature
from synthetic.blend import blend as image_blend


def compress(img, q=None):
    dest = io.BytesIO()
    img.save(dest, 'JPEG', quality=q if q is not None else random.randint(45, 80))
    return Image.open(dest)


def get_maxes(a):
    # Given an array of images, find local maxima
    # TODO: has some hard-coded values which are appropriate for finding colors but probably not in general
    idx, hts = find_peaks(a, height=-10, prominence=0.9, distance=10)
    hts = hts['peak_heights']
    idx = idx[np.argsort(hts)[::-1]]
    return idx


def compute_bounding_box_area(edge_img):
    # assume edge_img is in {0, 255}
    e = np.asarray(edge_img)
    X = np.arange(e.shape[0])[np.max(e, axis=1) == 255]
    xmin, xmax = X[0], X[-1]
    Y = np.arange(e.shape[1])[np.max(e, axis=0) == 255]
    ymin, ymax = Y[0], Y[-1]
    return (ymax-ymin) * (xmax-xmin)


def compute_angle(edge_img, N_sample=10):
    # TODO: Optimize e.g. with ternary search
    angles = np.linspace(-45, 45, N_sample)
    edge_img = no_bdy_artifact(edge_img)

    areas = []
    for t in angles:
        areas.append(compute_bounding_box_area(edge_img.rotate(t)))

    return angles[np.argmin(np.array(areas))]


def no_bdy_artifact(edge_img):
    # Expand img to ensure rotation by up to abs(45) degrees will not go out of bounds
    # TODO: Verify that this is correct

    D = int(sum(edge_img.size) * 1.2)
    new_img = Image.new(edge_img.mode, (D, D))
    ymin = (D - edge_img.size[0]) // 2
    xmin = (D - edge_img.size[1]) // 2
    new_img.paste(edge_img, (ymin, xmin))

    return new_img


def get_fonts(fonts_dir, font_sizes):
    fonts = defaultdict(lambda: defaultdict(dict))

    all_fonts = itertools.chain.from_iterable(
        [glob.glob(str(pathlib.Path(fonts_dir) / f'*.{ext}'), recursive=True)
         for ext in ['ttf', 'otf']]
    )

    for fn in all_fonts:
        for font_size in font_sizes:
            try:
                fonts[fn][font_size] = ImageFont.truetype(fn, font_size)
            except Exception as e:
                logging.error(e)

    return fonts


def get_bg_fg_cols(imgarr):
    kmeans = KMeans(n_clusters=2).fit(imgarr.reshape(-1, imgarr.shape[-1]))
    colors = kmeans.cluster_centers_[:2]

    # TODO: Sort colors by how close they are to the mean boundary image color, and say the closest is bg

    bdy_pixels = np.vstack([imgarr[:, 0, :], imgarr[:, -1, :], imgarr[0, :, :], imgarr[-1, :, :]])
    colors = [np.array(list(map(int, c))) for c in [colors[0], colors[1]]]
    colorarrs = [np.tile(c[np.newaxis, :], (bdy_pixels.shape[0], 1)) for c in colors]
    colordevs = [np.mean(np.abs(colorarr - bdy_pixels)) for colorarr in colorarrs]

    if colordevs[0] > colordevs[1]:
        colors = colors[::-1]

    return map(tuple, colors)


def arr_to_img(arr):
    return PIL.Image.fromarray(np.uint8(arr))


def text_to_img(text, img_size, fonts_by_size, bg_color, fg_color, angle):
    c = 1.5
    testimg = Image.new(mode='RGB', size=img_size, color=bg_color)
    draw = ImageDraw.Draw(testimg)

    while True:
        img_size_min = img_size[0] / c, img_size[1] / c

        valid_sizes = []
        for font_size in fonts_by_size.keys():
            tx, ty = draw.textsize(text=text, font=fonts_by_size[font_size])
            # am not checking x-size because i currently write really short strngs
            if (tx < img_size[0]) and (img_size_min[1] < ty < img_size[1]):
                valid_sizes.append(font_size)
        if not valid_sizes:
            c *= 1.5
            if c > 100:
                raise Exception
        else:
            break

    fontsize = (min(valid_sizes) + max(valid_sizes)) // 2
    font = fonts_by_size[fontsize]
    tx, ty = draw.textsize(text=text, font=font)

    xm, ym = [int(s/2) for s in testimg.size]
    x0, y0 = xm - tx/2, ym-ty/2
    draw.text((x0, y0), text, font=font, fill=fg_color)

    return testimg.rotate(angle, fillcolor=bg_color)


def glue_text(text, img_arr, xm, ym, xM, yM, angle, fonts_by_size):
    img_arr = np.array(img_arr)

    # Tedious arithmetic to have a small border around box we're working in
    dx, dy = xM-xm, yM-ym
    bdy_ratio = 0.005
    bdy_ratio_x, bdy_ratio_y = bdy_ratio * sqrt(dy/dx), bdy_ratio * sqrt(dx/dy)

    bx, by = max(int(dx*bdy_ratio_x), 1), max(int(dy*bdy_ratio_y), 1)
    xmb, ymb = [r - 2*b for r, b in [(xm, bx), (ym, by)]]
    xMb, yMb = [r + 2*b for r, b in [(xM, bx), (yM, by)]]

    xmb, xMb = np.clip([xmb, xMb], 1, img_arr.shape[1] - 1)
    ymb, yMb = np.clip([ymb, yMb], 1, img_arr.shape[0] - 1)

    # Make image mask which is = in the box, 0 outside
    img_mask = np.zeros((yMb - ymb, xMb - xmb), dtype=np.uint8)
    img_mask[by:-by, bx:-bx] = 1

    img_target = np.array(img_arr[ymb:yMb, xmb:xMb, :])  # array()-call to copy
    bg_color, fg_color = get_bg_fg_cols(img_target)

    img_source_img = text_to_img(text, img_target.shape[:2][::-1], fonts_by_size, bg_color, fg_color, angle)

    img_ret = image_blend(img_target, np.asarray(img_source_img), img_mask)

    # Put it back onto original img
    img_arr[ymb:yMb, xmb:xMb, :] = img_ret
    return img_arr


def tweak_box(edginess, box, r=0.1):
    """
    Given an array of 0., 1. indicating where edges are, tweak the box a bit to try to avoid them:

    :param edginess:
    :param box: [xmin, ymin, xmax, ymax]
    :param r:
    :return:
    """

    xmin, ymin, xmax, ymax = box

    dx, dy = int(xmax-xmin), int(ymax-ymin)
    dx, dy = int(r * sqrt(dy * dx)), int(r * sqrt(dy*dx))
    dx, dy = max(dx, 5), max(dy, 5)
    dxh, dyh = int(dx/3), int(dy/3)

    min_val = 1E20
    best_box = None
    D = max(2, int(np.sqrt(dx*dy)/10))
    for possible_box in itertools.product(
            range(xmin-dx, xmin+dxh + 1),
            range(ymin-dy, ymin+dyh + 1),
            range(xmax-dxh, xmax+dx+1),
            range(ymax-dxh, ymax+dy+1)
    ):
        xm, ym, xM, yM = possible_box
        E = np.sum(edginess[ym-D:yM+D+1, xm-D:xM+D+1]) - \
            np.sum(edginess[ym+D:yM-D+1, xm+D:xM-D+1])
        if E < min_val:
            min_val = E
            best_box = xm, ym, xM, yM

    xmin, ymin, xmax, ymax = best_box
    return xmin, ymin, xmax, ymax


def rel_to_abs(box_coords, image_size):
    """
    Convert from relative to absolute coordinates.

    :param box_coords: [xmin, ymin, xmax, ymax]
    :param image_size: [W, H]
    :return: [xmin, ymin, xmax, ymax] in absolute coordinates.
    """
    xmin, ymin, xmax, ymax = box_coords
    return tuple(map(int, [xmin*image_size[0], ymin*image_size[1], xmax*image_size[0], ymax*image_size[1]]))


def doodle_on(img, box_to_text_dict, fonts_by_size):
    """
    :param img: A PIL.Image object
    :param box_to_text_dict: A dictionary {(xmin, ymin, xmax, ymax): text_to_put}
    :param fonts_by_size: A dictionary {font_size: imagefont}
    :return:
    """

    MAX_SIZE = 1920 * 1080

    if img.size[0]*img.size[1] > MAX_SIZE:
        # Resize it down, do our thing, then resize back up
        s = math.sqrt(img.size[0]*img.size[1] / (MAX_SIZE))
        resized_img = img.resize(tuple(map(int, (img.size[0]/s, img.size[1]/s))))
        return doodle_on(resized_img, box_to_text_dict, fonts_by_size).resize(img.size)

    # Swap coordinates from relative to absolute
    box_to_text_dict = {
        rel_to_abs(box, img.size): text for box, text in box_to_text_dict.items()
    }

    # Carry out edge detection
    img_bw_arr = np.asarray(img.convert('L'))
    scale = np.sqrt(img_bw_arr.size)/900
    edginess = feature.canny(img_bw_arr, sigma=scale)
    edge_img = arr_to_img(255*edginess.astype(np.uint8))

    # Use edge detection to adjust boxes to hopefully snugly contain text
    tweaked_boxes = {tweak_box(edginess, box): text for box, text in box_to_text_dict.items()}

    img_arr = np.asarray(img)
    Ntheta = 360
    for box, text in tweaked_boxes.items():
        xmin, ymin, xmax, ymax = box
        sub_edge_img = edge_img.crop((xmin, ymin, xmax, ymax))
        angle = compute_angle(sub_edge_img, N_sample=Ntheta) + random.uniform(-180/Ntheta, 180/Ntheta)
        img_arr = glue_text(text, img_arr, xmin, ymin, xmax, ymax, -angle, fonts_by_size)

    return compress(arr_to_img(img_arr), q=80)
