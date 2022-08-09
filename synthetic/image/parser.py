import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import Type

from .poisson_blending import blend_image
from .synthesizer import ImageSynthesizer

from PIL import ImageFont, Image


logger = logging.getLogger(__name__)


class AlreadyProcessed(Exception):
    pass


class NoBoundingBoxes(Exception):
    pass


def synthesize_image(
    image_file,
    dst_dir,
    font_path,
    font_size_range,
    jitter,
    json_file,
    max_size,
    num_outputs_per_document,
    synthesizer_class,
):
    def _out_path(_i, suffix):
        return dst_dir / f'{json_file.stem}-{_i}{suffix}'

    k_to_process = []
    for i in range(num_outputs_per_document):
        if not (_out_path(i, '.jpeg').exists() and _out_path(i, '.json').exists()):
            k_to_process.append(i)

    if not k_to_process:
        raise AlreadyProcessed(f'Already processed {image_file} {json_file}')

    ground_truth = json.loads(json_file.read_text())
    from_size, to_size = font_size_range
    fonts = {i: ImageFont.truetype(str(font_path), i) for i in range(to_size, from_size, -1)}
    image = Image.open(image_file)
    synthesizer = synthesizer_class(ground_truth)
    bounding_box_key = 'bbox'

    if not any(gt.get(bounding_box_key) for gt in ground_truth):
        raise NoBoundingBoxes(
            f'No bounding boxes found in {json_file}. Add "bbox": [xmin, ymin, xmax, ymax] to each label in the ground'
            ' truth file. Values for xmin, ymin, xmax, ymax should be normalized in range 0-1 and (0, 0) is the top '
            'left corner of the image.'
        )

    for i in k_to_process:
        bounding_boxes = []
        new_ground_truth = []
        for gt in ground_truth:
            label = gt['label']
            if bounding_box_key not in gt:
                logger.warning(f'No bounding box found for label {label} in {json_file}')
                continue

            bounding_box = tuple(gt[bounding_box_key])
            if jitter:
                bounding_box = (
                    max(bounding_box[0] - random.random() * jitter, 0),
                    max(bounding_box[1] - random.random() * jitter, 0),
                    min(bounding_box[2] + random.random() * jitter, 1),
                    min(bounding_box[3] + random.random() * jitter, 1),
                )
            new_value = synthesizer.modify_text(gt['value'])
            bounding_boxes.append((bounding_box, new_value))
            new_ground_truth.append({'label': label, 'value': new_value})

        output_image = blend_image(image, bounding_boxes, fonts, max_size)
        output_image.save(_out_path(i, '.jpeg'))
        _out_path(i, '.json').write_text(json.dumps(new_ground_truth, indent=2))


def parse_image(
    name: str,
    image_file: Path,
    json_file: Path,
    synthesizer_class: Type[ImageSynthesizer],
    font_path: Path,
    font_size_range: int,
    num_outputs_per_document: int,
    dst_dir: Path,
    tmp_dir: Path,
    jitter: float = None,
    max_size: int = 720 * 1080,
):
    logger.info(f'{name}: {image_file} {json_file}')
    status = f'Error when synthesizing {name}'
    synthesize_fn = partial(
        synthesize_image,
        dst_dir=dst_dir,
        font_path=font_path,
        font_size_range=font_size_range,
        jitter=jitter,
        json_file=json_file,
        max_size=max_size,
        num_outputs_per_document=num_outputs_per_document,
        synthesizer_class=synthesizer_class,
    )

    try:
        synthesize_fn(image_file)
        status = f'Successfully synthesized {name}'
    except AlreadyProcessed as e:
        logger.warning(e)
    except NoBoundingBoxes as e:
        logger.error(e)
    except Exception as e:
        logger.exception(e)
        link = 'https://github.com/LucidtechAI/synthetic/issues'
        logger.error(f'This might be a bug, please open an issue here {link}')

    return status
