import json
import logging
from functools import partial
from pathlib import Path
from typing import Type

from .doodle import doodle_on
from .synthesizer import ImageSynthesizer

from PIL import ImageFont, Image


logger = logging.getLogger(__name__)


class AlreadyProcessed(Exception):
    pass


def synthesize_image(
    image_file,
    json_file,
    dst_dir,
    num_outputs_per_document,
    synthesizer_class,
    font_path,
    font_size,
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
    font = ImageFont.truetype(str(font_path), font_size)
    image = Image.open(image_file)
    synthesizer = synthesizer_class(ground_truth)

    for i in k_to_process:
        box_to_text_dict = {}
        new_ground_truth = []
        for gt in ground_truth:
            if 'bbox' not in gt:
                continue
            bbox = tuple(gt['bbox'])
            new_value = synthesizer.modify_text(gt['value'])
            box_to_text_dict[bbox] = new_value
            new_ground_truth.append({**gt, 'value': new_value})

        if not new_ground_truth:
            continue

        for box, text in box_to_text_dict.items():
            print(f"Writing '{text}' at {box}")

        output_image = doodle_on(image, box_to_text_dict, {font_size: font})
        output_image.save(_out_path(i, '.jpeg'))
        _out_path(i, '.json').write_text(json.dumps(new_ground_truth, indent=2))


def parse_image(
    name: str,
    image_file: Path,
    json_file: Path,
    synthesizer_class: Type[ImageSynthesizer],
    font_path: Path,
    font_size: int,
    num_outputs_per_document: int,
    dst_dir: Path,
    tmp_dir: Path,
):
    logger.info(f'{name}: {image_file} {json_file}')
    status = f'Error when synthesizing {name}'
    synthesize_fn = partial(
        synthesize_image,
        font_path=font_path,
        font_size=font_size,
        json_file=json_file,
        dst_dir=dst_dir,
        num_outputs_per_document=num_outputs_per_document,
        synthesizer_class=synthesizer_class,
    )

    try:
        synthesize_fn(image_file)
        status = f'Successfully synthesized {name}'
    except Exception as e:
        logger.exception(e)
        link = 'https://github.com/LucidtechAI/synthetic/issues'
        logger.error(f'This might be a bug, please open an issue here {link}')

    return status
