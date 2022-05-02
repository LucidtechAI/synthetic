import random
import PIL.Image
import yaml
import pathlib
import re

from synthetic.image import doodle_on, get_fonts


def synthesize_image(args):
    template_path = pathlib.Path(args.template)

    font_sizes = list(range(*map(int, args.font_size_range.split(':'))))
    fonts = get_fonts(args.fonts_dir, font_sizes)
    fonts_by_size = random.choice(list(fonts.values()))

    with template_path.open('r') as f:
        template = yaml.load(f)
        img_path = pathlib.Path(template['path']).relative_to(template_path.parent)
        img = PIL.Image.open(str(template_path.parent / img_path))

    box_to_text_dict = {}
    for label in template['labels']:
        for box in template['labels'][label]['boxes']:
            box = tuple([box[k] for k in ['xmin', 'ymin', 'xmax', 'ymax']])
            box_to_text_dict[box] = template['labels'][label]['text']

    ret_img = doodle_on(img, box_to_text_dict, fonts_by_size)
    ret_img.save(template['dest'])


def synthesize_pdf(args):
    template_path = pathlib.Path(args.template)

    with template_path.open('r') as f:
        template = yaml.load(f)
        pdf_path = pathlib.Path(template['path']).relative_to(template_path.parent)

    with pdf_path.open('rb') as f:
        pdf = f.read()

        for label in template['labels']:
            val = template['labels'][label]
            pdf = re.sub(rf'__\(-*{label}\)__'.encode('utf-8'), val.encode('utf-8'), pdf)

    with pathlib.Path(template['dest']).open('wb') as f:
        f.write(pdf)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    image_parser = subparsers.add_parser('image')
    image_parser.add_argument('template')
    image_parser.add_argument('fonts_dir')
    image_parser.add_argument('--font-size-range', type=str, default='20:40')
    image_parser.set_defaults(func=synthesize_image)

    pdf_parser = subparsers.add_parser('pdf')
    pdf_parser.add_argument('template')
    pdf_parser.set_defaults(func=synthesize_pdf)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
