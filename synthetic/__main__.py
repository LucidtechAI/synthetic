#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import logging
import pathlib
import textwrap
from importlib import import_module
from functools import partial

import argcomplete
from filetype.types.archive import Pdf
from filetype.types.image import Jpeg

from .__version__ import __version__
from .image.parser import parse_image
from .image.synthesizer import BasicSynthesizer as BasicImageSynthesizer
from .iterdata import parse_documents
from .pdf.parser import parse_pdf
from .pdf.synthesizer import BasicSynthesizer as BasicPdfSynthesizer


def load_class(synthesizer_class):
    module_name, class_name = synthesizer_class.rsplit('.', maxsplit=1)
    lib = import_module(module_name)
    return getattr(lib, class_name)


def add_common_args(parser):
    parser.add_argument('src_dir', type=pathlib.Path)
    parser.add_argument('dst_dir', type=pathlib.Path)
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--num-documents', '-n', type=int)
    parser.add_argument('--num-outputs-per-document', '-k', type=int, default=1)
    parser.add_argument('--num-processes', type=int)


def create_pdf_parser(subparsers):
    pdf_parser = subparsers.add_parser('pdf')
    add_common_args(pdf_parser)
    pdf_parser.add_argument('--max-fonts', type=int)
    pdf_parser.add_argument('--max-pages', type=int)
    pdf_parser.add_argument('--synthesizer-class', type=load_class, default=BasicPdfSynthesizer)
    cmd = partial(parse_documents, accepted_document_types=[Pdf], parse_fn=parse_pdf)
    pdf_parser.set_defaults(optionals=['max_fonts', 'max_pages'])
    pdf_parser.set_defaults(cmd=cmd)


def create_image_parser(subparsers):
    image_parser = subparsers.add_parser('image')
    add_common_args(image_parser)
    image_parser.add_argument('--font-path', type=pathlib.Path)
    image_parser.add_argument('--font-size', type=int)
    image_parser.add_argument('--synthesizer-class', type=load_class, default=BasicImageSynthesizer)
    cmd = partial(parse_documents, accepted_document_types=[Jpeg], parse_fn=parse_image)
    image_parser.set_defaults(optionals=['font_path', 'font_size'])
    image_parser.set_defaults(cmd=cmd)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=textwrap.dedent('''
            PDF anonymizer/synthesizer for Cradl, see --help for more info. To use tab completion make sure you 
            have global completion activated. See argcomplete docs for more information: 
            https://kislyuk.github.io/argcomplete/
        '''),
    )
    subparsers = parser.add_subparsers()
    create_pdf_parser(subparsers)
    create_image_parser(subparsers)
    argcomplete.autocomplete(parser)
    return parser


def set_verbosity(verbose):
    verbosity_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    verbosity = verbosity_levels[min(verbose, len(verbosity_levels) - 1)]
    logging.basicConfig()
    logging.getLogger().setLevel(verbosity)
    logging.getLogger('synthetic').setLevel(verbosity)


def main():
    parser = create_parser()
    args = vars(parser.parse_args())
    set_verbosity(args.pop('verbose'))

    try:
        cmd = args.pop('cmd')
        optionals = args.pop('optionals')
    except KeyError:
        parser.print_help()
        exit(1)

    kwargs = {k: v for k, v in args.items() if (v and k not in optionals)}
    options = {k: v for k, v in args.items() if (v and k in optionals)}
    cmd(**kwargs, options=options)


if __name__ == '__main__':
    main()
