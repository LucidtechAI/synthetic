import collections
import json
import logging
import re
import string
import subprocess
from functools import partial
from io import BytesIO, StringIO
from pathlib import Path
from typing import Type
from uuid import uuid4

import pikepdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage

from .synthesizer import PdfSynthesizer
from .utils import Font


logger = logging.getLogger(__name__)


class TextBlock:
    ERRORS = 'ignore'

    def __init__(self):
        self.original_content_stream = []
        self.text_objects = {}
        self.text_objects_and_ids = []

    def __iter__(self):
        for text_id, props in self.text_objects.items():
            font = props['font']
            yield text_id, self._decode(props['text'], font), font

    def add_text_object(self, text_object: pikepdf.String, font_object: Font):
        text_id = uuid4().hex
        self.text_objects_and_ids.append((text_object, text_id))
        self.text_objects[text_id] = {
            'font': font_object,
            'text': text_object,
        }

    def set_unicode_text(self, text_id: str, text: str):
        if text_id in self.text_objects:
            font = self.text_objects[text_id].get('font')
            self.text_objects[text_id]['text'] = self._encode(text, font)

    @property
    def content_stream(self):
        new_content_stream = []
        for operands, operator in self.original_content_stream:
            new_operands = operands[:]

            if operator == pikepdf.Operator('Tj'):
                new_operands = self._get_objects(operands)
            elif operator == pikepdf.Operator('TJ'):
                new_operands = []
                for tja in operands:
                    if _is_array_op(tja):
                        objects = self._get_objects(tja)
                        new_operands.append(pikepdf.Array(objects))
                    else:
                        new_operands.append(tja)

            new_content_stream.append((new_operands, operator))
        return new_content_stream

    def _get_objects(self, operands):
        objects = []
        for tj in operands:
            text_id = self._get_id(tj)
            if text_id in self.text_objects:
                objects.append(self.text_objects[text_id]['text'])
            else:
                objects.append(tj)
        return objects

    def _get_id(self, tj: pikepdf.String):
        for text_object, text_id in self.text_objects_and_ids:
            if tj == text_object:
                return text_id

    def _encode(self, s: str, font: Font = None) -> pikepdf.String:
        return pikepdf.String(font.encode(s) if font else s.encode(errors=self.ERRORS))

    def _decode(self, tj: pikepdf.String, font: Font = None) -> str:
        return font.decode(bytes(tj)) if font else bytes(tj).decode(errors=self.ERRORS)


def _is_string_op(op):
    return isinstance(op, pikepdf.String)


def _is_array_op(op):
    return isinstance(op, pikepdf.Array)


def _is_name_op(op):
    return isinstance(op, pikepdf.Name)


def _parse_font(ops, font_map):
    tf, *_ = ops  # Assuming first element of operands list is the font object
    return font_map.get(str(tf))


def _parse_text_block(font_map, start, content_stream, current_font):
    text_block = TextBlock()
    operands, operator = start
    text_block.original_content_stream.append(start)

    while operator != pikepdf.Operator('ET'):
        if operator == pikepdf.Operator('Tf'):
            current_font = _parse_font(operands, font_map)
        elif operator == pikepdf.Operator('Tj'):
            for tj in filter(_is_string_op, operands):
                text_block.add_text_object(tj, current_font)
        elif operator == pikepdf.Operator('TJ'):
            for tja in filter(_is_array_op, operands):
                for tj in filter(_is_string_op, tja):
                    text_block.add_text_object(tj, current_font)
        operands, operator = next(content_stream)
        text_block.original_content_stream.append((operands, operator))

    return text_block, current_font


class HasFormException(Exception):
    pass


class NoTextException(Exception):
    pass


class TooManyPagesException(Exception):
    pass


def has_form(qpdf_page, operands):
    for operand in filter(_is_name_op, operands):
        x_object = qpdf_page.Resources.XObject.get(str(operand))
        if x_object is not None:
            for k, v in x_object.items():
                if _is_name_op(v) and k == '/Subtype' and str(v) == '/Form':
                    return True
    return False


def parse_text(qpdf_page: pikepdf.Page, font_map, synthesizer: PdfSynthesizer):
    content_stream = iter(pikepdf.parse_content_stream(qpdf_page))
    new_content_stream = []
    last_used_font = None

    for operands, operator in content_stream:
        if operator == pikepdf.Operator('Do'):
            if has_form(qpdf_page, operands):
                raise HasFormException

        if operator == pikepdf.Operator('Tf'):
            last_used_font = _parse_font(operands, font_map)

        if operator == pikepdf.Operator('BT'):
            text_block, last_used_font = _parse_text_block(
                font_map=font_map,
                start=(operands, operator),
                content_stream=content_stream,
                current_font=last_used_font,
            )
            for text_id, text, font in text_block:
                modified_text = synthesizer.modify_text(text, font=font)
                text_block.set_unicode_text(text_id, modified_text)
            new_content_stream.extend(text_block.content_stream)
        else:
            new_content_stream.append((operands, operator))

    return new_content_stream


def synthesize_pdf(pdf_file, json_file, dst_dir, max_pages, num_outputs_per_document, synthesizer_class):
    ground_truth = json.loads(json_file.read_text())
    pdf_io = BytesIO(pdf_file.read_bytes())
    output_string = StringIO()
    rsrcmgr = PDFResourceManager(caching=True)
    device = TextConverter(rsrcmgr, output_string, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    interpreter_fonts = {}

    with pikepdf.Pdf.open(pdf_file) as pdf:
        if max_pages and len(pdf.pages) > max_pages:
            raise TooManyPagesException(f'Too many pages {len(pdf.pages)} > {max_pages} in PDF, skipping!')

        for page_number, (page, miner) in enumerate(zip(pdf.pages, PDFPage.get_pages(pdf_io))):
            interpreter.process_page(miner)
            interpreter_fonts.update(interpreter.fontmap)

    if not re.sub(f'[{re.escape(string.whitespace)}]', '', output_string.getvalue()):
        raise NoTextException('PDF does not have any text! Skipping')

    font_map = {f'/{k}': Font(f'/{k}', v) for k, v in interpreter_fonts.items()}
    synthesizer = synthesizer_class(ground_truth, font_map)

    with pikepdf.Pdf.open(pdf_file) as pdf:
        new_contents = collections.defaultdict(list)
        new_ground_truths = []

        for i in range(num_outputs_per_document):
            for page_number, page in enumerate(pdf.pages):
                new_content_stream = parse_text(page, font_map, synthesizer)
                new_contents[i].append(pdf.make_stream(pikepdf.unparse_content_stream(new_content_stream)))

            new_ground_truths.append(synthesizer.create_new_ground_truth())
            synthesizer.reset()

        for i in range(num_outputs_per_document):
            for page_number, page in enumerate(pdf.pages):
                page.Contents = new_contents[i][page_number]

            name = json_file.stem
            out_dst = dst_dir / f'{name}-{i}.pdf'
            pdf.save(out_dst)
            out_json_dst = dst_dir / f'{name}-{i}.json'
            out_json_dst.write_text(json.dumps(new_ground_truths[i], indent=2))


def parse_pdf(
    name: str,
    pdf_file: Path,
    json_file: Path,
    synthesizer_class: Type[PdfSynthesizer],
    max_pages: int,
    num_outputs_per_document: int,
    dst_dir: Path,
    tmp_dir: Path
):
    logger.info(f'{name}: {pdf_file} {json_file}')
    status = f'Error when synthesizing {name}'
    synthesize_fn = partial(
        synthesize_pdf,
        json_file=json_file,
        dst_dir=dst_dir,
        max_pages=max_pages,
        num_outputs_per_document=num_outputs_per_document,
        synthesizer_class=synthesizer_class,
    )

    try:
        synthesize_fn(pdf_file)
        status = f'Successfully synthesized {name}'
    except HasFormException:
        logger.info('Has form! Trying to flatten PDF')
        if flattened_pdf_file := flatten(pdf_file, tmp_dir):
            synthesize_fn(flattened_pdf_file)
            flattened_pdf_file.unlink()
            status = f'Successfully synthesized {name}'
    except (NoTextException, TooManyPagesException) as e:
        logger.error(e.message)
    except FileNotFoundError as e:
        logger.error(e)

    return status


def flatten(pdf_file, tmp_dir):
    try:
        dst_path = tmp_dir / f'{pdf_file.stem}.flattened.pdf'
        subprocess.run(f'gs -q -sDEVICE=pdfwrite -o {dst_path} {pdf_file}', shell=True, timeout=15)
        return dst_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(e)
