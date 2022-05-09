import json
import logging
import subprocess
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


logger = logging.getLogger('synthetic')


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


class NoFontException(Exception):
    pass


def has_form(qpdf_page, operands):
    for operand in filter(_is_name_op, operands):
        x_object = qpdf_page.Resources.XObject.get(str(operand))
        if x_object is not None:
            for k, v in x_object.items():
                if _is_name_op(v) and k == '/Subtype' and str(v) == '/Form':
                    return True
    return False


def parse_text(qpdf_page: pikepdf.Page, font_map, synthesizer_class: Type[PdfSynthesizer], gt):
    content_stream = iter(pikepdf.parse_content_stream(qpdf_page))
    new_content_stream = []
    last_used_font = None
    synthesizer = synthesizer_class(gt, font_map)

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

    return new_content_stream, synthesizer.create_new_ground_truth()


def synthesize_pdf(pdf_file, json_file, dst_dir, synthesizer_class):
    ground_truth = json.loads(json_file.read_text())
    pdf_io = BytesIO(pdf_file.read_bytes())
    output_string = StringIO()
    rsrcmgr = PDFResourceManager(caching=True)
    device = TextConverter(rsrcmgr, output_string, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    with pikepdf.Pdf.open(pdf_file) as pdf:
        for page, miner in zip(pdf.pages, PDFPage.get_pages(pdf_io)):
            interpreter.process_page(miner)

            font_map = {}
            for font_name, font_val in page.resources.get('/Font', {}).items():
                if font_obj := interpreter.fontmap.get(font_name.lstrip('/')):
                    font_map[font_name] = Font(font_name, font_obj)

            if not font_map:
                raise NoFontException

            new_content_stream, new_ground_truth = parse_text(page, font_map, synthesizer_class, ground_truth)
            page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_content_stream))

        out_dst = dst_dir / pdf_file.name
        pdf.save(out_dst, qdf=True, compress_streams=False)

    out_json_dst = dst_dir / json_file.name
    out_json_dst.write_text(json.dumps(new_ground_truth, indent=2))


def parse_pdf(
    name: str,
    pdf_file: Path,
    json_file: Path,
    synthesizer_class: Type[PdfSynthesizer],
    dst_dir: Path,
    tmp_dir: Path
):
    logger.info(name, pdf_file, json_file)
    status = f'Error when synthesizing {name}'

    try:
        synthesize_pdf(pdf_file, json_file, dst_dir, synthesizer_class)
        status = f'Successfully synthesized {name}'
    except HasFormException:
        logger.info('Has form! Trying to flatten PDF')
        flattened_pdf_file = flatten(pdf_file, tmp_dir)
        synthesize_pdf(flattened_pdf_file, json_file, dst_dir, synthesizer_class)
        flattened_pdf_file.unlink()
        status = f'Successfully synthesized {name}'
    except NoFontException:
        logger.warning('PDF does not have any fonts! Skipping')
    except FileNotFoundError as e:
        logger.error(e)

    return status


def flatten(pdf_file, tmp_dir):
    dst_path = tmp_dir / f'{pdf_file.stem}.flattened.pdf'
    subprocess.run(f'gs -q -sDEVICE=pdfwrite -o {dst_path} {pdf_file}', shell=True)
    return dst_path