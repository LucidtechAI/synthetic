import collections
import json
import logging
import re
import string
import subprocess
import sys
from functools import partial
from io import BytesIO, StringIO
from os import getpid
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


DEFAULT_TIMEOUT = 30
logger = logging.getLogger(__name__)

if sys.platform != 'win32':
    import signal

    def handler(*_):
        raise TimeoutError()

    def timeout(func, args=None, kwargs=None, timeout_in_seconds=1):
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_in_seconds)
        try:
            result = func(*(args or ()), **(kwargs or {}))
        except TimeoutError:
            raise
        finally:
            signal.alarm(0)

        return result
else:
    def timeout(func, args=None, kwargs=None, timeout_in_seconds=1):
        return func(*(args or ()), **(kwargs or {}))


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


class TooManyFontsException(Exception):
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


def _parse_pdf_objects(qpdf_page: pikepdf.Page, font_map, new_content_stream):
    content_stream = iter(pikepdf.parse_content_stream(qpdf_page))
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

            yield text_block
            new_content_stream.extend(text_block.content_stream)
        else:
            new_content_stream.append((operands, operator))

    return new_content_stream


def update_available_characters(qpdf_page: pikepdf.Page, font_map):
    for text_block in _parse_pdf_objects(qpdf_page, font_map, []):
        for text_id, text, font in text_block:
            font.available_characters |= set(text)


def split(s, template):
    pos = 0
    for c in template:
        yield s[pos:pos + len(c)]
        pos += len(c)


def parse_text(qpdf_page: pikepdf.Page, font_map, synthesizer: PdfSynthesizer):
    new_content_stream = []

    for text_block in _parse_pdf_objects(qpdf_page, font_map, new_content_stream):
        texts = []
        text_ids = []
        for text_id, text, font in text_block:
            texts.append(text)
            text_ids.append(text_id)

        modified_text = synthesizer.modify_text(''.join(texts))

        for (text_id, text, font), new_text in zip(text_block, split(modified_text, texts)):
            text_block.set_unicode_text(text_id, new_text)

    return new_content_stream


def out_path(_i, suffix, dst_dir, file_stem):
    return dst_dir / f'{file_stem}-{_i}{suffix}'


def calculate_k_to_process(num_outputs_per_document, dst_dir, file_stem):
    k_to_process = []
    for i in range(num_outputs_per_document):
        if not (
            out_path(i, '.pdf', dst_dir, file_stem).exists() and
            out_path(i, '.json', dst_dir, file_stem).exists()
        ):
            k_to_process.append(i)
    return k_to_process


def synthesize_pdf(
    pdf_file,
    json_file,
    dst_dir,
    max_fonts,
    max_pages,
    synthesizer_class,
    k_to_process,
):
    ground_truth = json.loads(json_file.read_text())
    pdf_io = BytesIO(pdf_file.read_bytes())
    output_string = StringIO()
    rsrcmgr = PDFResourceManager(caching=True)
    device = TextConverter(rsrcmgr, output_string, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    interpreter_fonts = {}

    _out_path = partial(out_path, dst_dir=dst_dir, file_stem=json_file.stem)

    with pikepdf.Pdf.open(pdf_file) as pdf:
        if max_pages and len(pdf.pages) > max_pages:
            raise TooManyPagesException(f'Too many pages {len(pdf.pages)} > {max_pages} in PDF, skipping!')

        for page_number, (page, miner) in enumerate(zip(pdf.pages, PDFPage.get_pages(pdf_io))):
            interpreter.process_page(miner)
            interpreter_fonts.update(interpreter.fontmap)

    if max_fonts and len(interpreter_fonts) > max_fonts:
        raise TooManyFontsException(f'Too many fonts {len(interpreter_fonts)} > {max_fonts} in PDF, skipping!')

    if not re.sub(f'[{re.escape(string.whitespace)}]', '', output_string.getvalue()):
        raise NoTextException('PDF does not have any text! Skipping')

    font_map = {f'/{k}': Font(f'/{k}', v) for k, v in interpreter_fonts.items()}

    with pikepdf.Pdf.open(pdf_file) as pdf:
        new_contents = collections.defaultdict(list)
        new_ground_truths = {}

        for page in pdf.pages:
            update_available_characters(page, font_map)

        synthesizer = synthesizer_class(ground_truth, font_map)

        for i in k_to_process:
            for page_number, page in enumerate(pdf.pages):
                new_content_stream = parse_text(page, font_map, synthesizer)
                new_contents[i].append(pdf.make_stream(pikepdf.unparse_content_stream(new_content_stream)))

            new_ground_truths[i] = synthesizer.create_new_ground_truth()
            synthesizer.reset()

        for i in k_to_process:
            for page_number, page in enumerate(pdf.pages):
                page.Contents = new_contents[i][page_number]

            pdf.save(_out_path(i, '.pdf'))
            _out_path(i, '.json').write_text(json.dumps(new_ground_truths[i], indent=2))


def parse_pdf(
    name: str,
    pdf_file: Path,
    json_file: Path,
    synthesizer_class: Type[PdfSynthesizer],
    num_outputs_per_document: int,
    dst_dir: Path,
    tmp_dir: Path,
    max_fonts: int = None,
    max_pages: int = None,
    timeout_in_seconds: int = DEFAULT_TIMEOUT,
    autofix_pdf: bool = False,
):
    logger.info(f'{name}: {pdf_file} {json_file}')
    status = f'Error when synthesizing {name}'

    k_to_process = calculate_k_to_process(num_outputs_per_document, dst_dir, json_file.stem)
    if not k_to_process:
        return f'Already processed {pdf_file} {json_file}'

    synthesize_fn = partial(
        synthesize_pdf,
        json_file=json_file,
        dst_dir=dst_dir,
        max_fonts=max_fonts,
        max_pages=max_pages,
        synthesizer_class=synthesizer_class,
        k_to_process=k_to_process,
    )

    try:
        if autofix_pdf and (rewritten_pdf_file := rewrite_pdf(pdf_file, tmp_dir, timeout_in_seconds)):
            timeout(synthesize_fn, args=(rewritten_pdf_file,), timeout_in_seconds=timeout_in_seconds)
        else:
            timeout(synthesize_fn, args=(pdf_file,), timeout_in_seconds=timeout_in_seconds)
        status = f'Successfully synthesized {name}'
    except HasFormException:
        logger.info('Has form! Trying to flatten PDF')
        if flattened_pdf_file := flatten(pdf_file, tmp_dir, timeout_in_seconds):
            try:
                timeout(synthesize_fn, args=(flattened_pdf_file,), timeout_in_seconds=timeout_in_seconds)
                status = f'Successfully synthesized {name}'
            except HasFormException:
                logger.error('Failed to get rid of forms in flattened PDF')
            except Exception as e:
                logger.exception(e)
                logger.error(f'Error when synthesizing {name}')
            finally:
                flattened_pdf_file.unlink(missing_ok=True)
    except (FileNotFoundError, NoTextException, TooManyFontsException, TooManyPagesException) as e:
        logger.error(e)
    except TimeoutError:
        logger.error(f'Synthesizing timed out, took longer than {timeout_in_seconds}s')
    except Exception as e:
        logger.exception(e)
        link = 'https://github.com/LucidtechAI/synthetic/issues'
        logger.error(f'This might be a bug, please open an issue here {link}')

    return status


def run_in_shell(cmd, timeout_in_seconds):
    return subprocess.run(cmd, shell=True, timeout=timeout_in_seconds)


def flatten(pdf_file, tmp_dir, timeout_in_seconds):
    try:
        dst_path = tmp_dir / f'{pdf_file.stem}.flattened.pdf'
        run_in_shell(f'gs -q -sDEVICE=pdfwrite -o {dst_path} {pdf_file}', timeout_in_seconds)
        return dst_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(e)


def rewrite_pdf(pdf_file, tmp_dir, timeout_in_seconds):
    try:
        dst = tmp_dir / f'{pdf_file.stem}.pdf'
        env = f'-env:UserInstallation=file:///tmp/{getpid()}'
        src = f'--convert-to pdf {pdf_file}'
        run_in_shell(f'libreoffice {env} --headless {src} --outdir {tmp_dir}', timeout_in_seconds)
        return dst
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(e)
