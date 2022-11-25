import itertools
import json
import logging
import multiprocessing
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Callable, Type, List

import filetype
from filetype.types import Type as FileType

from .core.synthesizer import Synthesizer


logger = logging.getLogger(__name__)


def is_json(path):
    try:
        json.loads(path.read_text())
        return True
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False


def iter_documents(src_dir, accepted_document_types):
    grouped_paths = defaultdict(list)

    for path in src_dir.iterdir():
        if path.is_file():
            grouped_paths[path.stem].append(path)

    for name, paths in grouped_paths.items():
        document_path = None
        ground_truth_path = None

        for path in paths:
            kind = filetype.guess(path)
            if not kind and is_json(path):
                if ground_truth_path:
                    logger.warning(f'Ground truth file for {name} already found (Old: {ground_truth_path} New: {path})')
                ground_truth_path = path
            elif not kind:
                continue
            elif any(isinstance(kind, document_type) for document_type in accepted_document_types):
                if document_path:
                    logger.warning(f'Document file for {name} already found (Old: {document_path} New: {path})')
                document_path = path

        if not document_path:
            logger.warning(f'Missing document file for {name}')
        if not ground_truth_path:
            logger.warning(f'Missing ground truth file for {name}')
        if document_path and ground_truth_path:
            yield name, document_path, ground_truth_path


def parse_documents(
    src_dir: Path,
    dst_dir: Path,
    accepted_document_types: List[FileType],
    synthesizer_class: Type[Synthesizer],
    parse_fn: Callable[[str, Path, Path, Type[Synthesizer], Path, Path], str],
    num_outputs_per_document: int,
    num_processes: int = max(1, multiprocessing.cpu_count() - 1),
    num_documents: int = None,
    options: dict = None,
):
    dst_dir.mkdir(exist_ok=True)

    if num_documents:
        documents = itertools.islice(iter_documents(src_dir, accepted_document_types), num_documents)
    else:
        documents = iter_documents(src_dir, accepted_document_types)

    with tempfile.TemporaryDirectory() as tmp_dir, ProcessPoolExecutor(max_workers=num_processes) as executor:
        _parse_fn = partial(
            parse_fn,
            synthesizer_class=synthesizer_class,
            num_outputs_per_document=num_outputs_per_document,
            dst_dir=dst_dir,
            tmp_dir=Path(tmp_dir),
            **(options or {}),
        )
        futures = []

        for args in documents:
            futures.append(executor.submit(_parse_fn, *args))

        for future in as_completed(futures):
            logger.info(future.result())
