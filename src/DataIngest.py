#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert all PDF scientific papers in a directory into a JSON file of text
chunks that follow a minimal ingestion → retrieval contract.

Contract per chunk
------------------
- chunk_id : str  (e.g., "paper_0001_chunk_0000")
- doc_id   : str  (derived from each PDF filename)
- order    : int  (0-based index of the chunk within that document)
- text     : str  (chunk content)

Chunking strategy
-----------------
- Text is cleaned to:
  * normalize common Unicode ligatures (e.g., "ﬀ" → "ff"),
  * join words split across lines with a hyphen
    (e.g., "for-\\n mulation" → "formulation"),
  * collapse multiple whitespace characters into a single space.
- Text is then split into word-based chunks with overlap:
  * default words_per_chunk = 100,
  * default overlap = 50 (50% overlap).

Usage
-----
python pdf_to_chunks.py
python pdf_to_chunks.py --input-dir data --output-dir outputs
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from loguru import logger
from PyPDF2 import PdfReader

from config import config as cfg


def configure_logging(log_level: str) -> None:
    """
    Configure basic logging with loguru.

    This function removes existing handlers and adds a simple stderr
    sink with timestamp, level, and message.

    Parameters
    ----------
    log_level : str
        Logging level to use (e.g., "INFO", "DEBUG").
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{message}</cyan>"
        ),
        level=log_level.upper(),
    )


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract plain text from all pages of a PDF file.

    Parameters
    ----------
    pdf_path : Path
        Path to the input PDF file.

    Returns
    -------
    str
        Concatenated text from all pages.
    """
    reader = PdfReader(str(pdf_path))
    pages_text: List[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    full_text = "\n\n".join(pages_text)
    return full_text


def clean_extracted_text(text: str) -> str:
    """
    Clean raw text extracted from a PDF.

    This function performs light, conservative cleaning so that content
    is not over-merged:

    - Normalize common Unicode ligatures (e.g., "ﬀ" → "ff").
    - Join words that were split with a hyphen at the end of a line
      (e.g., "for-\\n mulation" → "formulation").
    - Collapse multiple whitespace characters into a single space.

    Parameters
    ----------
    text : str
        Raw text as extracted from the PDF.

    Returns
    -------
    str
        Cleaned text.
    """
    ligatures = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "ft",
        "\ufb06": "st",
    }
    for bad, good in ligatures.items():
        text = text.replace(bad, good)

    # Join hyphenated words across line breaks.
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Collapse multiple whitespace characters into a single space.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text_with_overlap(
    text: str,
    words_per_chunk: int = 100,
    overlap: int = 50,
) -> List[str]:
    """
    Split a long text into consecutive word-based chunks with overlap.

    Parameters
    ----------
    text : str
        Full document text to be chunked.
    words_per_chunk : int, optional
        Target number of words per chunk, by default 100.
    overlap : int, optional
        Number of words shared between consecutive chunks, by
        default 50 (i.e., 50% overlap for 100-word chunks).

    Returns
    -------
    List[str]
        List of chunk texts. Each chunk has up to `words_per_chunk`
        words, and consecutive chunks share `overlap` words.
    """
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be positive.")

    if not (0 <= overlap < words_per_chunk):
        raise ValueError("overlap must be in [0, words_per_chunk).")

    words = text.split()
    chunks: List[str] = []
    stride = words_per_chunk - overlap

    for start in range(0, len(words), stride):
        end = start + words_per_chunk
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def build_chunk_records(doc_id: str, chunks: List[str]) -> List[Dict[str, object]]:
    """
    Build a list of chunk records that follow the minimal contract.

    Parameters
    ----------
    doc_id : str
        Identifier for the document, usually derived from the filename.
    chunks : List[str]
        List of chunk texts.

    Returns
    -------
    List[Dict[str, object]]
        List of dictionaries, each representing one chunk:
        {
            "chunk_id": str,
            "doc_id": str,
            "order": int,
            "text": str,
        }
    """
    records: List[Dict[str, object]] = []

    for idx, text in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{idx:04d}"
        record: Dict[str, object] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "order": idx,
            "text": text,
        }
        records.append(record)

    return records


def run_pdf_chunker(
    input_dir: Path,
    output_dir: Path,
    base_output_name: str,
    words_per_chunk: int,
    overlap: int,
    log_level: str,
) -> None:
    """
    Run the end-to-end pipeline to convert PDFs into chunk records.

    This function:
    - enumerates and validates PDF files in the input directory,
    - extracts and cleans text from each file,
    - chunks the text with overlap,
    - writes a JSON file with all chunk records, and
    - writes a separate metadata JSON file describing the run
      (parameters, timestamps, and counts).

    Parameters
    ----------
    input_dir : Path
        Directory containing PDF files.
    output_dir : Path
        Directory where the output JSON files will be saved.
    base_output_name : str
        Base name for output files; a timestamp will be appended.
    words_per_chunk : int
        Number of words per chunk.
    overlap : int
        Number of overlapping words between consecutive chunks.
    log_level : str
        Logging level used during the run.
    """
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Ensure that the output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDF files to be processed.
    pdf_files = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() == ".pdf"
    )
    num_pdfs = len(pdf_files)

    # Timestamp used both in metadata and in the output filenames.
    run_datetime = datetime.now()
    timestamp_label = run_datetime.strftime("%Y%m%d_%H%M%S")

    base_name = f"{base_output_name}_{timestamp_label}"
    output_path = output_dir / f"{base_name}.json"
    metadata_path = output_dir / f"{base_name}_metadata.json"

    logger.info(
        (
            "Starting PDF to chunks conversion | input_dir='{}' | "
            "output_dir='{}' | output_file='{}' | words_per_chunk={} | "
            "overlap={} | log_level='{}' | n_pdfs={}"
        ),
        input_dir,
        output_dir,
        output_path.name,
        words_per_chunk,
        overlap,
        log_level.upper(),
        num_pdfs,
    )

    all_records: List[Dict[str, object]] = []
    total_chunks = 0

    if num_pdfs == 0:
        logger.warning(
            "No PDF files found in directory '{}'. An empty output will be written.",
            input_dir,
        )
    else:
        for index, pdf_path in enumerate(pdf_files, start=1):
            logger.info(
                "Processing file {}/{}: {}",
                index,
                num_pdfs,
                pdf_path.name,
            )

            doc_id = pdf_path.stem
            raw_text = extract_text_from_pdf(pdf_path)
            full_text = clean_extracted_text(raw_text)
            chunks = chunk_text_with_overlap(
                full_text,
                words_per_chunk=words_per_chunk,
                overlap=overlap,
            )

            logger.info(
                "Document '{}' cleaned length: {} characters, produced {} chunks.",
                doc_id,
                len(full_text),
                len(chunks),
            )

            records = build_chunk_records(doc_id, chunks)

            for record in records:
                all_records.append(record)
                total_chunks += 1

                # Progress log every 50 chunks.
                if total_chunks % 50 == 0:
                    logger.info(
                        (
                            "Progress: {} chunks processed so far. "
                            "Last chunk_id='{}'."
                        ),
                        total_chunks,
                        record["chunk_id"],
                    )

    # Write the chunk records to the main JSON file.
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_records, file, ensure_ascii=False, indent=2)

    # Build and write metadata with parameters and run information.
    metadata: Dict[str, object] = {
        "run_datetime": run_datetime.isoformat(),
        "timestamp_label": timestamp_label,
        "parameters": {
            "input_dir": str(input_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "base_output_name": base_output_name,
            "words_per_chunk": words_per_chunk,
            "overlap": overlap,
            "log_level": log_level.upper(),
        },
        "output_file": output_path.name,
        "metadata_file": metadata_path.name,
        "n_pdfs": num_pdfs,
        "total_chunks": total_chunks,
        "pdf_filenames": [path.name for path in pdf_files],
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    logger.success(
        "Finished writing {} chunk records to '{}' and metadata to '{}'.",
        total_chunks,
        output_path,
        metadata_path.name,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments and return an argparse namespace.

    Command-line arguments override the default values defined in
    config/config.py.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert all PDFs in a directory into JSON chunks following "
            "the minimal contract (with basic cleaning and overlapping chunks)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=cfg.INPUT_DIR,
        help=(
            "Path to the directory containing PDF files "
            f"(default: '{cfg.INPUT_DIR}')."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=cfg.OUTPUT_DIR,
        help=(
            "Path to the directory where JSON outputs will be written "
            f"(default: '{cfg.OUTPUT_DIR}')."
        ),
    )
    parser.add_argument(
        "--base-output-name",
        type=str,
        default=cfg.BASE_OUTPUT_NAME,
        help=(
            "Base name for the output files. A timestamp will be appended "
            f"(default: '{cfg.BASE_OUTPUT_NAME}')."
        ),
    )
    parser.add_argument(
        "--words-per-chunk",
        type=int,
        default=cfg.WORDS_PER_CHUNK,
        help=(
            "Number of words per chunk "
            f"(default: {cfg.WORDS_PER_CHUNK})."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=cfg.OVERLAP,
        help=(
            "Number of overlapping words between chunks "
            f"(default: {cfg.OVERLAP})."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=cfg.LOG_LEVEL,
        help=(
            "Logging level, e.g. 'INFO' or 'DEBUG' "
            f"(default: '{cfg.LOG_LEVEL}')."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for command-line execution.

    This function configures logging, resolves the effective parameters
    (defaults plus CLI overrides), and runs the PDF chunker pipeline.
    """
    args = parse_args()
    configure_logging(args.log_level)

    run_pdf_chunker(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        base_output_name=args.base_output_name,
        words_per_chunk=args.words_per_chunk,
        overlap=args.overlap,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
