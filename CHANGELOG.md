# Changelog

## Version 0.3.5 - 2022-05-27

- Now catching additional exceptions that might occur when synthesizing PDFs.

## Version 0.3.1 - 2022-05-27

- Now checking `dst_dir` for existing files to avoid processing already processed PDFs.
- Added optional `--max-fonts` to specify the maximum number of fonts a PDF can contain for synthesizing. PDFs breaching the threshold will be discarded.

## Version 0.3.0 - 2022-05-19

- Added optional `--num-outputs-per-document` to specify how many output PDFs to create per input PDF
- Added optional `--max-pages` to specify the maximum number of pages a PDF can contain for synthesizing. PDFs breaching the threshold will be discarded.

## Version 0.2.2 - 2022-05-12

- Save new PDFs in compressed non-qpdf mode
