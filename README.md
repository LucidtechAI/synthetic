# Image/PDF synthesizer for Cradl

![Github Actions build status](https://github.com/LucidtechAI/synthetic/actions/workflows/pipeline.yaml/badge.svg)
![Python version](https://img.shields.io/pypi/pyversions/lucidtech-synthetic?logo=Python)
![PyPi version](https://img.shields.io/pypi/v/lucidtech-synthetic?logo=PyPi)
![Dockerhub version](https://img.shields.io/docker/v/lucidtechai/synthetic?logo=Docker)
![License](https://img.shields.io/github/license/LucidtechAI/synthetic)

## Disclaimer

This code does not guarantee that images/PDFs will be successfully synthesized. Use at your own risk.

## Installation

- [Link to PyPi](https://pypi.org/project/lucidtech-synthetic/)
- [Link to Dockerhub](https://hub.docker.com/r/lucidtechai/synthetic/tags)

```bash
$ pip install lucidtech-synthetic
```

Make sure to have the following software installed on your system before using the CLI:

- ghostscript

## Basic Usage

### Docker

We recommend disabling networking and setting `/path/to/src_dir` to read-only as shown below:

```bash
docker run --network none -v /path/to/src_dir:/root/src_dir:ro -v /path/to/dst_dir:/root/dst_dir -it lucidtechai/synthetic pdf /root/src_dir /root/dst_dir
docker run --network none -v /path/to/src_dir:/root/src_dir:ro -v /path/to/dst_dir:/root/dst_dir -it lucidtechai/synthetic image /root/src_dir /root/dst_dir /usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf 6-36
```

### CLI

```bash
synthetic pdf /path/to/src_dir /path/to/dst_dir
synthetic image /path/to/src_dir /path/to/dst_dir /usr/share/fonts/ubuntu/Ubuntu-B.ttf 6-36
```

`/path/to/src_dir` is the input directory and should contain your image/PDFs and JSON ground truths
`/path/to/dst_dir` is the output directory where synthesized image/PDFs and JSON ground truths will be written to

Here is an example of the directory layout for `/path/to/src_dir`:
```
/path/to/src_dir
├── a.pdf|jpeg
├── a.json
├── b.pdf|jpeg
└── b.json
```

The output directory will follow the same layout but with modified images/PDFs and JSON ground truths:
```
/path/to/dst_dir
├── a.pdf|jpeg
├── a.json
├── b.pdf|jpeg
└── b.json
```

## Using a custom Synthesizer

The following examples shown are for custom PDF synthesizers, but it works similarly for image synthesizers

### CLI

```bash
synthetic pdf /path/to/src_dir /path/to/dst_dir --synthesizer-class path.to.python.Class
```

Make sure that parent directory of `path.to.python.Class` is in your `PYTHONPATH`

Example using one of the example Synthesizers in `examples` directory

```bash
synthetic pdf /path/to/src_dir /path/to/dst_dir --synthesizer-class examples.exclude-words.synthesizer.ExcludeWordsSynthesizer
```

### Docker

```bash
docker run --network none -v /path/to/synthesizer:/root/synthesizer -v /path/to/src_dir:/root/src_dir:ro -v /path/to/dst_dir:/root/dst_dir -it lucidtechai/synthetic pdf /root/src_dir /root/dst_dir --synthesizer-class mypythonfile.ExcludeWordsSynthesizer
```

Note that the python module must be mounted into the docker container to `/root/synthesizer` for it to work. In the above example we assume a directory structure of your custom synthesizer to be like below.

```
/path/to/synthesizer
└── mypythonfile.py
```

Example using one of the example Synthesizers in `examples` directory. The `examples` directory should already exist in the image so that we don't need to mount anything additional.

```bash
docker run --network none -v /path/to/src_dir:/root/src_dir:ro -v /path/to/dst_dir:/root/dst_dir -it lucidtechai/synthetic pdf /root/src_dir /root/dst_dir --synthesizer-class examples.exclude-words.synthesizer.ExcludeWordsSynthesizer
```

## Help

All methods support the `--help` flag which will provide information on the purpose of the method, 
and what arguments could be added.

```bash
$ synthetic --help
```

## Known Issues

### Image Synthesizer

- Synthesized text does not follow the rotation of the document in the image if document is rotated
- Bounding boxes needed in ground truth

### PDF Synthesizer

- Does not synthesize images inside PDF
- Replaced strings are sometimes not hexadecimal encoded even when expected to be
- Text appearing as single characters with custom spacing in PDF will often yield poor results
