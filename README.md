# PDF anonymizer/synthesizer for Cradl

## Disclaimer

This code does not guarantee that PDFs will be successfully anonymized/synthesized. Use at your own risk.

## Installation

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
```

### CLI

```bash
synthetic pdf /path/to/src_dir /path/to/dst_dir
```

`/path/to/src_dir` is the input directory and should contain your PDFs and JSON ground truths
`/path/to/dst_dir` is the output directory where synthesized PDFs and JSON ground truths will be written to

Here is an example of the directory layout for `/path/to/src_dir`:
```
/path/to/src_dir
├── a.pdf
├── a.json
├── b.pdf
└── b.json
```

The output directory will follow the same layout but with modified PDFs and JSON ground truths:
```
/path/to/dst_dir
├── a.pdf
├── a.json
├── b.pdf
└── b.json
```

## Using a custom Synthesizer

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

### PDF Synthesizer

- Does not synthesize images
- Replaced strings are sometimes not hexadecimal encoded even when expected to be
- Text appearing as single characters with custom spacing in PDF will often yield poor results