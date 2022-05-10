# PDF anonymizer/synthesizer for Cradl

## Disclaimer

This code is experimental, and is not recommended for production use. The code is intended solely for educational purposes.

## Installation

```bash
$ pip install lucidtech-synthetic
```

## Usage

`/path/to/src_dir` should contain your PDFs and JSON ground truths
`/path/to/dst_dir` is the directory you want

Here is an example of the directory layout for `/path/to/src_dir`:
```
/path/to/src_dir
├── a.pdf
├── a.json
├── b.pdf
├── b.json
├── c.pdf
└── c.json
```

### Docker

We recommend disabling networking and setting `/path/to/src_dir` to read-only as shown below:

```bash
docker run --network none -v /path/to/src_dir:/root/src_dir:ro -v /path/to/dst_dir:/root/dst_dir -it lucidtechai/synthetic pdf /root/src_dir /root/dst_dir
```

### CLI

```bash
synthetic pdf /path/to/src_dir /path/to/dst_dir
```

All methods support the `--help` flag which will provide information on the purpose of the method, 
and what arguments could be added.

```bash
$ synthetic --help
```

## Known Issues

### PDF Synthesizer

- Does not synthesize images
- Replaced strings are never hexadecimal encoded