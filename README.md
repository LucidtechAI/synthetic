# PDF anonymizer/synthesizer for Cradl

## Disclaimer

This code is experimental, and is not recommended for production use. The code is intended solely for educational purposes.

## Installation

```bash
$ pip install lucidtech-synthetic
```

## Usage

### Docker

```bash
docker run -v /path/to/src_dir:/root/src_dir:ro -v /path/to/dst_dir:/root/dst_dir -it lucidtechai/synthetic pdf /root/src_dir /root/dst_dir
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