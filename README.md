# Introduction

This script generates synthetic training data from pre-defined templates. A template is defined by a set of labels with corresponding bounding boxes.

**Disclaimer: This code is experimental, and is not recommended for production use. The code is intended solely for educational purposes.**

# Installation
``` bash
$ pip install -r requirements.txt
```

# Usage
Example usage:
``` bash
$ python synth.py example.yaml /usr/share/fonts/TTF --font-size-range 40:60
```

example.yaml:
``` yaml
path: path/to/image.jpg
dest: path/to/image_synthetic.jpg
labels:
  first_label:
    text: '431.23'
    boxes:
      - xmin: 0.609375
        ymin: 0.698546
        xmax: 0.916666
        ymax: 0.740140
  second_label:
    text: 'foobar'
    boxes:
      - ...
```

The script generates a synthetic version of the source image where areas denoted by the bounding boxes have been substituted with the corresponding text.

# Examples
See the examples directory.
