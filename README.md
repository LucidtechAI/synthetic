# Introduction

This script generates synthetic training data from pre-defined templates.

**Disclaimer: This code is experimental, and is not recommended for production use. The code is intended solely for educational purposes.**

# Installation
``` bash
$ pip install -r requirements.txt
```

# Image synthetization
With image synthetization, documents are synthetized by drawing on top of an images.
Consequently, image synthetization can be effective on a broad range of document types.
Drawbacks with image synthetization is that image artifacts and font recognition issues
may reduce the quality of the synthetic date.

## Template definition
The template structure for image synthetization is defined below:

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

The bounding boxes are denoted with relative coordinates.

## Usage
Example usage:
``` bash
$ python synth.py image example.yaml /usr/share/fonts/TTF --font-size-range 40:60
```

The script generates a synthetic version of the source image where areas denoted by the bounding boxes have been substituted with the corresponding text.

# PDF synthetization
PDF synthetization eliminates some of the challenges with image synthetization. In particular, font detection
and image artifacts is less problematic.

## Template definition
The template definition for image synthetization consists of two parts. The first part is defined below:

example.yaml:
``` yaml
path: path/to/document_templated.pdf
dest: path/to/document_synthetic.pdf
labels:
  first_label:
    text: '431.23'
  second_label:
    text: 'foobar'
    boxes:
      - ...
```

The second part consists of a PDF with uncompressed object streams. [QPDF](https://github.com/qpdf/qpdf) can
be used to decompress object streams in PDF files so that you can edit the PDF in your favorite text editor:

```bash
$ qpdf --stream-data=uncompress input.pdf output.pdf
```

When you open output.pdf in a [text editor](https://www.vim.org), you will see lines like this:

``` PDF
...
82.815972 0 Td
(Some fancy text here)Tj
/F1 9 Tf
...
```

_synth.py_ assumes that the labels in _example.yaml_ correspond to variables in the uncompressed PDF. A variable is denoted as `__(variable_name)__`. For example:

``` PDF
...
82.815972 0 Td
(Some __(first_label)__)Tj
/F1 9 Tf
...
```

## Usage
When all variables have been added to the PDF, you are ready to synthesize data:

```bash
$ python synth.py pdf example.yaml
```

# Examples
See the examples directory.
