# Project 05: [Auto]Stitching Photo Mosaics

### Requirements:

Input images need to be in JPEG format.
Make sure all images are named in the format "name0.jpg" (or "name0.jpeg").

This program can only stitch 2 or 3 images. Please input them in stitching order.

### Instructions:

```
usage: main.py [-h] [-s] [-l] [-d] {manual,auto} images [images ...]

positional arguments:
    {manual,auto} Choose manual or auto stitching
    images 2 or 3 images to be stitched in entered in stitching order

optional arguments:
    -h, --help show this help message and exit
    -s, --save Save intermediate data.
    -l, --load Load points from pickle files.
    -d, --debug Show debugging print statements.
```
