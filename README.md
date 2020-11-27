# Project 05: [Auto]Stitching Photo Mosaics

### Requirements:

Make sure all images are named with the format "name0.jpg".
This program can only stitch up to 10 images due to 0-9 for now.

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
    -d, --debug Show constants.DEBUGging print statements.
```
