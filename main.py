import argparse
import sys
import os
import homography, rectification

parser = argparse.ArgumentParser()

parser.add_argument("detection", type=str, choices=["manual", "auto"], default="manual", help="Choose manual or auto stitching")
parser.add_argument("stitch", type=str, choices=["plane", "merge"])
parser.add_argument("images", nargs="+")

args = parser.parse_args()

def manual_stitching():
    pass

def auto_stitching():
    pass

if __name__ == "__main__":
    
    num_imgs = len(args.images)
    
    print(f"""
stitching {num_imgs} images...
    detection-method = {args.detection}
    stitching-method = {args.stitch}
    """)
    
    if args.detection == "manual":
        pass
        