import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser

debug = False

parser = ArgumentParser()
parser.add_argument("-r", "--root_dir", required=True, type=str)
parser.add_argument("-f", "--filter", type=str, default="")
args = parser.parse_args()

img_dir = os.path.join(args.root_dir, "image")
imgs = list(filter(lambda x: args.filter in x, glob("**/*.jpg", root_dir=img_dir)))

if not debug:
    output_dir = os.path.join(args.root_dir, "seg_output")
    os.makedirs(output_dir, exist_ok=True)

for img_path in tqdm(imgs):
    img = cv2.imread(os.path.join(img_dir, img_path))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:,:,1]
    th = ~cv2.threshold(a_channel,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    unmasked = cv2.bitwise_and(img, img, mask=th)
    masked = cv2.bitwise_and(img, img, mask=~th)

    if debug == True:
        cv2.imshow("mask", masked)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord('q'):
            break
    else:
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), masked)
