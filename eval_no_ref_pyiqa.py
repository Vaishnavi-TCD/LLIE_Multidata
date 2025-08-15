#!/usr/bin/env python3
import os
import sys

import torch
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import pyiqa

# pick device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the three no-reference metrics
niqe   = pyiqa.create_metric("niqe",   device=DEVICE)
brisq  = pyiqa.create_metric("brisque",device=DEVICE)
piqe_m = pyiqa.create_metric("piqe",   device=DEVICE)

def score_image(path):
    img = Image.open(path).convert("RGB")
    x   = ToTensor()(img).unsqueeze(0).to(DEVICE)  # shape (1,3,H,W), range [0,1]
    with torch.no_grad():
        return (
            niqe(x).item(),
            brisq(x).item(),
            piqe_m(x).item()
        )

def main(folder):
    exts = {".jpg",".jpeg",".png",".bmp",".tiff"}
    scores = []

    for fn in sorted(os.listdir(folder)):
        if os.path.splitext(fn)[1].lower() not in exts:
            continue
        path = os.path.join(folder, fn)
        try:
            n, b, p = score_image(path)
        except Exception as e:
            print(f"❌ {fn}: {e}")
            continue

        scores.append((n,b,p))
        print(f"{fn:20s} → NIQE: {n:6.2f}   BRISQUE: {b:6.2f}   PIQE: {p:6.2f}")

    if not scores:
        print("No images found.")
        return

    arr = np.array(scores)
    mn, mb, mp = arr.mean(axis=0)
    print("\n─── Averages over {:d} images ───".format(len(arr)))
    print(f"Mean NIQE:    {mn:.2f}   (↓ better)")
    print(f"Mean BRISQUE: {mb:.2f}   (↓ better)")
    print(f"Mean PIQE:    {mp:.2f}   (↓ better)")

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python eval_no_ref_pyiqa.py <folder_of_images>")
        sys.exit(1)
    main(sys.argv[1])
