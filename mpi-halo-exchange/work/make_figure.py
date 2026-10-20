#!/usr/bin/env python3
"""Makes results/seams_v1_vs_v2.png: |candidate - serial| for v1 and v2.
Run inside work/ after compare.py has been run for both versions."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

serial = np.load("test_image_filtered_serial.npy").astype(int)
panels = []
for name, title in (("test_image_filtered_v1", "v1: no halo"),
                    ("test_image_filtered_v2", "v2: halo exchange")):
    d = np.abs(np.load(f"{name}.npy").astype(int) - serial)
    img = Image.fromarray((np.clip(d * 40, 0, 255)).astype(np.uint8)).convert("RGB")
    panels.append((title, int((d > 0).sum()), img))

font = ImageFont.load_default(size=16)
h, w = panels[0][2].size[1], panels[0][2].size[0]
canvas = Image.new("RGB", (w * 2 + 30, h + 50), "white")
dr = ImageDraw.Draw(canvas)
for i, (title, n, img) in enumerate(panels):
    x = 10 + i * (w + 10)
    canvas.paste(img, (x, 40))
    dr.text((x, 10), f"{title}: {n} differing pixels", fill="black", font=font)
canvas.save("../results/seams_v1_vs_v2.png")
print("wrote ../results/seams_v1_vs_v2.png")
