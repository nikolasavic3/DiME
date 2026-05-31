"""
Standalone Grad-CAM sanity check.
Loads a few CelebA images, runs the smile classifier through three candidate
DenseNet blocks, and saves per-image overlays + a side-by-side grid.

Run on the server:
    python test_gradcam.py \
        --classifier_path models/celeba/classifier.pt \
        --image_dir img_align_celeba \
        --output_dir output/gradcam_test \
        --num_images 8 \
        --query_label 31
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from core.gradcam import compute_gradcam_mask
from core.classifier.densenet import ClassificationModel


def load_image(path, size=128):
    img = Image.open(path).convert("RGB")
    tf = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return tf(img)


def overlay(img_01, mask_01):
    """img_01, mask_01 in [0,1], shapes (3,H,W) and (1,H,W). Returns (3,H,W)."""
    heat = mask_01.expand(3, -1, -1).clone()
    heat[0] = mask_01[0]
    heat[1] = 0
    heat[2] = 1 - mask_01[0]
    return 0.55 * img_01 + 0.45 * heat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier_path", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", default="output/gradcam_test")
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--query_label", type=int, default=31)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--target", type=int, default=1,
                    help="target class (1=force smile, 0=force no-smile)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    classifier = ClassificationModel(args.classifier_path, args.query_label)
    classifier.eval().to(device)

    image_paths = sorted(Path(args.image_dir).glob("*.jpg"))[:args.num_images]
    if not image_paths:
        raise RuntimeError(f"No .jpg files in {args.image_dir}")

    imgs = torch.stack([load_image(p, args.image_size) for p in image_paths]).to(device)
    target = torch.full((imgs.size(0),), args.target, dtype=torch.long, device=device)

    with torch.no_grad():
        prob = torch.sigmoid(classifier(imgs))
    print("classifier p(smile):", [f"{p:.2f}" for p in prob.tolist()])

    layers = [
        ("denseblock2", 1.0),
        ("denseblock2", 2.0),
        ("denseblock3", 1.0),
        ("denseblock3", 2.0),
        ("denseblock4", 1.0),
        ("denseblock4", 2.0),
    ]

    masks_by_cfg = {}
    for name, sharpen in layers:
        m = compute_gradcam_mask(classifier, imgs, target,
                                 layer_name=name, sharpen=sharpen)
        masks_by_cfg[(name, sharpen)] = m.cpu()
        print(f"{name} sharpen={sharpen}: mean={m.mean():.3f} "
              f"frac>0.5={(m > 0.5).float().mean():.3f}")

    imgs_01 = (imgs.cpu() * 0.5 + 0.5).clamp(0, 1)

    # one big grid: rows = images, cols = original + each (layer, sharpen) overlay
    grid_rows = []
    for i in range(imgs_01.size(0)):
        row = [imgs_01[i]]
        for cfg in layers:
            row.append(overlay(imgs_01[i], masks_by_cfg[cfg][i]))
        grid_rows.append(torch.stack(row))
    grid = torch.cat(grid_rows, dim=0)
    out_path = os.path.join(args.output_dir, "grid.png")
    vutils.save_image(grid, out_path, nrow=1 + len(layers), padding=2)
    print(f"saved {out_path}")

    # also save raw masks for inspection
    for cfg, m in masks_by_cfg.items():
        name, sharpen = cfg
        vutils.save_image(m, os.path.join(args.output_dir,
                                         f"mask_{name}_s{sharpen}.png"))


if __name__ == "__main__":
    main()
