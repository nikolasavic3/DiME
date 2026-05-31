"""
Find smile-selective channels in DenseNet's denseblock3 empirically.

Method:
- Score N random CelebA images with the classifier.
- Bucket by confidence: high-smile (p > thr_pos) vs high-nonsmile (p < thr_neg).
- For each channel, compute mean spatial-mean activation in each bucket.
- Selectivity = mean_smile - mean_nonsmile (high = fires more on smiles).
- Visualize the top-K channels' spatial maps on a few test faces.
- Build a "channel-selective" mask using only those channels and compare
  to vanilla Grad-CAM.

Run:
    python channel_scan.py \
        --classifier_path models/classifier.pth \
        --image_dir /workdir/DiME_2/celeba/img_align_celeba \
        --output_dir output/channel_scan \
        --layer_name denseblock3 \
        --scan_size 200 \
        --top_k 8
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from core.classifier.densenet import ClassificationModel
from core.gradcam import compute_gradcam_mask


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
    heat = mask_01.expand(3, -1, -1).clone()
    heat[0] = mask_01[0]
    heat[1] = 0
    heat[2] = 1 - mask_01[0]
    return 0.55 * img_01 + 0.45 * heat


def get_layer(classifier, name):
    return getattr(classifier.feat_extract.feat_extract.features, name)


@torch.no_grad()
def channel_activations(classifier, imgs, layer):
    """Return (B, C, H, W) feature maps from the given layer."""
    feats = {}
    h = layer.register_forward_hook(lambda m, i, o: feats.update({"v": o}))
    _ = classifier(imgs)
    h.remove()
    return feats["v"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier_path", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", default="output/channel_scan")
    ap.add_argument("--layer_name", default="denseblock3")
    ap.add_argument("--scan_size", type=int, default=200,
                    help="how many random images to score for bucketing")
    ap.add_argument("--bucket_size", type=int, default=24,
                    help="max images per confidence bucket")
    ap.add_argument("--thr_pos", type=float, default=0.95)
    ap.add_argument("--thr_neg", type=float, default=0.05)
    ap.add_argument("--top_k", type=int, default=8,
                    help="how many top-selective channels to visualize")
    ap.add_argument("--query_label", type=int, default=31)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    classifier = ClassificationModel(args.classifier_path, args.query_label)
    classifier.eval().to(device)
    layer = get_layer(classifier, args.layer_name)

    paths = sorted(Path(args.image_dir).glob("*.jpg"))
    if len(paths) < args.scan_size:
        raise RuntimeError(f"Need >= {args.scan_size} images, found {len(paths)}")
    idx = torch.randperm(len(paths))[:args.scan_size]
    scan_paths = [paths[i] for i in idx.tolist()]

    print(f"Scoring {len(scan_paths)} images...")
    probs_all, imgs_all = [], []
    with torch.no_grad():
        for i in range(0, len(scan_paths), args.batch):
            chunk = scan_paths[i:i + args.batch]
            batch = torch.stack([load_image(p, args.image_size) for p in chunk]).to(device)
            probs_all.append(torch.sigmoid(classifier(batch)).cpu())
            imgs_all.append(batch.cpu())
    probs = torch.cat(probs_all)
    imgs_cpu = torch.cat(imgs_all)

    pos_mask = probs > args.thr_pos
    neg_mask = probs < args.thr_neg
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    print(f"high-smile (>{args.thr_pos}): {n_pos}   "
          f"high-nonsmile (<{args.thr_neg}): {n_neg}")
    if n_pos < 4 or n_neg < 4:
        raise RuntimeError("Not enough samples per bucket; lower thresholds "
                           "or raise --scan_size.")

    pos_idx = torch.nonzero(pos_mask).flatten()[:args.bucket_size]
    neg_idx = torch.nonzero(neg_mask).flatten()[:args.bucket_size]
    pos_imgs = imgs_cpu[pos_idx].to(device)
    neg_imgs = imgs_cpu[neg_idx].to(device)

    print(f"Hooking {args.layer_name}...")
    pos_feats = channel_activations(classifier, pos_imgs, layer)
    neg_feats = channel_activations(classifier, neg_imgs, layer)
    print(f"feature shape per image: {tuple(pos_feats.shape[1:])}")

    # per-channel mean activation across images and spatial dims
    pos_score = pos_feats.mean(dim=[0, 2, 3])
    neg_score = neg_feats.mean(dim=[0, 2, 3])
    selectivity = (pos_score - neg_score).cpu()
    top_vals, top_chans = torch.topk(selectivity, args.top_k)
    print("\nTop-K smile-selective channels:")
    for c, v in zip(top_chans.tolist(), top_vals.tolist()):
        print(f"  channel {c:4d}  selectivity={v:+.4f}")

    # visualize: for first few smiling test images, overlay each top channel's map
    n_show = min(4, pos_imgs.size(0))
    test_imgs = pos_imgs[:n_show]
    test_feats = channel_activations(classifier, test_imgs, layer)

    rows = []
    test_imgs_01 = (test_imgs.cpu() * 0.5 + 0.5).clamp(0, 1)
    for i in range(n_show):
        row = [test_imgs_01[i]]
        for c in top_chans.tolist():
            m = test_feats[i, c:c + 1].clamp(min=0)
            if m.max() > 0:
                m = m / m.max()
            m = F.interpolate(m.unsqueeze(0),
                              size=test_imgs.shape[2:],
                              mode="bilinear", align_corners=False)[0]
            row.append(overlay(test_imgs_01[i], m.cpu()))
        rows.append(torch.stack(row))
    grid = torch.cat(rows, dim=0)
    out = os.path.join(args.output_dir, "top_channels_grid.png")
    vutils.save_image(grid, out, nrow=1 + args.top_k, padding=2)
    print(f"\nsaved {out}")

    # alternative mask: weighted sum of top-K channels vs vanilla Grad-CAM
    target = torch.ones(n_show, dtype=torch.long, device=device)
    gradcam = compute_gradcam_mask(classifier, test_imgs, target,
                                   layer_name=args.layer_name, sharpen=2.0)

    sel_weights = top_vals.to(device).view(1, -1, 1, 1)
    sel_feats = test_feats[:, top_chans].clamp(min=0)
    sel_mask = (sel_feats * sel_weights).sum(dim=1, keepdim=True)
    sel_mask = F.relu(sel_mask)
    sel_mask = sel_mask / (sel_mask.amax(dim=[2, 3], keepdim=True) + 1e-8)
    sel_mask = sel_mask ** 2
    sel_mask = F.interpolate(sel_mask, size=test_imgs.shape[2:],
                             mode="bilinear", align_corners=False)

    cmp_rows = []
    for i in range(n_show):
        cmp_rows.append(torch.stack([
            test_imgs_01[i],
            overlay(test_imgs_01[i], gradcam[i].cpu()),
            overlay(test_imgs_01[i], sel_mask[i].cpu()),
        ]))
    cmp = torch.cat(cmp_rows, dim=0)
    out2 = os.path.join(args.output_dir, "gradcam_vs_selective.png")
    vutils.save_image(cmp, out2, nrow=3, padding=2)
    print(f"saved {out2}  (cols: original | Grad-CAM | top-K channels)")


if __name__ == "__main__":
    main()
