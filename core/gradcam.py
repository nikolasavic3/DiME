import torch
import torch.nn.functional as F


def compute_gradcam_mask(classifier, x0, target_y):
    """
    Compute Grad-CAM mask from the DenseNet classifier's last dense block.
    Uses the same binary CE loss as clean_class_cond_fn so gradients are consistent.

    classifier: ClassificationModel instance
    x0: image tensor (B, C, H, W) in [-1, 1]
    target_y: target label tensor (B,)
    Returns: (B, 1, H, W) mask in [0, 1]
    """
    feats, grads = {}, {}
    layer = classifier.feat_extract.feat_extract.features.denseblock4

    fh = layer.register_forward_hook(lambda m, i, o: feats.update({"v": o}))
    bh = layer.register_full_backward_hook(lambda m, gi, go: grads.update({"v": go[0]}))

    x = x0.detach().float().requires_grad_(True)
    logits = classifier(x)
    classifier.zero_grad()

    y = target_y.float().to(logits.device)
    selected = y * logits - (1 - y) * logits
    loss = -F.logsigmoid(selected).sum()
    loss.backward()

    fh.remove()
    bh.remove()

    feat = feats["v"].float()
    grad = grads["v"].float()
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1, keepdim=True))

    masks = []
    for i in range(cam.shape[0]):
        c = cam[i:i+1]
        if c.max() > 0:
            c = c / c.max()
        c = F.interpolate(c, size=x0.shape[2:], mode="bilinear", align_corners=False)
        masks.append(c)

    return torch.cat(masks, dim=0).detach()
