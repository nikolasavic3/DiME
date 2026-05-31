import torch
import torch.nn.functional as F


def _resolve_layer(classifier, layer_name):
    features = classifier.feat_extract.feat_extract.features
    if not hasattr(features, layer_name):
        raise ValueError(f"DenseNet has no layer '{layer_name}'")
    return getattr(features, layer_name)


def compute_gradcam_mask(classifier, x0, target_y,
                         layer_name="denseblock3", sharpen=2.0):
    """
    Grad-CAM mask from a chosen DenseNet feature block.
    Same binary CE loss as clean_class_cond_fn so gradients are consistent.

    layer_name: 'denseblock2' (16x16), 'denseblock3' (8x8), 'denseblock4' (4x4)
    sharpen: exponent applied after normalization to concentrate mass (1.0 = off)
    Returns: (B, 1, H, W) mask in [0, 1]
    """
    feats, grads = {}, {}
    layer = _resolve_layer(classifier, layer_name)

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
        if sharpen != 1.0:
            c = c ** sharpen
        c = F.interpolate(c, size=x0.shape[2:], mode="bilinear", align_corners=False)
        masks.append(c)

    return torch.cat(masks, dim=0).detach()
