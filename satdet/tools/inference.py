import cv2
import numpy as np
import torch

from datasets import to_tensor, ImageTransform


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, cfg, device):
    img = cv2.imread(img)
    data = _prepare_data(img, img_transform, cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, cfg, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, device)


def inference_detector(model, imgs, cfg, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, cfg, device)
    else:
        return _inference_generator(model, imgs, img_transform, cfg, device)
