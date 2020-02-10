import argparse
import os
import sys

from PIL import Image
import numpy as np
import torch
from torchvision import models
from torchvision import transforms

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

def tensor2image(t):
    return transforms.ToPILImage("RGB")(t)

def resize_and_crop(img, size):
    w, h = img.size
    if w < size or h < size:
        return None

    if w > h:
        scale = size / w
        w = size
        h = int(h * scale)
    else:
        scale = size / h
        h = size
        w = int(w * scale)

    trans = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.CenterCrop(size),
    ])
    return trans(img)

def segment_image(img, net):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406], 
            std = [0.229, 0.224, 0.225]),
    ])

    inp = trans(img).unsqueeze(0)
    out = net(inp)["out"]
    return torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

def render_mask(mask, layer):
    idx = mask == layer
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    r[idx] = 192
    g[idx] = 128
    b[idx] = 128
    return Image.fromarray(np.stack([r, g, b], axis=2))

def apply_mask(img, mask, layer):
    arr = np.array(img)
    idx = mask != layer
    arr[idx] = [1, 1, 1]
    return Image.fromarray(arr)

def preprocess_image(img, args):
    result = []

    img = resize_and_crop(img, args.size)
    if img is None:
        return None
    result.append(img)

    mask = segment_image(img, fcn)
    result.append(render_mask(mask, 15))

    img = apply_mask(img, mask, 15)
    result.append(img)

    return result

def preprocess_images(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_files = list(os.listdir(args.input_dir))
    num_files = len(input_files)
    num = 1
    for i, input_file in enumerate(input_files):
        input_img = Image.open(os.path.join(args.input_dir, input_file)).convert("RGB")
        result = preprocess_image(input_img, args)
        if result is not None:
            if not args.debug:
                result = result[-1:]
            output_img = Image.new("RGB", (args.size*len(result), args.size))
            for i, img in enumerate(result):
                output_img.paste(img, (i*args.size, 0))

            output_file = "{}/{:05d}.jpg".format(args.output_dir, num)
            output_img.save(output_file)
            num = num + 1
        sys.stderr.write("{}/{}\r".format(i+1, num_files))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="input images dir")
    parser.add_argument("--output-dir", type=str, help="output images dir")
    parser.add_argument("--size", type=int, default=256, help="output image size")
    parser.add_argument("--debug", action="store_true", help="store intermediate results")

    args = parser.parse_args()

    preprocess_images(args)

if __name__ == "__main__":
    main()
