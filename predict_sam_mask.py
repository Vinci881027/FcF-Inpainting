"""
Predict mask for images using SAM.
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from utils_sam import show_box, show_mask, show_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", help="which gpu to use", nargs="+", default=[0], type=int
    )
    parser.add_argument(
        "--ckpt_sam",
        help="path to the SAM checkpoint",
        type=str,
        default="ckpt_sam/sam_vit_h_4b8939.pth",
    )
    parser.add_argument("--model_type", help="model type", type=str, default="vit_h")
    parser.add_argument(
        "--img_data",
        help="path to the image data",
        type=str,
        default="datasets/places2_dataset/evaluation/places2_256",
    )
    parser.add_argument(
        "--input_points", help="input points", type=str, default="datasets/places2_dataset/evaluation/places2_256_input_point.txt"
    )
    parser.add_argument(
        "--resolution", help="resolution of the images", type=int, default=256
    )
    parser.add_argument(
        "--dilation", help="dilation of the mask", type=int, default=5
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(e) for e in args.gpu)
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt_sam).cuda()
    predictor = SamPredictor(sam)

    input_point_arr = np.loadtxt(args.input_points, delimiter=",")
    input_point_arr = input_point_arr.reshape(-1, 1, 2)
    assert len(input_point_arr) == len(os.listdir(args.img_data))

    for i, img in enumerate(sorted(os.listdir(args.img_data))):
        image = cv2.imread(os.path.join(args.img_data, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_point = input_point_arr[i] * args.resolution // 256
        input_label = np.array([1])

        # predict
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        # save mask which has the highest score
        mask = masks[np.argmax(scores)]
        score = scores[np.argmax(scores)]
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        if not os.path.exists(f"{args.img_data}_with_points"):
            os.makedirs(f"{args.img_data}_with_points")
        plt.savefig(
            f"{args.img_data}_with_points/{img}"
        )
        plt.close()

        # change mask's shape from [resolution, resolution] to [3, resolution, resolution]
        mask = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
        # save mask to png, mask set to [255, 255, 255], background set to [0, 0, 0]
        mask_000 = np.where(mask == 1, 255, 0).astype(np.uint8)
        mask_000 = np.transpose(mask_000, (1, 2, 0))
        # enlarge mask area to include the whole object
        if args.dilation > 0:
            mask_000 = cv2.dilate(mask_000, np.ones((args.dilation, args.dilation), np.uint8), iterations=1)
        if not os.path.exists(f"{args.img_data}_masks"):
            os.makedirs(f"{args.img_data}_masks")
        plt.imsave(
            f"{args.img_data}_masks/{img.split('.')[0]}_mask000.png",
            mask_000,
        )
        plt.close()


if __name__ == "__main__":
    main()
