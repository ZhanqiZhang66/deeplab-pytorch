#!/usr/bin/env python
# coding: utf-8
# Author: Kazuto Nakashima
# modified by: Victoria Zhang
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019
# Purpose: This code will iteratively read in pictures in folder, do segmentation based on pre-trained config file, and combine the 
#          segmentaions that include animal/human classes into a merged class. 
#          The class-segmented masks (tree, human, water, bird, bear, desk, etc) will be saved as a .png file.
#          The merged mask will be saved at a .mat file.
#          

from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from libs.models import *
from libs.utils import DenseCRF
import os

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
    return classes


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize
    #print(CONFIG.IMAGE.SIZE.TEST)
    #scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    scale = 715 / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap


@click.group()
@click.pass_context
def main(ctx):
    """
    Demo with a trained model
    """

    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "-i",
    "--image-path",
    type=click.Path(exists=True),
    required=True,
    help="Image to be processed",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")


def single(config_path, model_path, image_path, cuda, crf):
    """
    Inference from a single image
    """
    import os
    # Setup
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # Locate Image Input Folder to iteratively readin image
    Picture_dir_List = [image_path + file for file in os.listdir(image_path) if file.endswith('.png')]
    # Inference
    for img_path in Picture_dir_List:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image, raw_image = preprocessing(image, device, CONFIG)
        labelmap = inference(model, image, raw_image, postprocessor)
        labels = np.unique(labelmap)


        # Show result for each class
        rows = np.floor(np.sqrt(len(labels) + 1))
        cols = np.ceil((len(labels) + 2) / rows)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(rows, cols, 1)
        ax.set_title("Input image")
        ax.imshow(raw_image[:, :, ::-1])
        ax.axis("off")

        h,w = np.shape(raw_image)[0], np.shape(raw_image)[1]
        print(h,w)
        print(img_path)
        animal_mask = np.zeros([h,w])
        animal_labels = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        #person bird car dog horse sheep cow elephant bear zebra giraffe
        for i, label in enumerate(labels):
            mask = labelmap == label
            if label in animal_labels:
                animal_mask = np.logical_or(animal_mask, mask)
                #print('animal mask ', animal_mask)
            ax = plt.subplot(rows, cols, i + 2)
            ax.set_title(classes[label])
            ax.imshow(raw_image[..., ::-1])
            ax.imshow(mask.astype(np.float32), alpha=0.5)
            ax.axis("off")
        #print(animal_mask)
        ax = plt.subplot(rows, cols, i + 2)
        ax.set_title("animal mask")
        ax.imshow(raw_image[:, :, ::-1])
        ax.imshow(animal_mask.astype(np.float32), alpha=0.5)
        ax.axis("off")

        plt.tight_layout()

        import os
        import scipy.io
        animal_mask_fl32 = animal_mask.astype(np.float32)
        mypath = os.path.abspath(__file__)[0:-7]
        myfile_img = img_path.split('/', 1)[1]
        myfile_mat = myfile_img[0:-4] + '.mat'

        mypath = mypath  + '\output_heatmaps'
        mypath_mat = mypath + '\\' + myfile_mat
        scipy.io.savemat(mypath_mat, mdict={'semantic_mask': animal_mask_fl32}) #the name cannot have space!
        plt.savefig(os.path.join(mypath, myfile_img))
        plt.show()


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")
@click.option("--camera-id", type=int, default=0, show_default=True, help="Device ID")
def live(config_path, model_path, cuda, crf, camera_id):
    """
    Inference from camera stream
    """

    # Setup
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # UVC camera stream
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    def colorize(labelmap):
        # Assign a unique color to each label
        labelmap = labelmap.astype(np.float32) / CONFIG.DATASET.N_CLASSES
        colormap = cm.jet_r(labelmap)[..., :-1] * 255.0
        return np.uint8(colormap)

    def mouse_event(event, x, y, flags, labelmap):
        # Show a class name of a mouse-overed pixel
        label = labelmap[y, x]
        name = classes[label]
        print(name)

    window_name = "{} + {}".format(CONFIG.MODEL.NAME, CONFIG.DATASET.NAME)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        _, frame = cap.read()
        image, raw_image = preprocessing(frame, device, CONFIG)
        labelmap = inference(model, image, raw_image, postprocessor)
        colormap = colorize(labelmap)

        # Register mouse callback function
        cv2.setMouseCallback(window_name, mouse_event, labelmap)

        # Overlay prediction
        cv2.addWeighted(colormap, 0.5, raw_image, 0.5, 0.0, raw_image)

        # Quit by pressing "q" key
        cv2.imshow(window_name, raw_image)
        if cv2.waitKey(10) == ord("q"):
            break


if __name__ == "__main__":
    main()
