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
import csv
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
    #print(classes)
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

        h,w = np.shape(raw_image)[0], np.shape(raw_image)[1]

        sports_mask = np.zeros([h,w])
        sports_labels = [42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
        #tennis-racket surfboard skateboard baseball-glove baseball-bat kite sports-ball snowboard frisbee

        animal_mask = np.zeros([h,w])
        animal_labels = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # person bird cat dog horse sheep cow elephant bear zebra giraffe

        accessory_mask = np.zeros([h,w])
        accessory_labels = [32, 31, 30, 29, 28, 27, 26, 25]
        #suitcase tie backpack glasses shoe umbrella backpack hat

        outdoor_mask = np.zeros([h,w])
        outdoor_labels = [14, 13, 12, 11, 10, 9]
        #bench parking-meter stop-sign street-sign fire-hydrant traffic-light

        vehicle_mask = np.zeros([h,w])
        vehicle_labels = [8, 7, 6, 5, 4, 3, 2, 1]
        #boat truck train bus airplane motorcylye car bicycle

        person_mask = np.zeros([h,w])
        person_labels = [0]
        #person

        indoor_mask = np.zeros([h,w])
        indoor_labels = [90, 89, 88, 87, 86, 85, 84, 83]
        #hair-brush, toothbrush, hair-dyer, teddy-bear, scissors, case, clock, book

        appliance_mask = np.zeros([h,w])
        appliance_labels = [82, 81, 80, 79, 78, 77]
        #blender fridge sink toaster oven microwave

        electronics_mask = np.zeros([h,w])
        electronics_labels = [71, 72, 73, 74, 75, 76]
        # TV laptop mouse remote keyboard cell-phone

        furniture_mask = np.zeros([h,w])
        furniture_labels = [70, 69, 68, 67, 66, 65, 64, 63, 62, 61,
                            122, 160, 129, 106, 65, 107, 97, 155, 164, 68, 70]
        #door toilet desk window dinning-table mirror bed potted-plant couch chair
        # furniture-other stairs light counter mirror cupboard cabinet shelf table desk door

        food_mask = np.zeros([h, w])
        food_labels = [51,52,53,54,55,56,57,58,59,60,120,121,152,169]
        # banana apple sandwich orange broccoli carrot hot-dog piazza donut cake food-other fruit salad vegetable

        kitchen_mask = np.zeros([h,w])
        kitchen_labels = [50, 49, 48, 47, 46, 45, 44, 43]
        #bowl spoon knife fork cup wine-glass plate bottle


        water_mask = np.zeros([h,w])
        water_labels = [177, 178, 154, 147, 119]
        # water-other, waterdrops sea river fog

        ground_mask = np.zeros([h,w])
        ground_labels = [125, 144, 143, 146, 139, 148, 124, 135, 110, 158, 153]
        #groud-other, playingfield, platform railroad pavement road gravel mud dirt snow sand

        solid_mask = np.zeros([h,w])
        solid_labels = [159, 126, 134, 161, 149, 181]
        #solid-other hill mountain stone rock wood

        sky_mask = np.zeros([h,w])
        sky_labels = [156, 105]
        #sky clouds

        plant_mask = np.zeros([h, w]) #
        plant_labels = [96, 123, 128, 141, 162, 133, 93, 118, 168, 63]
        # bush grass leaves plant-other straw moss branch flower tree potted-plant

        structural_mask = np.zeros([h,w])
        structural_labels = [163, 145, 137, 98, 112]
        #structural-other railing net cage fence

        building_mask = np.zeros([h,w])
        building_labels = [95, 150, 165, 94, 157, 127]
        #building-other roof tent bridge skycraper house

        textile_mask = np.zeros([h,w])
        textile_labels = [166, 91, 140, 92, 108, 103, 136, 167, 130, 151]
        # textile-other banner pillow blanket curtain cloth napkin towel mat rug

        window_mask = np.zeros([h,w])
        window_labels = [180, 179]
        #window-other window-blind

        floor_mask = np.zeros([h,w])
        floor_labels = [114, 115, 113, 117, 116, 100]
        #floor-other floor-stone floor-marble floor-wood floor-tile carpet

        ceiling_mask = np.zeros([h,w])
        ceiling_labels = [101, 102]
        #ceiling-other ceiling-tile

        wall_mask = np.zeros([h,w])
        wall_labels = [172, 171, 174, 170, 176, 173, 175]
        #wall-other wall-concrete wall-stone wall-brick wall-wod wall-panel wall-tile

        material_mask = np.zeros([h,w])
        material_labels = [131, 142, 138, 99]
        #metal plastic paper carboard

        import scipy.io


        # Show result for each class
        # rows = np.floor(np.sqrt(len(labels) + 1))
        # cols = np.ceil((len(labels) + 2) / rows)
        # plt.figure(figsize=(10, 10))
        # ax = plt.subplot(rows, cols, 1)
        # ax.set_title("Input image")
        # ax.imshow(raw_image[:, :, ::-1])
        # ax.axis("off")

        myfile_img = img_path.split('/', 1)[1]
        mypath = os.path.abspath(__file__)[0:-7]


        masks = [myfile_img[0:-4]]
        for i, label in enumerate(labels):
            mask_name = classes[label]
            masks.append(classes[label])
            mask = labelmap == label
            # save mask to mat file

            myfile_mat = myfile_img[0:-4] + '_' + mask_name + '.mat'

            mypath_ = mypath + '\output_freeviews_masks'  # TODO
            mypath_mat = mypath_ + '\\' + myfile_mat

            scipy.io.savemat(mypath_mat, mdict={classes[label]: mask.astype(np.float32)})  # the name cannot have space!

            if label in sports_labels:
                sports_mask = np.logical_or(sports_mask, mask)
                print("sports")
            elif label in accessory_labels:
                accessory_mask = np.logical_or(accessory_mask, mask)
                print("accessory")
            elif label in animal_labels:
                animal_mask = np.logical_or(animal_mask, mask)
                print('animal')
            elif label in outdoor_labels:
                outdoor_mask = np.logical_or(outdoor_mask, mask)
                print('outdoor')
            elif label in vehicle_labels:
                vehicle_mask = np.logical_or(vehicle_mask, mask)
                print('vehicle')
            elif label in person_labels:
                person_mask = np.logical_or(person_mask, mask)
                print('person')
            elif label in indoor_labels:
                indoor_mask = np.logical_or(indoor_mask, mask)
                print('indoor')
            elif label in appliance_labels:
                appliance_mask = np.logical_or(appliance_mask, mask)
                print('appliance')
            elif label in electronics_labels:
                electronics_mask = np.logical_or(electronics_mask, mask)
                print('electronics')
            elif label in furniture_labels:
                furniture_mask = np.logical_or(furniture_mask, mask)
                print('furniture')
            elif label in food_labels:
                food_mask = np.logical_or(food_mask, mask)
                print('food')
            elif label in kitchen_labels:
                kitchen_mask = np.logical_or(kitchen_mask, mask)
                print('kitchen')
            elif label in plant_labels:
                plant_mask = np.logical_or(plant_mask, mask)
                print('plant')
            elif label in water_labels:
                water_mask = np.logical_or(water_mask, mask)
                print('water')
            elif label in ground_labels:
                ground_mask = np.logical_or(ground_mask, mask)
                print('ground')
            elif label in solid_labels:
                solid_mask = np.logical_or(solid_mask, mask)
                print('solid')
            elif label in sky_labels:
                sky_mask = np.logical_or(sky_mask, mask)
                print('sky')
            elif label in structural_labels:
                structural_mask = np.logical_or(structural_mask, mask)
                print('structural')
            elif label in building_labels:
                building_mask = np.logical_or(building_mask, mask)
                print('building')
            elif label in textile_labels: 
                textile_mask = np.logical_or(textile_mask, mask)
                print('textile')
            elif label in window_labels:
                window_mask = np.logical_or(window_mask, mask)
                print('window')
            elif label in floor_labels:
                floor_mask = np.logical_or(floor_mask, mask)
                print('floor')
            elif label in ceiling_labels:
                ceiling_mask = np.logical_or(ceiling_mask, mask)
                print('ceiling')
            elif label in wall_labels:
                wall_mask = np.logical_or(wall_mask, mask)
                print('wall')
            elif label in material_labels:
                material_mask = np.logical_or(material_mask, mask)
                print('material')
            else:
                print('no label')

            ax = plt.subplot(rows, cols, i + 2)
            ax.set_title(classes[label])
            ax.imshow(raw_image[..., ::-1])
            ax.imshow(mask.astype(np.float32), alpha=0.5)
            ax.axis("off")

        my_masks = ['sports_mask', 'accessory_mask', 'animal_mask', 'outdoor_mask', 'vehicle_mask',
                    'person_mask', 'indoor_mask', 'appliance_mask', 'electronics_mask', 'furniture_mask',
                    'food_mask', 'kitchen_mask', 'plant_mask', 'water_mask', 'ground_mask',
                    'solid_mask', 'sky_mask', 'structural_mask', 'building_mask', 'textile_mask',
                    'window_mask', 'floor_mask', 'ceiling_mask', 'wall_mask', 'material_mask']

        with open('picture_mask.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(masks)
        with open('picture_mask_hierarchy.csv', 'a') as file1:
            writer = csv.writer(file1)
            temp = [myfile_img[0:-4]] + my_masks
            writer.writerow(temp)

        for mask_name in my_masks:
            this_mask = eval(mask_name)
            this_mask_fl32 = this_mask.astype(np.float32)
            myfile_mat = myfile_img[0:-4] + mask_name + '.mat'
            mypath_ = mypath  + '\output_freeviews_masks' #TODO
            mypath_mat = mypath_ + '\\' + myfile_mat
            scipy.io.savemat(mypath_mat, mdict={mask_name: this_mask_fl32}) #the name cannot have space!

        myfile_img_name = myfile_img[0:-4] + '_masks.png'
        print(myfile_img_name)
        plt.tight_layout()
        plt.savefig(os.path.join(mypath_, myfile_img_name))
        plt.close()

        # Show result for 25 super class
        rows_super = 5
        cols_super = 5
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(rows_super, rows_super, 1)
        ax1.set_title("Input image")
        ax1.imshow(raw_image[:, :, ::-1])
        ax1.axis("off")

        for i, mask in enumerate(my_masks):
            this_mask = eval(my_masks[i])
            this_mask_fl32 = this_mask.astype(np.float32)
            ax1 = plt.subplot(rows_super, cols_super, i + 1)
            ax1.set_title(my_masks[i])
            ax1.imshow(raw_image[:, :, ::-1])
            ax1.imshow(this_mask_fl32, alpha=0.5)
            ax1.axis("off")

        myfile_img_name = myfile_img[0:-4] + '_super_mask.png'
        plt.savefig(os.path.join(mypath_, myfile_img_name))
        plt.tight_layout()

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
