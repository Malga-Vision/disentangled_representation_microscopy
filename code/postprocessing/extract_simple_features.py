


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




import os
import pickle
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
import skimage.morphology


from code.choose_dataset import get_named_dataset


def rgbtoint32(rgb):
    color = 0
    for c in rgb[::-1]:
        color = (color<<8) + c
        # Do not forget parenthesis.
        # color<< 8 + c is equivalent of color << (8+c)
    return color

def int32torgb(color):
    rgb = []
    for i in range(3):
        rgb.append(color&0xff)
        color = color >> 8
    return rgb


def solidity(mask):
    mask = np.where(mask == 255, 1, 0)

    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming we are working with a single object mask, use the first contour
    cnt = max(contours, key=cv2.contourArea)  # contours[0]

    # Compute the convex hull
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    #print(f"Convex Hull Area: {hull_area}")
    #print(f"Convex Hull Perimeter: {hull_perimeter}")

    # Compute the solidity
    contour_area = cv2.contourArea(cnt)
    solidity = contour_area / hull_area
    #print(f"Solidity: {solidity}")


    return solidity


def eccentricity(mask):
    mask = np.where(mask == 255, 1, 0)

    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming we are working with a single object mask, use the first contour
    cnt = max(contours, key=cv2.contourArea)  # contours[0]

    # plt.imshow(mask)
    # plt.show()

    # Hu Moments
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    eccentricity = ((moments["mu20"] - moments["mu02"]) ** 2 + 4 * moments["mu11"] ** 2) / (
                (moments["mu20"] + moments["mu02"]) ** 2)

    return eccentricity



def roundness(mask):
    mask = np.where(mask == 255, 1, 0)

    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming we are working with a single object mask, use the first contour
    cnt = max(contours, key=cv2.contourArea) #contours[0]

    #plt.imshow(mask)
    #plt.show()

    # Perimeter
    perimeter = cv2.arcLength(cnt, True)

    print(perimeter)

    # Area
    area = cv2.contourArea(cnt)

    # Bounding Box
    x, y, w, h = cv2.boundingRect(cnt)

    # Aspect Ratio
    aspect_ratio = float(w) / h

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    return circularity



def elongation(mask):
    print(np.max(mask), np.min(mask))
    m = cv2.moments(mask)
    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
    return (x + y**0.5) / (x - y**0.5)



def compute_diameter(img):
    mask = np.where(img > 0, 1, 0)

    # get distance transform
    distance = mask.copy()
    distance = cv2.distanceTransform(distance, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)

    # get skeleton (medial axis)
    binary = mask.copy()
    binary = binary.astype(np.float32) / 255
    skeleton = skimage.morphology.skeletonize(binary).astype(np.float32)

    # apply skeleton to select center line of distance
    thickness = cv2.multiply(distance, skeleton)

    # get average thickness for non-zero pixels
    average = np.mean(thickness[skeleton != 0])

    # thickness = 2*average
    thick = 2 * average

    return  thick



def compute_area(img):
    #mask = np.where(img[:,:,0] > 0, 1, 0)

    count = cv2.countNonZero(img[:,:, 0])
    return count


def compute_avg_color(img, mask):
    #mask = mask[:,:,0]
    img = img/255. * mask/255.
    img = img[mask[:,:, 0] == 255]

    # img_idx = np.where(np.all(img != [0,0,0]))
    # average = img[img_idx].mean(axis=0)#.mean(axis=0)

    img = img * 255
    average = img.mean(axis=0)#.mean(axis=0)
    average = np.array(average, dtype=np.int64)
    print(average)
    return average #rgbtoint32(average)


def compute_dominant_color(img, mask):
    # mask = mask[:,:,0]
    img = img / 255. * mask / 255.
    img = img[mask[:, :, 0] == 255]


    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    #dominant = palette[np.argmax(counts)]
    dominant = palette[np.argsort(counts)[-2]]
    dominant = np.array(dominant, dtype=np.int64)
    return dominant #rgbtoint32(dominant)


def compute_orientation(img):
    hh, ww, cc = img.shape

    # convert to mask
    thresh = np.where(img[:, :, 0] == 255, 1, 0)

    # find outer contour
    cntrs = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #print(cntrs)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    #print(cntrs)

    # get rotated rectangle from outer contour
    rotrect = cv2.minAreaRect(cntrs[0])
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)


    # get angle from rotated rectangle
    angle = rotrect[-1]
    return angle



def get_representation(dl):
    representation_to_save = None

    print("Saving representation of {} samples (batches) ".format(len(dl)))

    # iterate over the dataset, with sampling
    for i, (image, mask, labels, filename) in enumerate(dl):

        image = image.numpy()* 255.
        image = np.moveaxis(image, 0, -1).astype(np.int64)

        mask = mask.numpy() * 255.
        mask = np.moveaxis(mask, 0, -1).astype(np.int64)

        # update representation list
        old = representation_to_save
        new = [solidity(mask[:,:,0]),compute_avg_color(image, mask), compute_area(mask) ]

        representation_to_save = new if old is None else np.vstack((old, new))

    return representation_to_save


def create_preprocessing_directory(directory):

    process_dir = os.path.join(directory, "SIMPLE_FEATURES")



    # make experiment directory
    if not os.path.exists(process_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(process_dir)
    else:
        raise FileExistsError("Preprocessing folder exists")

    return process_dir


def postprocess_model(directory, args):

    # set fixed seed
    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])

    directory = create_preprocessing_directory(directory)


    train_dataset = get_named_dataset(args["postprocess_dataset"])(split="train", get_filename=True)
    test_dataset = get_named_dataset(args["postprocess_dataset"])(split="test", get_filename=True)

    # split train and validation BALANCED
    train_dataset, val_dataset = train_test_split(train_dataset, test_size= args["perc_val_set"], random_state=args["split_random_seed"], stratify=[ y for x, y in train_dataset.samples])

    representation_to_save = {}
    for split, dl in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):
        if split in ["train", "val"]:
            continue
        rep = get_representation(dl)
        representation_to_save[split] = rep

    # save representation
    np.savez_compressed(os.path.join(directory, "representations"), **representation_to_save)
