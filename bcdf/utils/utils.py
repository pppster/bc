from glob import glob
import pandas as pd
import numpy as np

from bci.core.bcimage import BCImage
from bci.utils.bcimageenum import BCImageEnum, IMAGE, MASK, NOBACKGROUND
import cv2


def remove_datatype(filename: str) -> str:
    return filename[:filename.rfind('.')]


def just_image_names(dir: str) -> list:
    all_files = glob(dir)
    filtered = [file for file in all_files if "mask" not in file]
    filtered = [file for file in filtered if "nobackground" not in file]
    return sorted(filtered)


def _create_labeled_dataframe(dir: list) -> pd.DataFrame:
    dataframe = pd.DataFrame(columns=['Name', 'Label'])
    for path in dir:
        label = path.split('/')[-2]
        filename = path.split('/')[-1]
        name = filename[:filename.rfind('.')]
        if not any(dataframe['Name'] == name):
            dataframe.loc[len(dataframe)] = [name, label]
    return dataframe


def return_one(img: BCImage, image_type: BCImageEnum = IMAGE):
    return 1


def get_aspect_ratio(img: BCImage, image_type: BCImageEnum = MASK):
    contours, hierarchy = cv2.findContours(img[image_type], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    ratio = float(w)/h
    return ratio


def get_area(img: BCImage, image_type: BCImageEnum = MASK):
    contours, hierarchy = cv2.findContours(img[image_type], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    return area


def get_perimeter(img: BCImage, image_type: BCImageEnum = MASK):
    contours, hierarchy = cv2.findContours(img[image_type], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=False)
    return perimeter


def get_centroid_x(img: BCImage, image_type: BCImageEnum = MASK):
    contours, hierarchy = cv2.findContours(img[image_type], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    centroid_x = int(moments['m10'] / moments['m00'])
    return centroid_x


def get_num_corners(img: BCImage, image_type: BCImageEnum = MASK):
    corners = cv2.goodFeaturesToTrack(image=img[image_type], maxCorners=100, qualityLevel=0.01, minDistance=5)
    return len(corners)


def get_extend(img: BCImage, image_type: BCImageEnum = MASK):
    contours, hierarchy = cv2.findContours(img[image_type], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = float(area) / rect_area
    return extent


def get_solidity(img: BCImage, image_type: BCImageEnum = MASK):
    contours, hierarchy = cv2.findContours(img[image_type], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return solidity

def get_mean_brown(img: BCImage, image_type: BCImageEnum = NOBACKGROUND):
    gray = cv2.cvtColor(img[image_type], cv2.COLOR_BGR2GRAY)
    gray[gray > 30] = 0
    mean_brown = np.mean(gray[gray != 0])
    return mean_brown


def get_mean_green(img: BCImage, image_type: BCImageEnum = NOBACKGROUND):
    gray = cv2.cvtColor(img[image_type], cv2.COLOR_BGR2GRAY)
    gray[gray < 45] = 0
    gray[gray > 100] = 0
    mean_green = np.mean(gray[gray != 0])
    return mean_green

def get_mean_other(img: BCImage, image_type: BCImageEnum = NOBACKGROUND):
    gray = cv2.cvtColor(img[image_type], cv2.COLOR_BGR2GRAY)
    mean_other = np.mean(gray[gray != 0])
    return mean_other




if __name__ == '__main__':
    print('Just bcdataframe util functions')

