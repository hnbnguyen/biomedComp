import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


"""
    Importing functions from A1 & A2 
"""
def show_image(image, name = "Image"):
    """
        This function displays an image. It doesn't return anything.

        Parameters:
            name - title of the image
            image - image array
    """
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap= 'gray')
    plt.title(name)
    plt.show()

def volume_reslice(volume, axis, slice_location):
    """
        This function creates volume slice basded on the axis and the location 

        Parameters:
            volume - 3D numpy volume 
            axis - axis 0 - axial, 1 - sagittal, 2 - coronal
            slice_location - location of the slice
    """
    if axis == 0:   
        res_slice = volume[slice_location, :, :]
    elif axis == 1:
        res_slice = np.rot90(volume[:, slice_location, :])
    elif axis == 2:
        res_slice = np.rot90(volume[:, :, slice_location])
    return res_slice

def readImage(fileName):
    '''
    Args:
        fileName: The path to an image file
    Returns:
        img: your image as a numpy array. shape=(128,128,1)
    '''
    # the image starts with (128, 128, 3) dimension, needs to convert it to (128, 128, 1) as well as 
    # some preprocessing
    
    img = cv2.imread(fileName)
    img = np.mean(img, axis=2, keepdims=True)
    # threshold
    img_res = np.zeros((128, 128, 1))
    img_res[(img >= 230) & (img <= 255)] = 1
    return img_res


if __name__ == "__main__":
    # QUESTION 1 - Data exploration
    exploration = ["TrainingData/Case00.npy", "TrainingData/Case00_segmentation.npy"]
    tag = ["MRI Scan", "Segmentation"]

    for i in range(2):
        q1_volume = np.load(exploration[i])
        volume_shape = list(q1_volume.shape)
        mid_slice = [int(volume_shape[i]/2) for i in range(3)]
        q1_axial = volume_reslice(q1_volume, 0, mid_slice[0])
        q1_sag = volume_reslice(q1_volume, 1, mid_slice[1])
        q1_cor = volume_reslice(q1_volume, 2, mid_slice[2])

        show_image(q1_axial, "Case00 Axial Slice {}".format(tag[i]))
        show_image(q1_sag, "Case00 Sagittal Slice {}".format(tag[i]))
        show_image(q1_cor, "Case00 Coronal Slice {}".format(tag[i]))

    onlyfiles = [f for f in listdir('TrainingData')]
    onlyfiles = sorted(onlyfiles)
    all_shapes = set()
    for i in range(100):
        volume = np.load('TrainingData/{}'.format(onlyfiles[i]))
        all_shapes.add(volume.shape)
    print(all_shapes)

        



