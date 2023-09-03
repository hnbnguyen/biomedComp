####################################################################################
#   UnetSequence.py
#       Script for implementing a custom generator for the unet model defined in
#       Train_unet.py. This script is responsible for efficiently loading and preprocessing data
#       to save on memory during training.
#   Name:
#   Student Number: 20188794 
#   Date: Mar 19th, 2023
#####################################################################################
import numpy
import os
import cv2
import math
import gc
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence, to_categorical
import tensorflow

class UnetSequence(Sequence):
    def __init__(self,images_and_segmentations,batchSize=8,shuffle=True):
        self.inputs = images_and_segmentations[0]
        self.targets = images_and_segmentations[1]
        self.shuffle = shuffle
        self.batch_size=batchSize
        self.on_epoch_end()

    def __len__(self):
        length = len(self.inputs)/self.batch_size
        return math.ceil(length)

    def on_epoch_end(self):
        if self.shuffle:
            self.inputs,self.targets = shuffle(self.inputs,self.targets)
        gc.collect()

    #############################################################################################################
    # Question 4:
    #    Complete the following function that will read in your image. Include any preprocessing that you wish to
    #    perform on your images here. Document in your PDF what enhancement techniques you chose (or why you chose
    #    not to use any), and why.
    #############################################################################################################
    def readImage(self,fileName):
        '''
        Args:
            fileName: The path to an image file
        Returns:
            img: your image as a numpy array. shape=(128,128,1)
        '''
        # img = cv2.imread(fileName)
        # # reduce dimention from (128, 128, 3) to (128, 128, 1)
        # img = numpy.mean(img, axis=2, keepdims=True)
        # # thresholding
        # img_res = numpy.zeros((512, 512, 1))
        # img_res[(img >= 230) & (img <= 255)] = 1

        img = cv2.imread(fileName)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    #############################################################################################################
    # Question 4:
    #    Complete the following function that will read in your ground truth segmentation.
    #############################################################################################################
    def readSegmentation(self,fileName):
        '''
        Args:
            fileName: The path to a segmentation file
        Returns:
            one_hot_img: your segmentation as a numpy array. shape=(128,128,2)
        '''
        # segmentation = cv2.imread(fileName)
        # # # Create an empty numpy array for the one-hot encoded representation
        # # # to categorical
        # one_hot_img = numpy.zeros((512, 512, 2), dtype=numpy.uint8)
        # a, b, _ = segmentation.shape
        # for i in range(a):
        #     for j in range(b):
        #         if numpy.array_equal(segmentation[i][j], [0, 0, 0]):
        #             one_hot_img[i][j] = [1, 0]
        #         else:
        #             one_hot_img[i][j] = [0, 1]

        # segmentation = cv2.imread(fileName)
        # one_hot_img = numpy.zeros((512, 512, 2), dtype=numpy.uint8)
        # mask = (segmentation == [0, 0, 0]).all(axis=2)
        # one_hot_img[..., 0] = mask
        # one_hot_img[..., 1] = ~mask
        # # if i don't cast to float32, the IOU loss function won't run properly
        # one_hot_img = tensorflow.cast(one_hot_img, tensorflow.float32)

        img = cv2.imread(fileName)
        img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(float) / 255
        one_hot_img = to_categorical(img, num_classes= 2)

        return one_hot_img

    def __getitem__(self, index):
        startIndex = index*self.batch_size
        index_of_next_batch = (index+1)*self.batch_size
        inputBatch = [self.readImage(x) for x in self.inputs[startIndex:index_of_next_batch]]
        outputBatch = [self.readSegmentation(x) for x in self.targets[startIndex:index_of_next_batch]]
        return (numpy.array(inputBatch),numpy.array(outputBatch))

