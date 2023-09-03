####################################################################################
#   Train_unet.py
#       Script for implementing and training a unet model for segmentation
#   Name: Ngoc Bao Han, Nguyen (Mimi)
#   Student Number: 20188794
#   Date: Mar 19th, 2023
#####################################################################################

import cv2
import numpy 
import os

from scipy.spatial.distance import cdist
from sklearn.metrics import precision_recall_curve, average_precision_score, recall_score, precision_score
import tensorflow
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l1
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
import math
import random
from UnetSequence import UnetSequence
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import directed_hausdorff
from tensorflow.keras.models import load_model

def define_UNet_Architecture(imageSize,numClasses,filterMultiplier=10):
    input_ = layers.Input(imageSize)
    skips = []
    output = input_

    num_layers = int(numpy.floor(numpy.log2(imageSize[0])))
    down_conv_kernel_sizes = numpy.zeros([num_layers],dtype=int)
    up_conv_kernel_sizes = numpy.zeros([num_layers], dtype=int)

    down_filter_numbers = numpy.zeros([num_layers],dtype=int)
    up_filter_numbers = numpy.zeros([num_layers],dtype=int)

    for layer_index in range(num_layers):
        up_conv_kernel_sizes[layer_index]=int(4)
        down_conv_kernel_sizes[layer_index] = int(3)
        down_filter_numbers[layer_index] = int((layer_index+1)*filterMultiplier + numClasses)
        up_filter_numbers[layer_index] = int((num_layers-layer_index-1)*filterMultiplier + numClasses)
        
    #Create contracting path
    for kernel_shape,num_filters in zip(down_conv_kernel_sizes,down_filter_numbers):
        skips.append(output)
        output = layers.Conv2D(num_filters,(kernel_shape,kernel_shape),
                               strides=2,
                               padding="same",
                               activation="relu",
                               bias_regularizer=l1(0.))(output)

    #Create expanding path
    lastLayer = len(up_conv_kernel_sizes)-1
    layerNum = 0
    for kernel_shape,num_filters in zip(up_conv_kernel_sizes,up_filter_numbers):
        output = layers.UpSampling2D()(output)
        skip_connection_output = skips.pop()
        output = layers.concatenate([output,skip_connection_output],axis=3)
        if layerNum!=lastLayer:
            output = layers.Conv2D(num_filters,(kernel_shape,kernel_shape),
                                   padding="same",
                                   activation="relu",
                                   bias_regularizer=l1(0.))(output)
        else: #Final output layer
            output = layers.Conv2D(num_filters, (kernel_shape, kernel_shape),
                                   padding="same",
                                   activation="softmax",
                                   bias_regularizer=l1(0.))(output)
        layerNum+=1
    return Model([input_],[output])

#############################################################################################################
# Question 2:
#    Complete the following function to generate your simulated images and segmentations. You may implement
#    many helper functions as necessary to do so.
#############################################################################################################
"""
    QUESTION 2 HELPER FUNCTIONS: Generate image and depth modification function
"""

def generateImage(height, width, depth):
    """
        This function generate random simulated greyscale images. The image will have
        a black background and containt a white ellipse. The location of the ellipse will
        ensure that it is wtihin the frame.

        Parameters: 
            height - image height
            width - image width
            depth - image depth (input as power of 2)
    """
    color = pow(2, depth) - 1
    image_res = numpy.zeros((height, width))
    # make sure the axis is big enough so the ellipse is visible
    axis_x, axis_y = numpy.random.randint(height/5, height/2), numpy.random.randint(width/5,width/2)
    # make sure the center allows for the ellipse to not overfill outside the frame
    x, y = numpy.random.randint(axis_y, width-axis_y), numpy.random.randint(axis_x, height-axis_x)

    for i in range(height):
        for j in range(width):
            if pow((j - x),2)/pow(axis_y, 2) + pow((i-y), 2)/pow(axis_x, 2) <= 1:
                image_res[i][j] = color
    return image_res

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

def reshape_image(image):
    # Add a third dimension to the grayscale image to make it have shape (128, 128, 1)
    reshaped_image = numpy.expand_dims(image, axis=-1)
    return reshaped_image

def generateDataset(datasetDirectory,num_images,imageSize):
    '''
    Args:
        datasetDirectory: the path to the directory where your images and segmentations will be stored
        num_images: the number of images that you wish to generate
        imageSize: the shape of your images and segmentations
    Returns:
        None: Saves all images and segmentations to the dataset directory
    '''
    for i in range(num_images):
        a,b,_ = imageSize
        # change to 8 because the last argument for generateImage is the depth of the image
        gen_img = generateImage(a, b, 8)
        gen_img = reshape_image(gen_img)
        # save the segmentation with 1-bit depth
        threshold_value = 127
        max_value = 1
        _, ground_truth_seg = cv2.threshold(gen_img, threshold_value, max_value, cv2.THRESH_BINARY)

        cv2.imwrite(os.path.join(datasetDirectory, "segmentation_{}.png".format(i)), ground_truth_seg)

        random = numpy.random.randint(1, 100)
        for j in range(random):
            gen_img = cv2.blur(gen_img, (3, 3))
        cv2.imwrite(os.path.join(datasetDirectory, "image_{}.png".format(i)), gen_img)


#############################################################################################################
# Question 3:
#    Complete the following function so that it returns your data divided into 3 non-overlapping sets
# You do not need to read the images at this stage
#############################################################################################################
def splitDataIntoSets(images,segmentations):
    '''
    Args:
        images: list of all image filepaths in the dataset
        segmentations: list of all segmentation filepaths in the dataset

    Returns:
        trainImg: list of all image filepaths to be used for training
        trainSeg: list of all segmentation filepaths to be used for training
        valImg: list of all image filepaths to be used for validation
        valSeg: list of all segmentation filepaths to be used for validation
        testImg: list of all image filepaths to be used for testing
        testSeg: list of all segmentation filepaths to be used for testing
    '''
    
    # split the data in to 3:1:1 for train, validate, test
    x, testImg, y, testSeg = train_test_split(images, segmentations, test_size = 0.2, train_size = 0.8, random_state= 42)
    trainImg, valImg, trainSeg, valSeg = train_test_split(x, y, test_size = 0.25, train_size = 0.75, random_state= 42)

    return (trainImg,trainSeg),(valImg,valSeg),(testImg,testSeg)

#############################################################################################################
# Question 5:
#    Complete the following function so that it will create a plot for the training and validation loss/metrics.
#    Training and validation should be shown on the same graph so there should be one plot per loss/metric
#############################################################################################################
def plotLossAndMetrics(trainingHistory):
    '''
    Args:
        trainingHistory: The dictionary containing the progression of the loss and metrics for training and validation
    Returns:
        None: should save each graph as a png
    '''
    # Extract the relevant data from the training history dictionary
    training_loss = trainingHistory.history['loss']
    validation_loss = trainingHistory.history['val_loss']
    training_accuracy = trainingHistory.history['accuracy']
    validation_accuracy = trainingHistory.history['val_accuracy']

    # Plot the training and validation loss
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

    # Plot the training and validation accuracy
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    return

# Precision, Recall, F1 helper function
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def main():
    dataSetPath = os.path.join(os.getcwd(), "CISC_472_dataset")
    # generateDataset(dataSetPath, 100,imageSize = (128,128,1)) #this line only needs to be run once

    images = sorted([os.path.join(dataSetPath,x) for x in os.listdir(dataSetPath) if "image" in x])
    segmentations = sorted([os.path.join(dataSetPath,x) for x in os.listdir(dataSetPath) if "segmentation" in x])

    trainData,valData,testData = splitDataIntoSets(images,segmentations)

    trainSequence = UnetSequence(trainData)
    valSequence = UnetSequence(valData)
    testSequence = UnetSequence(testData,shuffle=False)


    unet = define_UNet_Architecture(imageSize=(128,128,1),numClasses=2)
    unet.summary()

    #############################################################################################################
    # Set the values of the following hyperparameters
    #############################################################################################################

    learning_rate = 1e-5
    lossFunction = 'categorical_crossentropy'
    # lossFunction = IOU_Loss
    # lossFunction = DICE_Loss
    # lossFunction = hausdorffDistance
    metrics=["accuracy"]
    optimizer = optimizers.Adam(
        learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
    # optimizer = optimizers.SGD(
    #     learning_rate=0.01, momentum=0.0, nesterov=False)
    # optimizer = optimizers.Adagrad(
    #     learning_rate=0.01, initial_accumulator_value=0.1, epsilon=1e-7)
    # optimizer = optimizers.RMSprop(
    #     learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-7, centered=False)
    numEpochs = 20

    #############################################################################################################
    # Create model checkpoints here, and add the variable names to the callbacks list in the compile command
    #############################################################################################################

    unet.compile(optimizer= optimizer,
                 loss=lossFunction,
                 metrics=metrics,
                 run_eagerly=True)
    
    # using early stopping along with checkpoint
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)

    checkpoint_filepath = 'test.h5' 
    model_checkpoint_callback = ModelCheckpoint(
        filepath = checkpoint_filepath,
        save_weights_only= True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    callbacks_list = [model_checkpoint_callback, early_stopping]
    history = unet.fit(x=trainSequence,
                       validation_data=valSequence,
                       epochs=numEpochs,
                       callbacks=callbacks_list)

    plotLossAndMetrics(history)

    unet.evaluate(testSequence)
    predictions = unet.predict(testSequence)
    
    #############################################################################################################
    # Add additional code for generating predictions and evaluations here
    #############################################################################################################

    predicted_images = numpy.argmax(predictions, axis=-1)
    # Visual inspection of predicted images
    # I ran these during training and on early stopping, thus the images being produced is with the best weights at each given loss function
    # There isn't a need to oepn up the best weights file later on for evaluation since they are all done while training. 

    # for i in range(predicted_images.shape[0]):
    #     plt.imshow(predicted_images[i], cmap='gray')
    #     plt.title('Predicted Image')
    #     plt.show()
    
    # For precision, recall, and F1 score, I add that as another metrics while training the model
    # with different loss functions, thus there isn't a code to run that over the models after they are being trained

# I commented out the given IOU function and created my own since the tensor
# created by the given function alters the prediction images , I rewrite my own underneath!
# def IOU(y_true,y_pred):
#     y_true_f = K.flatten(y_true[:,:,1])
#     y_pred_f = K.flatten(y_pred[:,:,1])
#     intersection = K.sum(y_true_f*y_pred_f)
#     return(intersection)/(K.sum(y_true_f)+K.sum(y_pred_f)-intersection)

# def IOU_Loss(y_true,y_pred):
#     # y_true = tensorflow.cast(y_true, tensorflow.float32)
#     return 1-IOU(y_true,y_pred)

def IOU_Loss(y_true, y_pred): # REWRITTEN IOU_LOSS: similar functionality to the given function, cleaner
    smooth = 1e-5
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return 1 - iou

#############################################################################################################
# Question 7:
#    Complete the following function to compute the mean hausdorff distance
#############################################################################################################
def hausdorffDistance(y_true, y_pred):
    # THIS FUNCTION DOES NOT WORK! I tried 
    
    batch_size = y_true.shape[0]
    num_classes = y_true.shape[-1]
    distances = numpy.zeros((batch_size, num_classes))
    
    for i in range(batch_size):
        for j in range(num_classes):
            true_mask = y_true[i,:,:,j]
            pred_mask = y_pred[i,:,:,j]
            
            # Compute pairwise distances between true and predicted mask pixels
            true_pixels, pred_pixels = numpy.array(numpy.where(true_mask)).T, numpy.array(numpy.where(pred_mask)).T
            pairwise_distances = cdist(true_pixels, pred_pixels)
            
            # Compute Hausdorff distance in both directions
            max_distance_true_to_pred = pairwise_distances.max(axis=1).max()
            max_distance_pred_to_true = pairwise_distances.max(axis=0).max()
            distances[i,j] = max(max_distance_true_to_pred, max_distance_pred_to_true)
    
    # Compute mean Hausdorff distance across all classes and samples in batch
    mean_hausdorff_distance = numpy.mean(distances)
    
    return mean_hausdorff_distance

#############################################################################################################
# Question 7:
#    Complete the following function to compute the mean dice coefficient
#############################################################################################################

def DICE_Loss(y_true, y_pred): # this one does work, phew
    smooth = 1e-5
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    total = K.sum(y_true,-1) + K.sum(y_pred,-1) 
    dice = K.mean(2 * (intersection + smooth) / (total + smooth), axis=0)
    return 1 - dice

main()



