"""
Functions for Assignment 4 CISC 472
Student name: Ngoc Nguyen
Student id: 20188794

"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from collections import deque
from skimage import measure
import skimage
from skimage.filters import threshold_otsu
import pandas as pd

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

def thresholding_segmentation(image, lower, upper):
    """
        This function create point based segmnetation for the volume with lower and upper threshold

        Parameters:
            volume - 3D numpy volume
            lower - lower bound of threshold
            upper - upper bound of threshold
    """
    segmentation = np.zeros_like(image)

    segmentation[(image >= lower) & (image <= upper)] = 1
    return segmentation

def region_growing_segmentation(image, seeds, threshold):
    """
        This function implements BFS to perform region based segmentation on a volume

        Parameters:
            image - 2D greyscale image
            seeds - list of 2 seeds one for background one for object
            threshold - tolerance for difference
    """
    # implementing BFS to perform region based segmentation
    segmentation = np.zeros_like(image)
    height, width = image.shape
    visited = np.zeros_like(image)
    segmentation[seeds] = 1

    # steps to neighbors (8 connected)
    neighbors = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    # actually only taking the seed of object cause background stay 0 anyway!
    for i in range(2):
        queue = deque()
        queue.append(seeds[i])

        while queue:
            cur_x, cur_y = queue.popleft()

            # list of all possible neighbors 8-connected
            cur_neighbors = []
            for i in range(len(neighbors)):
                cur_neighbors.append([cur_x + neighbors[i][0], cur_y + neighbors[i][1]])

            # keeping the neighbors that are within frame and unsegmented
            valid_neighbors = []
            for neighbor in cur_neighbors:
                if neighbor[0] in range(0, width) and neighbor[1] in range(0, height) and visited[neighbor[0], neighbor[1]] == 0:
                    valid_neighbors.append(neighbor)

            for neighbor in valid_neighbors:
                a, b, = neighbor
                if abs(image[cur_x, cur_y] - image[a, b]) <= threshold:
                    queue.append([a, b])
                    segmentation[a][b] = i
                visited[a][b] = 1

    segmentation = 1 - segmentation
    return segmentation

# Question 3 
def largest_segment(binary_img):
    biggest = np.amax(binary_img)
    if biggest == 0:
        return binary_img

    labels = measure.label(binary_img)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    largest_label = unique_labels[np.argmax(label_counts[1:]) + 1]
    largest_segment = (labels == largest_label).astype(np.uint8)
    
    return largest_segment

def find_contours(binary_image, factor=10):
    # preparing the image before downsampling and get contour
    binary_image = binary_image.astype(np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    downsampeled = np.empty((0, 2), int)

    for contour in contours:
        downsampelled_contour = contour[::factor, 0, :]
        downsampeled = np.append(downsampeled, downsampelled_contour, axis=0)

    return downsampeled


def contour_to_ras(contour, image_name):
    # load the required csv files
    probe_to_ref_timestamps = pd.read_csv("transforms/ProbeToReferenceTimeStamps.csv")
    test_timestamps = pd.read_csv("Test_Images/Case_05/Test_Images_Ultrasound_Labels.csv")

    # extract the time recorded for the given image
    match_img_name = os.path.basename(image_name).replace("_segmentation.png", ".png")
    time_recorded = test_timestamps.loc[test_timestamps["FileName"] == match_img_name, "Time Recorded"].iloc[0]

    # extract the corresponding ProbeToReference transform file path
    file_path = probe_to_ref_timestamps.loc[probe_to_ref_timestamps["Time"] == time_recorded, "Filepath"].iloc[0]
    probe_to_ref = np.load(os.path.join("transforms/ProbeToReference", file_path))

    # load the required transform files
    image_to_probe = np.load("transforms/ImageToProbe.npy")
    reference_to_ras = np.load("transforms/ReferenceToRAS.npy")

    # convert contour points from ijk to ras coordinates
    contour_points_ijk = np.hstack((contour, np.zeros((len(contour), 1)), np.ones((len(contour), 1))))
    ijk_to_ras = np.matmul(np.matmul(np.matmul(contour_points_ijk, image_to_probe), probe_to_ref), reference_to_ras)
    ras_coordinates = ijk_to_ras[:, :3]

    return ras_coordinates

def readImage(fileName):
    img = cv2.imread(fileName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_seeds(img):
    brightest_idx = np.unravel_index(np.argmax(img), img.shape)
    darkest_idx = np.unravel_index(np.argmin(img), img.shape)
    return brightest_idx, darkest_idx

def iou_loss(pred, gt):
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    union_sum = np.sum(union)
    if union_sum == 0:
        iou_score = 0.0
    else:
        iou_score = np.sum(intersection) / union_sum
    return iou_score

def calculate_iou_score(pred_slices, gt_slices):
    iou_score = 0 
    for i in range(len(gt_slices)):
        iou_score += iou_loss(pred_slices[i], gt_slices[i])
    iou_score /= len(gt_slices)
    return iou_score

def accuracy_score(pred, gt):
    num_matches = np.sum(pred == gt)
    num_pixels = pred.size
    accuracy = float(num_matches) / num_pixels
    return accuracy

def calculate_accuracy(pred_slices, gt_slices):
    accuracy = 0
    for i in range(len(gt_slices)):
        accuracy += accuracy_score(pred_slices[i], gt_slices[i])
    accuracy /= len(gt_slices)
    return accuracy


def main():
    dataSetPath = os.path.join(os.getcwd(), "Training_Images")
    dataSetPathTest = os.path.join(os.getcwd(), "Test_Images/Case_05")
    ### LOADING IN TEST IMAGES AND SEGMENTATIONS
    test_img = sorted([os.path.join(dataSetPathTest,x) for x in os.listdir(dataSetPathTest) if "segmentation" not in x and "Labels" not in x])
    test_segmentation = sorted([os.path.join(dataSetPathTest, x) for x in os.listdir(dataSetPathTest) if "segmentation" in x])
    ### LOADING IN TRAINING IMAGES AND SEGMENTATIONS
    cases = ['Case_01', 'Case_02', 'Case_03', 'Case_04']
    train_img, segmentations = [], []
    for i in range(4):
        case_images, segmentations_images = [], []
        path = "{}/{}".format(dataSetPath, cases[i])
        case_images = [os.path.join(path,x) for x in os.listdir(path) if "segmentation" not in x and "Labels" not in x]
        segmentations_images = [os.path.join(path,x) for x in os.listdir(path) if "segmentation" in x]
        train_img += case_images
        segmentations += segmentations_images
    train_img, segmentations = sorted(train_img), sorted(segmentations)
    

    ### QUESTION 2 - UNET, THRESHOLDING AND REGION GROWING SEGMENTATIONS

    # THRESHOLDING SEGMENTATION ON TEST IMAGES
    # trying out different thresholds on training data 
    image = readImage(train_img[i + 32])
    blurred_image = skimage.filters.gaussian(image, sigma=1.0)
    histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))

    ### THIS ONLY NEED TO RUN ONCE - SAVING ALL SEGMENTATIONS CREATED BY THRESHOLD METHOD
    os.makedirs('threshold_segmentation')
    for i in range(len(test_img)):
        image = readImage(test_img[i])
        y_pred = thresholding_segmentation(image, 0, 2)
        np.save("threshold_segmentation/threshold_segmentation_{}".format(i), y_pred)
    
    ## THIS ONLY NEED TO RUN ONCE - SAVING ALL SEGMENTATIONS CREATED BY REGION GROWING METHOD
    ## THIS FUNCTION WILL TAKES A BIT TO RUN - 10 min
    os.makedirs('region_growing_segmentation')
    for i in range(len(test_img)):
        image = readImage(test_img[i])
        a, b = get_seeds(image)
        seed_list = [a, b]
        threshold_region = 20
        y_pred = region_growing_segmentation(image, seed_list, threshold_region)
        np.save("region_growing_segmentation/region_growing_segmentation_{}".format(i), y_pred)

    ### UNET SEGMENTATIONS ARE CREATED IN train_unet.py file

    ### QUESTION 3: APPLYING LARGEST SEGMENT FUNCTION ONTO CREATED SEGMENTATIONS
    data_path = ["unet_segmentation", 'threshold_segmentation', 'region_growing_segmentation']
    data_path_save = ["UNET_segmentation_largest", 'THRESHOLD_segmentation_largest', 'REGION_GROWING_segmentation_largest']
    
    for i in range(3):
        if not os.path.exists(data_path_save[i]):
            os.makedirs(data_path_save[i])

        img_count = 0
        img_list = os.listdir(data_path[i])
        for file in sorted(img_list):
            image = np.load(os.path.join(data_path[i], file))
            biggest_segment = largest_segment(image)
            np.save("{}/{}_{}".format(data_path_save[i], data_path[i], img_count), biggest_segment)
            img_count += 1


    ### QUESTION 4-6: CREATING CONTOURS, TRANSFORMING CONTOURS, SAVE FILES
    unet_files = [os.path.join(data_path_save[0], f) for f in os.listdir(data_path_save[0]) if f.endswith('.npy')]
    threshold_files = [os.path.join(data_path_save[1], f) for f in os.listdir(data_path_save[1]) if f.endswith('.npy')]
    region_growing_files = [os.path.join(data_path_save[2], f) for f in os.listdir(data_path_save[2]) if f.endswith('.npy')]
    
    unet_slices = [np.load(file) for file in unet_files]
    threshold_slices = [np.load(file) for file in threshold_files]
    region_growing_slices = [np.load(file) for file in region_growing_files]
    ground_truth_slices = [readImage(file) for file in test_segmentation]

    volume_name = ['unetContours', 'thresholdingContours', 'regionGrowingContours', 'groundTruthContours.npy']
    slices = [unet_slices, threshold_slices, region_growing_slices, ground_truth_slices]

    for i in range(4):
        img_count = 0
        volume = np.zeros((0, 3))
        for slice_img in slices[i]:
            contour = find_contours(slice_img) # question 4 function 
            if len(contour) != 0:
                ras_points = contour_to_ras(contour, test_img[img_count]) # question 5 function
                img_count += 1
                volume = np.concatenate((volume, ras_points), axis=0)

        print(volume_name[i], volume.shape)
        np.save(volume_name[i], volume)
    
    ### QUESTION 8 - calculating the IoU metric for generated volumes

    print('threshold IOU score', calculate_iou_score(threshold_slices, ground_truth_slices))
    print('region growing IOU score', calculate_iou_score(region_growing_slices, ground_truth_slices))
    print('unet IOU score', calculate_iou_score(unet_slices, ground_truth_slices))

    print('threshold accuracy score', calculate_accuracy(threshold_slices, ground_truth_slices))
    print('region growing accuracy score', calculate_accuracy(region_growing_slices, ground_truth_slices))
    print('unet accuracy score', calculate_accuracy(unet_slices, ground_truth_slices))
main()