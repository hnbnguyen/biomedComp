"""
    CISC 472 Assignment 1
    Student: Ngoc Nguyen (20188794)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import median, random

# Question 1: Read Image
def readImage(path):
    """
        This function determine if an image is a color image or greyscale image.
        If it is a color image, the function returns a 3 channel image, else it
        will return a 1 channel image.

        Parameters: 
            path - path to image file
    """
    image = cv2.imread(path)
    w, h, _ = image.shape
    flag = True # default assume image is grey scale
    for i in range(w):
        for j in range(h):
            r, g, b = image[i][j]
            # if the r,g,b channel has different value -> color, early stop
            if r != g != b:
                flag = False
                break
    if flag:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(path, cv2.IMREAD_COLOR)

# Question 2: Show Image
def showImage(name, image):
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
    
# Question 3: Linear filter for greyscale image
def linearFilter(filter, image):
    # add padding 
    """
        This function applies a square kernel to an image with a linear function.
        It returns the result as an image array.

        Parameters:
            filter - square array filter
            image - image array
    """
    image_h, image_w = image.shape
    filter_h, filter_w = filter.shape

    w_start, h_start = filter_w//2, filter_h//2
    image_res = np.array(image)

    for i in range(h_start, image_h - h_start):
        for j in range(w_start, image_w - w_start):
            res_sum = 0
            for k in range(filter_h):
                for l in range(filter_w):
                    res_sum += filter[k][l] * image[i - h_start + k][j - w_start + l]
            image_res[i][j] = res_sum
    return image_res

# Question 4: Non-linear filter (bluring function)
def medianFilter(image, filter_size):
    """
        This function applies median filtering to an image, taking in a filter and 
        an image and return the result as an array.

        Parameters: 
            image - image array
            fitler_size - size of a square filter
    """
    image_res = np.array(image)
    filter_count = filter_size * filter_size
    filter = np.ones((filter_size, filter_size)) * 1/filter_count 
    image_h, image_w = image.shape
    h_start = w_start = filter_size//2

    for i in range(h_start, image_h - h_start):
        for j in range(w_start, image_w - w_start):
            res_median = 0
            arr_median = []
            for k in range(filter_size):
                for l in range(filter_size):
                    arr_median.append(image[i - h_start + k][j - w_start + l])
            image_res[i][j] = median(arr_median)
    return image_res

# Question 5: Depth modification 
def depthModification(image, depth):
    """
        This function modifies the depth of an image using a linear function.
        It takes an image and a new depth, and return the modified image as an array.

        Parameters:
            image - image array
            depth - new desired depth (input as power of 2)
    """

    new_depth = pow(2, depth) - 1 # expecting input to be multiples of 2
    image_res  = image.astype(float)/image.max() 
    image_res = image_res * new_depth
    h, w = image.shape
    # value outside of interval [0, new_depth] are cut to stay within the interval
    for i in range(h):
        for j in range(w):
            if image_res[i][j] < 0:
                image_res[i][j] = 0
            elif image_res[i][j] > new_depth:
                image_res[i][j] = new_depth
            image_res[i][j] = image_res[i][j].round() # round to the nearest int
    image_res = image_res.astype(np.uint16)
    return image_res

# Question 6: Contrast enhancement
def enhanceContrast(image, lower, upper):
    """
        This function implement a constrast enhancement method as shown in class. The 
        function takes in the image to be modified and 2 values defining the range for
        enhancement. 

        Parameters:
            image - image array
            lower - lower bound of the range (input in ratio to max value)
            upper - upper bound of the rang (input in ratio to max value)
    """
    max_depth = image.max()
    l1, l2 = max_depth * lower, max_depth * upper
    print(max_depth, l1, l2)
    image_res = np.array(image)
    # getting dimension of image
    image_h, image_w = image.shape
    for i in range(image_h):
        for j in range(image_w):
            if image[i][j] < l1:
                image_res[i][j] = 0
            elif l1 <= image[i][j] and image[i][j] < l2:
                image_res[i][j] = max_depth * (image[i][j] - l1)/(l2-l1)
            else:
                image_res[i][j] = max_depth
    return image_res

# Question 7: Generate a simulated image
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
    image_res = np.zeros((height, width))
    # make sure the axis is big enough so the ellipse is visible
    axis_x, axis_y = random.randint(height/5, height/2), random.randint(width/5,width/2)
    print('length of axis', axis_x, axis_y)
    # make sure the center allows for the ellipse to not overfill outside the frame
    x, y = random.randint(axis_y, width-axis_y), random.randint(axis_x, height-axis_x)
    print('x and y', x, y)

    for i in range(height):
        for j in range(width):
            if pow((j - x),2)/pow(axis_y, 2) + pow((i-y), 2)/pow(axis_x, 2) <= 1:
                image_res[i][j] = color
    return image_res


if __name__ == "__main__":
    # TESTING CODE
    color_pic_path = 'color_puppy.jpeg'
    grey_pic_path = 'grey_scale_puppy.jpeg'
    
    # QUESTION 1 
    color_pic = readImage(color_pic_path)
    grey_pic = readImage(grey_pic_path)
    print('color pic shape', color_pic.shape)
    print('grey pic shape', grey_pic.shape)

    # QUESTION 2 
    showImage('Color picture', color_pic)
    showImage('Grey picture', grey_pic)

    # QUESTION 3 
    smooth_kernel = np.array([
        [1, 1, 1], 
        [1, 2, 1],
        [1, 1, 1]
    ]) * 1/10
    sharp_kernel = np.array([
        [-1, -1, -1], 
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    edge_kernel = np.array([
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0]
    ])
    q3_smooth_pic = linearFilter(smooth_kernel, grey_pic)
    showImage('Grey picture with smoothing filter', q3_smooth_pic)
    q3_sharp_pic = linearFilter(sharp_kernel,grey_pic)
    showImage('Grey picture with sharp filter', q3_sharp_pic)
    q3_edge_pic = linearFilter(edge_kernel, grey_pic)
    showImage('Grey picture with edge filter', q3_edge_pic)
    
    # QUESTION 4
    q4_blur_pic = medianFilter(grey_pic, 9)
    print('showing blurred pic')
    showImage('Grey picture with median filter', q4_blur_pic)

    # QUESTION 5
    q5_depth_mod = depthModification(grey_pic, 16)
    showImage('Grey Image with Depth Modification to 16 bit', q5_depth_mod)
    # modify to CT image depth (12 bit depth)
    q5_depth_mod_ct = depthModification(grey_pic, 12)
    q5_depth_mod_2 = depthModification(grey_pic, 2)
    q5_depth_mod_4 = depthModification(grey_pic, 4)
    showImage('Grey Image with Depth Modification to CT depth (12 bit)', q5_depth_mod_ct)
    showImage('Grey Image with Depth Modification to 2 bit', q5_depth_mod_2)
    showImage('Grey Image with Depth Modification to 4 bit', q5_depth_mod_4)

    # QUESTION 6
    q6_contrast_t1 = enhanceContrast(grey_pic, 0.25, 0.75)
    q6_contrast_t2 = enhanceContrast(grey_pic, 0.45, 0.55)
    q6_contrast_t3 = enhanceContrast(grey_pic, 0.10, 0.60)
    q6_contrast_t4 = enhanceContrast(grey_pic, 0.50, 0.80)
    showImage('Grey Image with Enhanced Contrast (L1 = 0.25d, L2 = 0.75d)', q6_contrast_t1)
    showImage('Grey Image with Enhanced Contrast (L1 = 0.45d, L2 = 0.55d)', q6_contrast_t2)
    showImage('Grey Image with Enhanced Contrast (L1 = 0.10d, L2 = 0.60d)', q6_contrast_t3)
    showImage('Grey Image with Enhanced Contrast (L1 = 0.50d, L2 = 0.80d)', q6_contrast_t4)

    # QUESTION 7
    q7_gen = generateImage(500,500,8)
    showImage('Greyscale image with white ellipse', q7_gen)

    q7_median_3 = medianFilter(q7_gen, 3)
    q7_median_7 = medianFilter(q7_gen, 7)
    q7_median_15 = medianFilter(q7_gen, 15)
    showImage('Generated image with median filter size 3', q7_median_3)
    showImage('Generated image with median filter size 7', q7_median_7)
    showImage('Generated image with median filter size 15', q7_median_15)

    q7_median_5 = medianFilter(q7_gen, 5)
    showImage('Greyscale image with white ellipse filter 5', q7_median_5)

    # for i in range(100):
    #     q7_median_5 = medianFilter(q7_median_5, 5)
    # showImage('Greyscale image with white ellipse filter 5 for 100 times', q7_median_5)