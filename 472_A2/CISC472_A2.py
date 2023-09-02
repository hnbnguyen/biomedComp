import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from collections import deque
import SimpleITK as itk 

### FUNCTIONS FROM ASSIGNMENT 1 (MODIFIED)

def generate_helper(lower, upper, con1, con2, x):
    """
        This function helps generate the center and size of ellipse with conditions
        to make sure the ellipse will stay within the frame even moved in different directions

        Parameters: 
            lower - lowest possible value the size can alter to
            upper - highest possible value the size can alter to
            con1 - condition for lower bound to keep ellipse not overflowing to the left
            con2 - condition for upper bound to keep ellipse not overflowing to the right
            x - old value to be modified
    """
    if lower < con1:
        x = int(upper)
    elif upper > con2:
        x = random.randint(lower, x)
    else:
        x = random.randint(lower, upper)
    return x

def generate_center(width, height, axis_1, axis_2, x, y):
    """
        This function generate a new center for the ellipse

        Parameters:
            width - width of the frame
            height - height of the frame
            axis_1 - ellipse radius along y axis
            axis_2 - ellipse radius along x axis
            x - previous x center
            y - previous y center
    """
    x = generate_helper(x - 5, x + 5, axis_2, width - axis_2, x)
    y = generate_helper(y - 5, y + 5, axis_1, height - axis_1, y)

    return x, y 

def generate_ellipse_axis(width, height, ellipse_width, ellipse_height):
    """
        This function generate new ellipse radius along x and y axis

        Parameters:
            width - width of the frame
            height - height of the frame
            ellipse_width - previous ellipse radius along x axis
            ellipse_height - previous ellipse radius along y axis
    """
    ellipse_width = generate_helper(ellipse_width * 0.95, ellipse_width * 1.05 + 1, int(width/5), width / 2, ellipse_width)
    ellipse_height = generate_helper(ellipse_height * 0.95, ellipse_height * 1.05 + 1, int(height/5), height / 2, ellipse_height)

    return ellipse_height, ellipse_width

def generate_image(width, height, depth, ellipse_width, ellipse_height, x, y):
    """
        This function generate random simulated greyscale images. The image will have
        a black background and containt a white ellipse. The location of the ellipse will
        ensure that it is wtihin the frame.

        Parameters: 
            height - image height
            width - image width
            depth - image depth (input as power of 2)
            ellipse_width - previous ellipse radius along x axis
            ellipse_height - previous ellipse radius along y axis
            x - x center for ellipse
            y - y center for ellipse

    """
    color = pow(2, depth) - 1
    image_res = np.zeros((height, width))
    # make sure the axis is big enough so the ellipse is visible
    # axis_x, axis_y = random.randint(height/5, height/2), random.randint(width/5,width/2)
    axis_1, axis_2 = generate_ellipse_axis(width, height, ellipse_width, ellipse_height)
    # make sure the center allows for the ellipse to not overfill outside the frame
    x, y = generate_center(width, height, axis_1, axis_2, x, y)
    print('x: {}, y: {}, axis_width: {}, axis_height: {}'.format(x, y, axis_2, axis_1))
    
    for i in range(height):
        for j in range(width):
            if pow((j - x),2)/pow(axis_2, 2) + pow((i-y), 2)/pow(axis_1, 2) <= 1:
                image_res[i][j] = color
    return image_res, axis_2, axis_1, x, y

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

def linear_fiter(filter, image):
    """
        This function applies a square kernel to an image with a linear function.
        It returns the result as an image array.

        Parameters:
            filter - square array filter
            image - image array
    """
    filter_size = filter.shape[0]
    padded_image = np.pad(image, filter_size // 2, mode='constant')
    result = np.zeros(image.shape)

    for row in range(result.shape[0]):
        for col in range(result.shape[1]):
            cur = padded_image[row:row+filter_size, col:col+filter_size]
            result[row, col] = (cur * filter).sum()
    return result


### FUNCTIONS FOR ASSIGNMENT 2

# Question 1: Tumor simulation
def tumor_simulation(smoothing_filter, ellipse_height, ellipse_width, x, y):
    """
        This function create a simulate tumor size (224, 224, 224)

        Parameters 
            smoothing_filter - smoothing filter
            ellipse_width - previous ellipse radius along x axis
            ellipse_height - previous ellipse radius along y axis
            x - x center for ellipse
            y - y center for ellipse
    """
    volume = np.zeros((224, 224, 224), dtype=np.uint8)
    
    for z in range(224):
        gen_img, ellipse_width, ellipse_height, x, y = generate_image(224, 224, 8, ellipse_width, ellipse_height, x, y)
        
        # applying smoothing filter multiple times to distort the edges
        for i in range(10):
            gen_img = linear_fiter(smoothing_filter, gen_img)
        
        volume[z] = gen_img.astype(np.uint8)
    
    np.save('simulatedTumor.npy', volume)
    return volume
    
# Question 2: Volume reslicing
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

# Question 3: Rendering 
# Function 1: Create maximum intesity projection image
def create_mip(volume, axis = 0):
    """
        This function create a MIP for a slice depends on the axis

        Parameters:
            volume - 3D numpy volume 
            axis - axis definition: 0 is axial, 1 is sagittal, 2 is coronal
    """
    # modify the volume to match the viewing plane 
    mip = np.amax(volume, axis= axis)
    return mip

# Function 2: 
def create_ddr(volume, axis):
    """
        This function create a DDR for a slice depends on the axis

        Parameters:
            volume - 3D numpy volume 
            axis - axis definition: 0 is axial, 1 is sagittal, 2 is coronal
    """
    projection = np.sum(volume, axis = axis)
    return projection

# Question 4: Point based segmentation
def point_based_segmentation(image, lower, upper):
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

# Question 5
def region_based_segmentation(image, seeds, threshold):
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

def resample_to_isotropic(volume):
    """
        This function converts volume to (1.0, 1.0, 1.0) scale using ITK

        Parameters:
            volume - 3D numpy volume
    """
    # convert the numpy volume to a SimpleITK image
    itk_image = itk.GetImageFromArray(volume)
    isotropic_spacing= (1.0, 1.0, 1.0)
    original_spacing = (0.9375000000000001, 0.9375000000000001, 1.4000000000000001)
    itk_image.SetSpacing(original_spacing)

    # calculate the new size of the volume based on the desired spacing
    original_size = itk_image.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, isotropic_spacing)]

    # create a new identity transformation and resample image
    identity_transform = itk.Transform()
    resampling_method = itk.sitkLinear
    resampled_image = itk.Resample(itk_image, new_size, identity_transform, resampling_method, itk_image.GetOrigin(), isotropic_spacing, itk_image.GetDirection(), 0, itk_image.GetPixelID())
    resampled_volume = itk.GetArrayFromImage(resampled_image)
    return resampled_volume

if __name__ == '__main__':
    # QUESTION 1
    smoothing_filter = (1/10) * np.array([
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1]
    ])
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
    ellipse_height, ellipse_width = random.randint(224/5, 224/3), random.randint(224/5, 224/3)
    x, y = random.randint(ellipse_width, 224-ellipse_width), random.randint(ellipse_height, 224-ellipse_height)
    # (below) Already run tumor simulation function to generate volume file
    # volume = tumor_simulation(smoothing_filter, ellipse_height, ellipse_width, x, y)
    
    # QUESTION 2
    # Open the volume created in question 1
    generated_volume = np.load('simulatedTumor.npy')
    axial = volume_reslice(generated_volume, 0, 112)
    sagittal = volume_reslice(generated_volume, 1, 112)
    coronal = volume_reslice(generated_volume, 1, 112)

    show_image(axial, "Axial center slice")
    show_image(sagittal, "Sagittal center slice")
    show_image(coronal, "Coronal center slice")

    # QUESTION 3
    tumor_axial_mip = create_mip(generated_volume, 0)
    show_image(tumor_axial_mip, "Axial MIP")
    tumor_sagittal_mip = create_mip(generated_volume, 1)
    show_image(tumor_sagittal_mip, "Sagittal MIP")
    tumor_coronal_mip = create_mip(generated_volume, 2)
    show_image(tumor_coronal_mip, "Coronal MIP")

    tumor_axial_ddr = create_ddr(generated_volume, 0)
    show_image(tumor_axial_ddr, "Axial DDR")
    tumor_sagittal_ddr = create_ddr(generated_volume, 1)
    show_image(tumor_sagittal_ddr, "Sagittal DDR")
    tumor_coronal_ddr = create_ddr(generated_volume, 2)
    show_image(tumor_coronal_ddr, "Coronal DDR")

    # QUESTION 4
    axial_pb_segmentation = point_based_segmentation(axial, 100, 200)
    sagittal_pb_segmentation = point_based_segmentation(sagittal, 100, 200)
    coronal_pb_segmentation = point_based_segmentation(coronal, 100, 200)
    show_image(axial_pb_segmentation, "Axial point based segmentation")
    show_image(sagittal_pb_segmentation, "Sagittal point based segmentation")
    show_image(coronal_pb_segmentation, "Coronal point based segmentation")

    # QUESTION 5
    show_image(axial, "Axial center slice")
    axial_region_segmentation = region_based_segmentation(axial, [[200, 200], [66, 55]], 100)
    show_image(axial_region_segmentation, "Axial region segmentation threshold 100")
    
    # QUESTION 6
    brain_tumor = np.load("MRBrainTumour.npy")
    brain_tumor_resampled = resample_to_isotropic(brain_tumor)
    np.save('resampled_MRBrainTumor.npy', brain_tumor_resampled)

    brain_tumor_axial = volume_reslice(brain_tumor_resampled, 0, 118)
    brain_tumor_sagittal = volume_reslice(brain_tumor_resampled, 1, 100)
    brain_tumor_coronal = volume_reslice(brain_tumor_resampled, 2, 130) 

    # Part a
    show_image(brain_tumor_axial, 'Brain tumor axial')
    show_image(brain_tumor_sagittal, 'Brain tumor sagittal')
    show_image(brain_tumor_coronal, 'Brain tumor coronal')

    # Part b
    brain_tumor_axial_mip = create_mip(brain_tumor_resampled, 0)
    brain_tumor_sagittal_mip = create_mip(brain_tumor_resampled, 1)
    brain_tumor_coronal_mip = create_mip(brain_tumor_resampled, 2)
    show_image(brain_tumor_axial_mip, "Brain tumor axial MIP")
    show_image(brain_tumor_sagittal_mip, "Brain tumor sagittal MIP")
    show_image(brain_tumor_coronal_mip, "Brain tumor coronal MIP")

    brain_tumor_axial_ddr= create_ddr(brain_tumor, 0)
    brain_tumor_sagittal_ddr = create_ddr(brain_tumor, 1)
    brain_tumor_coronal_ddr = create_ddr(brain_tumor, 2)
    show_image(brain_tumor_axial_ddr, "Brain tumor axial DDR")
    show_image(brain_tumor_sagittal_ddr, "Brain tumor sagittal DDR")
    show_image(brain_tumor_coronal_ddr, "Brain tumor coronal DDR")

    # Part c - testing with axial slice
    lower, upper = 193, 255
    brain_axial_pb_segmentation = point_based_segmentation(brain_tumor_axial, lower, upper)
    show_image(brain_axial_pb_segmentation, "Brain tumor axial point based segmentation [{}, {}]".format(lower, upper))

    seed_list = [[10, 10], [130, 96]]
    threshold_region = 30
    brain_axial_region_segmentation = region_based_segmentation(brain_tumor_axial, seed_list, threshold_region)
    show_image(brain_axial_region_segmentation, 'Brain tumor region segmentation threshold {}'.format(threshold_region))

    # smoothing out brain MRI
    for i in range(30):
        brain_tumor_axial =linear_fiter(smoothing_filter, brain_tumor_axial)   
    show_image(brain_tumor_axial, "Smoothed before segmentation")
    lower_pb, upper_pb = 193, 255
    brain_axial_pb_smooth = point_based_segmentation(brain_tumor_axial, lower_pb, upper_pb)
    show_image(brain_axial_pb_smooth, "Brain tumor axial point based segmentation [{}, {}]".format(lower_pb, upper_pb))
