import numpy as np
import matplotlib.pyplot as plt
from utils.crossing_number import calculate_minutiaes
from scipy import ndimage
from skimage.draw import polygon
import skimage.segmentation as seg
import os
import cv2

FOLDER = "retina_database"

def compute_polar_histogram(minutiae):

    # Compute the polar coordinates
    r = np.sqrt(np.sum(minutiae**2, axis=1))
    theta = np.arctan2(minutiae[:,1], minutiae[:,0])

    # Define the number of bins for the polar histogram
    num_bins = 16

    # Compute the histogram
    hist, _ = np.histogram(theta, bins=num_bins, range=(-np.pi, np.pi))
    hist = hist / np.sum(hist)  # normalize the histogram

    return r, hist

def save_template(template, individual):
    path_to_individual = os.path.join(FOLDER, individual)
    np.save(path_to_individual, template)

def compute_template(image):
    template = np.zeros_like(image)
    _, endings, bifurcations = calculate_minutiaes(image.astype('uint8'))
    r_end, hist_end = compute_polar_histogram(np.array(endings))
    r_bif, hist_bif = compute_polar_histogram(np.array(bifurcations))
    template = np.hstack([r_end, hist_end, r_bif, hist_bif])
    #template = np.hstack(endings+bifurcations)
    return template

def get_contour(image):
    init = image*1
    image = image*255

    segmented = seg.chan_vese(image, mu=1.05, lambda1=0.05, lambda2=1.5, tol=0.001, dt=1.5, init_level_set=init, extended_output=False)

    plt.imshow(segmented)
    plt.show()

def get_contour_active(image):
    result = np.zeros_like(image)
    init = np.zeros(image.shape)
    s = np.linspace(0, 2*np.pi, 3000)
    rad = 600
    r = image.shape[0]//2 + rad*np.sin(s)
    c = image.shape[1]//2 + rad*np.cos(s)
    init = np.array([r, c]).T
    snake = seg.active_contour(image*255, init, alpha=0.5, w_edge=0, w_line=1, coordinates='rc')
    for i in range(snake.shape[0]):
        x = int(snake[i, 0])
        y = int(snake[i, 1])
        result[x, y] = 1
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.dilate(result, kernel, iterations=1)
    return result

def compute_template(image):
    image = image.astype('uint8')
    result = get_contour_active(image)
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    # Compute centroid
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    # Compute the distance from the centroid to each point of the contour
    distances = []
    shape_signatura_size = 500
    offset = len(contours[0]) // shape_signatura_size
    for i in range(shape_signatura_size):
        x, y = contours[0][i*offset][0]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        distances.append(dist)
    return np.array(distances)