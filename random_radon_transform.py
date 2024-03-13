import numpy as np
import matplotlib.pyplot as plt
import math
from random import randint
import scipy.cluster.hierarchy as hcluster


def get_nonzero(image):
    '''
    Get index pairs of non-white pixels.
    '''
    nonzero_x = np.nonzero(255 - image)[0]
    nonzero_y = np.nonzero(255 - image)[1]
    pairs = np.vstack((nonzero_x, nonzero_y)).T
    return pairs


def choice(pixels):
    '''
    Randomly choose two items at different indices from given array.
    '''
    n = len(pixels)
    idx1 = randint(0, n - 1)
    pixel1 = pixels[idx1]
    idx2 = randint(0, n - 1)
    if (idx2 == idx2):
        idx2 = (idx2 + 1) % n
    pixel2 = pixels[idx2]
    return (pixel1, pixel2)


def random_radon_transformation(image, rho_steps = 220, theta_steps = 440, n_iter = int(1e5)):
    '''
    Apply Radon transform to a black-and-white image by sampling random pairs of points.

    Rho denotes the offset from coordinates' origin, Theta denotes slope angle [0;pi/2]
    '''
    nonzero_pixels = get_nonzero(image)

    height = image.shape[1]
    width = image.shape[0]

    n_rhos = 2 * rho_steps
    n_thetas = theta_steps

    rho_max = np.sqrt(height ** 2 + width ** 2)
    rho_step = rho_max / rho_steps
    theta_step = np.pi / n_thetas
    
    transformed = np.zeros((n_rhos, n_thetas), dtype='float64')
    for i in range(n_iter):
        # get two unique random points
        pixel1, pixel2 = choice(nonzero_pixels)

        x_1 = pixel1[0]
        y_1 = pixel1[1]
        x_2 = pixel2[0]
        y_2 = pixel2[1]

        # consider a line connecting them; represent it as (rho, theta) pair and increment respective accumulator
        theta = np.pi / 2
        if (y_1 != y_2):
            theta = np.arctan((x_2 - x_1) / (y_1 - y_2))
        if (theta < 0):
            theta = np.pi + theta
        rho = x_1 * np.cos(theta) + y_1 * np.sin(theta)

        rho += rho_max 

        theta_int = int(theta / theta_step)
        rho_int = int(rho / rho_step)
        transformed[rho_int][theta_int] += 1

    transf_max = np.max(np.unique(transformed))
    transformed *= (255/transf_max)
    return transformed


def get_clusters(cluster_thresh, radon, nonzero_thresh=1e-3):
    '''
    Group Radon transform output by clusters
    '''
    rad_width  = radon.shape[0]
    rad_height = radon.shape[1]

    # Generate (x, y, value) triples for each 
    coord_val = np.array([[ (i, j, radon[i][j] / 255) for j in range(0, rad_height)] for i in range(0, rad_width)])
    coord_val = list(coord_val.reshape(rad_width * rad_height, 3))
    # Exclude points with zero or close to zero values
    coord_val = np.array(list(filter(lambda x: x[2] > nonzero_thresh, coord_val)))

    if (coord_val.shape[0] < 2):
        return [coord_val]
    # assign each point to a cluster; thresh denotes maximum cluster radius
    clusters = hcluster.fclusterdata(coord_val, cluster_thresh, criterion="distance")
    clusters_types = set(clusters)

    # group points by clusters
    for i in range(0, len(coord_val)):
        coord_val[i][2] = clusters[i]
    final_clusters = []
    for i in clusters_types:
        filtered = np.array(list(filter(lambda x: x[2] == i, coord_val)))
        final_clusters.append(filtered)
    return final_clusters


def get_lines(clusters):
    '''
    Detect lines on clusterized Radon transform output
    '''
    # height = img.shape[1]
    # width = img.shape[0]
    # rho_max = np.sqrt(height ** 2 + width ** 2)
    # n_thetas = 2 * steps
    # rho_steps = steps

    # rho_step = rho_max / rho_steps
    # theta_step = np.pi / n_thetas
    
    n = len(clusters)
    rhos = [0] * n
    thetas = [0] * n
    for i in range(n):
        rhos[i] = np.mean(clusters[i], axis=0)[0]
        thetas[i] = np.mean(clusters[i], axis=0)[1]
    return rhos, thetas


def detect_straight_lines(img, rho_steps, theta_steps, cluster_ident_thres, cluster_size_thres, n_iter=int(1e4)):
    '''
    Detect straight lines on an image using randomized Radon transform.
    '''
    radon = random_radon_transformation(img, rho_steps=rho_steps, theta_steps=theta_steps, n_iter=n_iter)
    radon_clusterized = get_clusters(cluster_size_thres, radon, cluster_ident_thres)
    return get_lines(radon_clusterized)

# test example generation
def generate_line_points(im_width, slope, offset, thickness=1, noise=0.0):
    '''
    Generate list of points placed along line with given offset and slope
    '''
    x = np.array([], 'int64')
    y = np.array([], 'int64')
    
    for i in range(0, thickness):
        
        x_i = np.arange(0, im_width, 1)
        y_i = slope * x_i + offset + i + (np.random.normal(0.0, 1.0, x_i.shape)*noise if noise != 0.0 else 0.0)

        points = [(x_i[j], y_i[j]) for j in range(im_width)]
        filtered_points = np.array(list(filter(lambda x: (x[1] >= 0) * (x[1] < im_width), points))) # bounds check
        if (filtered_points.shape[0] != 0):
            x_i = filtered_points[:, 0]
            y_i = filtered_points[:, 1]
            x = np.hstack([x, x_i])
            y = np.hstack([y, y_i])
    return x, y


def generate_line_points_angle(im_len, offset, slope_theta, thickness=1, noise=0.0):
    '''
    Generate list of points placed along line with given offset and angle
    '''
    x = np.array([], 'int64')
    y = np.array([], 'int64')
    x0 = y0 = im_len//2
    
    for i in range(0, thickness):
        x_i = np.array([])
        y_i = np.array([])
        if (abs(slope_theta - np.pi / 2) < 1e-4):
            x_i = np.array([x0 + i] * im_len)
            y_i = np.arange(0, im_len, 1)
            x = np.hstack([x, x_i])
            y = np.hstack([y, y_i])
        else:
            x_i = np.arange(0, im_len, 1)
            y_i = np.array(y0 + offset/(math.cos(slope_theta)) + (x_i-x0)*math.tan(slope_theta)).astype('int')+i \
                                                               + (np.random.normal(0.0, 1.0, x_i.shape)*noise if noise != 0.0 else 0.0)
            points = [(x_i[j], y_i[j]) for j in range(im_len)]
            filtered_points = np.array(list(filter(lambda pixel: (pixel[1]>=0)*(pixel[1]<im_len), points))) # bounds check
            if (filtered_points.shape[0] != 0):
                x_i = filtered_points[:, 0]
                y_i = filtered_points[:, 1]
                x = np.hstack([x, x_i])
                y = np.hstack([y, y_i])
    return x, y