"""Utils for testing."""

import numpy as np

import math


def generate_line_points_angle(im_length: int, im_hight: int, r: float , theta: float, thickness: int = 1) -> tuple:
    """
    Generate points on the line inside picture by rho and theta.

    Arg:
        im_length : int
            length of the picture
        im_hight : int
            hight og the picture
        r : float
            rho
        theta : float
            angle
        thickness : int
            thickness of the line in pixles

    Returns:
        x, y : tuple
            tuple of arrays of points
    """
    x = np.array([], 'int64')
    y = np.array([], 'int64')
    x0 = im_length // 2
    y0 = im_hight // 2

    for i in range(0, thickness):
        x_i = np.array([])
        y_i = np.array([])
        if (abs(theta - np.pi / 2) < 1e-4):
            x_i = np.array([x0 + i] * im_length)
            y_i = np.arange(0, im_hight, 1)
            x = np.hstack([x, x_i])
            y = np.hstack([y, y_i])
        else:
            x_i = np.arange(0, im_length, 1)
            y_i = np.array(y0 + r / (math.cos(theta)) + (x_i - x0) * math.tan(theta)).astype('int') + i
            points = [(x_i[j], y_i[j]) for j in range(im_length)]
            filtered_points = np.array(list(filter(lambda pixel: (pixel[1] >= 0) * (pixel[1] < im_hight), points)))
            if (filtered_points.shape[0] != 0):
                x_i = filtered_points[:, 0]
                y_i = filtered_points[:, 1]
                x = np.hstack([x, x_i])
                y = np.hstack([y, y_i])
    return x, y


def generate_one_line(im_length: int, im_height: int, rho: float, theta: float, thickness: int = 2) -> np.ndarray:
    """
    Generate one line on the plot of given size and with given rho and theta.

    Arg:
        im_length : int
            length of the picture
        im_height : int
            height of the picture
        rho : float
        theta : float
        thickness : int
            in pixels

    Returns:
          img : np.ndarray
            image with one line
    """
    x_points, y_points = generate_line_points_angle(im_length, im_height, rho, theta, thickness)
    img = np.zeros((im_length, im_height), dtype=np.uint8)
    img[x_points, y_points] = 255
    img = 255 - img
    return img


def generate_n_lines(im_length: int, im_height: int, rhos: list, thetas: list, thickness: int = 2) -> np.ndarray:
    """
    Generate n lines on the plot of given size and with given rhos and thetas.

    Arg:
         im_length : int
            length of the picture
        im_height : int
            height of the picture
        rhos : list
            rhos for all the lines
        thetas : list
            thetas for all the lines
        thickness : int
            in pixels

    Returns:
        img : np.ndarray
            image with two lines

    """
    n = len(rhos)
    img = np.zeros((im_length, im_height), dtype=np.uint8)
    for i in range(n):
        x_points, y_points = generate_line_points_angle(im_length, im_height, rhos[i], thetas[i], thickness)
        img[x_points, y_points] = 255
    img = 255 - img
    return img


def add_noise(img: np.ndarray, level: int, rigidity: int) -> np.ndarray:
    """
    Adds noise on the given image.

    Arg:
        img : np.ndarray
            initial image
        level : int
            quality of the noise
        rigidity : int
            quantity of the noise

    Returns:
        ret : np.ndarray
            noised image
    """
    ret = np.copy(img)
    for i in range(level):
        i, j = np.random.randint(0, ret.shape[0], 2)
        ret[i][j] = max(0, ret[i][j] - rigidity)
    return ret
