"""Perform random radon transformation."""

from random import randint

import numpy as np

import cv2 as cv


def get_nonzero(image: np.ndarray) -> np.ndarray:
    """
    Finds all non-zero pixels (white) on the image.

    Args:
        image : np.ndarray
            initial image

    Returns:
        pixels : np.ndarray
            list of non-zero pixel pairs
    """
    nonzero_x = np.nonzero(255 - image)[0]
    nonzero_y = np.nonzero(255 - image)[1]
    pairs = np.vstack((nonzero_x, nonzero_y)).T
    return pairs


def choice(pixels: np.ndarray) -> tuple:
    """

    Give random pair of different nonzero pixels from given list.

    Args:
        pixels : np.ndarray
            list of pixels given by pair of coordinates

    Returns:
        pixel1, pixel2 : tuple[list, list]
            pair of pixels
    """
    n = len(pixels)
    idx1 = randint(0, n - 1)
    pixel1 = pixels[idx1]
    idx2 = randint(0, n - 1)
    if idx1 == idx2:
        idx2 = (idx2 + 1) % n
    pixel2 = pixels[idx2]
    return pixel1, pixel2


def transform(image: np.ndarray, rho_steps: int = 220, theta_steps: int = 440, n_iter: int = int(1e5)) -> np.ndarray:
    """

    Perform random radon transformation of given image.

    Args:
        image : np.ndarray
            initial picture
        rho_steps : int
            step by rho for sampling
        theta_steps : int
            step by theta for sampling
        n_iter : int
            number of random iterations

    Returns:
        R : np.ndarray
            result of the transformation
    """
    nonzero_pixels = get_nonzero(image)

    a = image.shape[1]
    b = image.shape[0]

    n_rhos = 2 * rho_steps
    n_thetas = theta_steps

    rho_max = np.sqrt(a ** 2 + b ** 2)
    rho_step = rho_max / rho_steps
    theta_step = np.pi / n_thetas

    R = np.zeros((n_rhos, n_thetas), dtype='float64')
    for i in range(n_iter):
        pixel1, pixel2 = choice(nonzero_pixels)

        x_1 = pixel1[0]
        y_1 = pixel1[1]
        x_2 = pixel2[0]
        y_2 = pixel2[1]

        theta = np.pi / 2
        if y_1 != y_2:
            theta = np.arctan((x_2 - x_1) / (y_1 - y_2))
        if theta < 0:
            theta = np.pi + theta
        rho = x_1 * np.cos(theta) + y_1 * np.sin(theta)

        rho += rho_max

        theta_int = int(theta / theta_step)
        rho_int = int(rho / rho_step)
        R[rho_int][theta_int] += 1

    m = np.max(np.unique(R))
    R *= (255 / m)
    return R


def create_line(img: np.ndarray, a: int, b: int) -> tuple:
    """
    Create a line.

    Args:
        img : np.ndarray
            image on which we will add line
        a : int
            coordinate of the pixel by rho on the rrt picture
        b : int
            coordinate of the pixel by theta on the rrt picture

    Returns:
        x_points, y_points : tuple[list, list]
            two lists of coordinates of pixels on the line
    """
    d = (np.cos(b * np.pi), np.sin(b * np.pi))
    length = (img.shape[0] ** 2 + img.shape[1] ** 2) ** 0.5
    r = 2 * length * a
    px = np.linspace(-d[0] * length, d[0] * length, int(4 * length), endpoint=False).astype(int)
    py = np.linspace(-d[1] * length, d[1] * length, int(4 * length), endpoint=False).astype(int)
    px += int(-r * d[1])
    py += int(r * d[0])

    x_points = []
    y_points = []
    for x, y in zip(px, py):
        if (0 <= x < img.shape[0]) & (0 <= y < img.shape[1]):
            x_points.append(x)
            y_points.append(y)

    return x_points, y_points


def revert_radon(img: np.ndarray, rrt: np.ndarray) -> None:
    """
    On the initial picture add lines got by random radon transformation.

    Args:
        img : np.ndarray
            the initial image
        rrt : np.ndarray
            the random radon transformation picture
    """
    sh = rrt.shape
    for x in range(sh[0]):
        for y in range(sh[1]):
            if rrt[x, y] > 0.5:
                y_points, x_points = create_line(img, x / sh[0] - 0.5, y / sh[1] - 0.5)
                cv.line(img, (x_points[0], y_points[0]), (x_points[-1], y_points[-1]), (0, 0, 255), 1)


def visualisation(img: np.ndarray, rrt: np.ndarray, name: str) -> None:
    """
    Save files with pictures of initial picture, rrt picture and visualized lines got from rrt.

    Args:
        img : np.ndarray
            initial image
        rrt : np.ndarray
            random radon transformation picture
        name : str
            specific name of the file

    """
    cv.imwrite("pics/initial_pic_" + name + ".png", img)
    cv.imwrite("pics/rrt_" + name + ".png", 255 - rrt)
    filtered_rrt = (rrt > 100).astype(int) * 255
    img_after = cv.imread("pics/initial_pic_" + name + ".png")
    revert_radon(img_after, filtered_rrt)
    cv.imwrite("pics/result_" + name + ".png", img_after)
