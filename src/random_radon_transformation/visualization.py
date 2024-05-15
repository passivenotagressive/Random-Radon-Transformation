import numpy as np
import cv2 as cv
import os
import shutil
from random_radon_transformation import random_radon_transform


def create_line(img, a, b):
    d = (np.cos(b * np.pi), np.sin(b * np.pi))
    l = (img.shape[0] ** 2 + img.shape[1] ** 2) ** 0.5
    r = 2 * l * a
    px = np.linspace(-d[0] * l, d[0] * l, int(4 * l), endpoint=False).astype(int)
    py = np.linspace(-d[1] * l, d[1] * l, int(4 * l), endpoint=False).astype(int)
    px += int(-r * d[1])
    py += int(r * d[0])

    x_points = []
    y_points = []
    for x, y in zip(px, py):
        if (x >= 0) & (x < img.shape[0]) & (y >= 0) & (y < img.shape[1]):
            x_points.append(x)
            y_points.append(y)

    return x_points, y_points


def revert_radon(img, rrt):
    sh = rrt.shape
    for x in range(sh[0]):
        for y in range(sh[1]):
            if rrt[x, y] > 0.5:
                y_points, x_points = create_line(img, x / sh[0] - 0.5, y / sh[1] - 0.5)
                cv.line(img, (x_points[0], y_points[0]), (x_points[-1], y_points[-1]), (0, 0, 255), 1)


def visualisation(img, rrt):
    path = f".../images"
    try:
        os.makedirs(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.makedirs(path)

    cv.imwrite("initial pic.png", img)
    cv.imwrite("random radon transform.png", 255 - rrt)
    filtered_rrt = (rrt > 100).astype(int) * 255
    img_after = cv.imread("initial pic.png")
    revert_radon(img_after, filtered_rrt)
    cv.imwrite("result.png", img_after)
