import pytest
import numpy as np
from random import random
from random_radon_transform import generate_line_points, generate_line_points_angle

@pytest.fixture
def radon_static():
    im_len = 256
    img = np.full((im_len, im_len), fill_value=255, dtype=np.uint8)

    rhos = [40, 200]
    thetas = [2, -1]

    for rho, theta in zip(rhos, thetas):
        x_points, y_points = generate_line_points_angle(im_len, slope_theta=theta, offset=rho, thickness=5)
        img[x_points, y_points] = 0

    return img, rhos, thetas

@pytest.fixture
def radon_static_inverted():
    img, rhos, thetas = radon_static()
    img = 255 - img
    return img, rhos, thetas

@pytest.fixture
def radon_dynamic_random():
    n_lines = 3
    im_len = 256
    im_diam = im_len * np.sqrt(2)
    img = np.full((im_len, im_len), fill_value=255, dtype=np.uint8)

    rhos   = [random()*np.pi/2 for _ in range(n_lines)]
    thetas = [random()*im_diam for _ in range(n_lines)]

    for rho, theta in zip(rhos, thetas):
        x_points, y_points = generate_line_points_angle(im_len, slope_theta=theta, offset=rho, thickness=5)
        img[x_points, y_points] = 0

    return img, rhos, thetas

@pytest.fixture
def radon_dynamic_random_noise(): # noisy lines
    n_lines = 3
    im_len = 256
    im_diam = im_len * np.sqrt(2)
    img = np.full((im_len, im_len), fill_value=255, dtype=np.uint8)

    rhos   = [random()*np.pi/2 for _ in range(n_lines)]
    thetas = [random()*im_diam for _ in range(n_lines)]

    for rho, theta in zip(rhos, thetas):
        x_points, y_points = generate_line_points_angle(im_len, slope_theta=theta, offset=rho, thickness=5, noise=2)
        img[x_points, y_points] = 0

    return img, rhos, thetas

@pytest.fixture
def radon_noise():
    img = np.random.normal(127, 32, (255, 255))
    img = np.clip(img, 0, 255)

    return img, [], []

@pytest.fixture
def tiny_image():
    array = np.array([[255, 0],
                      [0, 255]])
    return array, [], []