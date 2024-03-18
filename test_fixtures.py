import pytest
import numpy as np
from random import random
from random_radon_transform import generate_line_points, generate_line_points_angle

@pytest.fixture
def radon_static():
    im_len = 256
    img = np.full((im_len, im_len), fill_value=255, dtype=np.uint8)

    rhos = [20, 80]
    thetas = [30*np.pi/180, 60*np.pi/180]

    for rho, theta in zip(rhos, thetas):
        x_points, y_points = generate_line_points_angle(im_len, im_len, theta=theta, offset=rho, thickness=5)
        img[x_points, y_points] = 0

    return img, rhos, thetas

@pytest.fixture
def radon_static_empty():
    im_len = 256
    img = np.full((im_len, im_len), fill_value=255, dtype=np.uint8)

    rhos = []
    thetas = []

    return img, rhos, thetas

@pytest.fixture
def radon_static_inverted(radon_static):
    img, rhos, thetas = radon_static
    img = 255 - img
    return img, rhos, thetas


@pytest.fixture
def radon_static_rect():
    im_height, im_width = 255, 129
    img = np.full((im_height, im_width), fill_value=255, dtype=np.uint8)

    rhos = [20, 80]
    thetas = [30*np.pi/180, 60*np.pi/180]

    for rho, theta in zip(rhos, thetas):
        x_points, y_points = generate_line_points_angle(im_height, im_width, theta=theta, offset=rho, thickness=5)
        img[x_points, y_points] = 0

    return img, rhos, thetas

@pytest.fixture
def radon_dynamic_random():
    n_lines = 1
    im_len = 256
    img = np.full((im_len, im_len), fill_value=255, dtype=np.uint8)

    rhos   = [random()*im_len/2   for _ in range(n_lines)]
    thetas = [random()*np.pi/2    for _ in range(n_lines)]

    for rho, theta in zip(rhos, thetas):
        x_points, y_points = generate_line_points_angle(im_len, im_len, theta=theta, offset=rho, thickness=5)
        img[x_points, y_points] = 0

    return img, rhos, thetas

@pytest.fixture
def radon_dynamic_random_noise(): # noisy lines
    n_lines = 1
    im_len = 256
    img = np.full((im_len, im_len), fill_value=255, dtype=np.uint8)

    rhos   = [random()*im_len/2   for _ in range(n_lines)]
    thetas = [random()*np.pi/2    for _ in range(n_lines)]

    for rho, theta in zip(rhos, thetas):
        x_points, y_points = generate_line_points_angle(im_len, im_len, theta=theta, offset=rho, thickness=5, noise=2)
        img[x_points, y_points] = 0

    return img, rhos, thetas

@pytest.fixture
def radon_noise():
    img = np.random.normal(127, 32, (255, 255))
    img = np.clip(img, 0, 255)

    return img, [], []

@pytest.fixture
def tiny_image():
    array = np.zeros((1, 1))
    return array, [], []