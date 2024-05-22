"""Tests of the transformation."""

import numpy as np

import cv2 as cv

from src.random_radon_transformation import random_radon_transform

from . import utils


def test_horizontal() -> None:
    """Test an image with horizontal line."""
    img = utils.generate_one_line(244, 244, 10, np.pi / 2)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'horizontal', 'pics/test')


def test_vertical() -> None:
    """Test an image with vertical line."""
    img = utils.generate_one_line(244, 244, 10, 0)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'vertical', 'pics/test')


def test_positive() -> None:
    """Test an image with a line with positive decline."""
    img = utils.generate_one_line(244, 244, 10, -np.pi / 4)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'positive', 'pics/test')


def test_negative() -> None:
    """Test an image with a line with negative decline."""
    img = utils.generate_one_line(244, 244, 10, np.pi / 3)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'negative', 'pics/test')


def test_on_the_edge() -> None:
    """Test an image with one line on the edge of the picture."""
    img = utils.generate_one_line(244, 244, 100, np.pi / 10)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'on_the_edge', 'pics/test')


def test_intersecting() -> None:
    """Test an image with two intersecting lines."""
    img = utils.generate_n_lines(244, 244, [10, 10], [np.pi / 3, -np.pi / 4])
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'intersecting', 'pics/test')


def test_non_intersecting() -> None:
    """Test an image with two non-intersecting lines."""
    img = utils.generate_n_lines(244, 244, [30, -30], [np.pi / 3, np.pi / 4])
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'nonintersecting', 'pics/test')


def test_thick() -> None:
    """Test an image with one thick line."""
    img = utils.generate_one_line(244, 244, 10, -np.pi / 4, 6)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'thick', 'pics/test')


def test_elongated() -> None:
    """Test an image with elongated edge."""
    img = utils.generate_one_line(1000, 10, 0, -np.pi / 4)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'elongated', 'pics/test')


def test_small() -> None:
    """Test an image with small edge."""
    img = utils.generate_one_line(10, 10, 0, -np.pi / 2)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'small', 'pics/test')


def test_rectangle() -> None:
    """Test an image of rectangle."""
    img = cv.imread('pics/source/rectangle.png', 0)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'rectangle', 'pics/test')


def test_intervals() -> None:
    """Test an image with intervals."""
    img = cv.imread('pics/source/intervals.png', 0)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'intervals', 'pics/test')


def test_circle() -> None:
    """Test an image with a circle."""
    img = cv.imread('pics/source/rectangle.png', 0)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'rectangle', 'pics/test')


def test_many_lines() -> None:
    """Test with really a lot of lines."""
    img = cv.imread('pics/source/many_lines.png', 0)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'many_lines', 'pics/test')


def test_light_noise() -> None:
    """Test with some light noise."""
    img = utils.generate_one_line(244, 244, 10, np.pi / 3)
    noised = utils.add_noise(img, 10000, 60)
    rrt = random_radon_transform.transform(noised)
    random_radon_transform.visualisation(img, rrt, 'light_noise', 'pics/test')


def test_rough_noise() -> None:
    """Test with rough noise."""
    img = utils.generate_one_line(244, 244, 10, np.pi / 3)
    noised = utils.add_noise(img, 100000, 40)
    rrt = random_radon_transform.transform(noised)
    random_radon_transform.visualisation(img, rrt, 'rough_noise', 'pics/test')
