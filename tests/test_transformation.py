import click.testing
import pytest
import numpy as np
import cv2 as cv

from src.random_radon_transformation import random_radon_transform

from . import utils
def test_non_zero():
    """Check the get_nonzero() function."""
    img = np.zeros((244, 244), dtype=np.uint8)
    img[0][10] = 10
    img[57][179] = 40
    img = 255 - img
    print(len(random_radon_transform.get_nonzero(img)))
    assert len(random_radon_transform.get_nonzero(img)) == 2

def test_choice():
    """Check the choice() function."""
    img = np.zeros((244, 244), dtype=np.uint8)
    img[57][179] = 10
    img[179][57] = 10
    img = 255 - img
    pixel1, pixel2 = random_radon_transform.choice(random_radon_transform.get_nonzero(img))
    assert ((utils.check_identical(pixel1, [179, 57]) and utils.check_identical(pixel2, [57, 179])) or
            (utils.check_identical(pixel2, [179, 57]) and utils.check_identical(pixel1, [57, 179])))

def test_one_line():
    """Test an image with one line."""
    img = utils.random_img_one_line(244, 244)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'one_line')

def test_horizontal():
    """Test an image with horizontal line."""
    img = utils.generate_one_line(244, 244, 10, np.pi / 2)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'horizontal')

def test_vertical():
    """Test an image with vertical line."""
    img = utils.generate_one_line(244, 244, 10, 0)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'vertical')

def test_positive():
    """Test an image with a line with positive decline."""
    img = utils.generate_one_line(244, 244, 10, -np.pi / 4)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'positive')

def test_negative():
    """Test an image with a line with negative decline."""
    img = utils.generate_one_line(244, 244, 10, np.pi / 3)
    rrt = random_radon_transform.transform(img)
    random_radon_transform.visualisation(img, rrt, 'negative')

