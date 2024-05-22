"""CLI for Random Radon Transform."""

import click

import cv2 as cv

import sys

from . import __version__
from .random_radon_transform import transform, visualisation


@click.option("-s", "--source", help="Input file")
@click.option("-o", "--output", default="drt.png", help="Output file")
@click.version_option(version=__version__)
def main() -> None:
    """Finally main!"""
    source = sys.argv[1]
    output = sys.argv[2]

    img = cv.imread(source)
    rrt = transform(img)
    visualisation(img, rrt, "result", output)
