import click

from . import __version__, random_radon_transform


@click.command()
@click.version_option(version=__version__)
def main():
    click.echo("Hello, world!")