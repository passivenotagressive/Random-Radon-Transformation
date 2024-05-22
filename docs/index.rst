The Radon Random Transformation Project
=========================================

.. toctree::
   :hidden:
   :maxdepth: 1

The command-line interface runs Random Radon Transform on any .png images.

Installation
------------

To install the Radon project, copy  `Kuznetsova Alina's <https://github.com/passivenotagressive/Random-Radon-Transformation>`_
repository and use poetry:

.. code-block:: console

   $ pip install poetry

Usage
------------

Radon Radon Transformation usage looks like:

.. code-block:: console

   $ poetry run random-radon-transformation [OPTIONS]

.. option:: -s <PATH>

    Path to the source image for the transformation. Pay attention that it should be a file.

.. option:: -o <PATH>

   Path where output will be written. Pay attention that it should be a directory.


Testing
------------
To run tests use:

.. code-block:: console

   $ nox -r

