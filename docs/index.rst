The Radon Random Transformation Project
==============================

.. toctree::
   :hidden:
   :maxdepth: 1

   reference

The command-line interface runs Random Radon Transform on any .png images.

Installation
------------

To install the Radon project, copy  `Kuznetsova Alina's <https://github.com/passivenotagressive/Random-Radon-Transformation>`_
repository and use poetry:

.. code-block:: console

   $ pip install poetry

Usage
----

Radon Radon Transformation usage looks like:

.. code-block:: console

   $ poetry run random-radon-transformation [OPTIONS]

.. option:: -f <PATH>, --file <PATH>

Path to the source image for the transformation.

.. option:: -o <PATH>, --output <PATH>

   Path where output will be written


Testing
----
To run tests use:

.. code-block:: console

   $ nox -r

