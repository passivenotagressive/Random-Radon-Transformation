# Random Radon Transformation

This is implementation of fast, but not very accurate Random Radon Transformation, which is presented in [the following article](https://backend.orbit.dtu.dk/ws/portalfiles/portal/5529668/Binder1.pdf). 

Usage
------------

Radon Radon Transformation usage looks like:

.. code-block:: console

   $ poetry run random-radon-transformation [OPTIONS]

.. option::  <PATH>

    Path to the source image for the transformation. Pay attention that it should be a file.

.. option::  <PATH>

   Path where output will be written. Pay attention that it should be a directory.

Mind the right order.

Testing
------------
To run tests use:

.. code-block:: console

   $ nox -r

## Examples

### simple line

![](pics/final/result_thick.png)

### many lines

![](pics/final/result_many_lines.png)

### light noise

![](pics/final/result_light_noise.png)

We see that algorithm successfully cope with different number of lines, however can't deal with noise due to it's random nature.
