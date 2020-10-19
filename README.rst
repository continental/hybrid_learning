Hybrid Learning for DNNs
========================

This is an implementation of a suggested hybrid learning life-cycle
which aims to ensure the correct embedding of visual semantic concepts
(defined by labeled examples) in the latent space of DNNs.
The current core functionalities of the provided modules are:

- *Analysis* (finding and quality assessment) of concept embeddings.
- *Custom dataset handles* for some standard concept datasets.
- *Model extension* methods which allow to e.g. extend model output by
  concept predictions for multi-task training.

.. entry-point: installation instructions

Installation
------------

Getting the Source Code
^^^^^^^^^^^^^^^^^^^^^^^^

For now just use ``git clone``.


Preliminaries
^^^^^^^^^^^^^^^^^^^^^^^^

The project is built against ``Python 3.6.10``.
Find

- requirements for deployment in the ``requirements.txt`` file,
- additional requirements for development in the ``requirements-dev.txt`` file, and
- the direct dependencies in the ``setup.py`` file.

Follow the instructions below for (machine specific) installation.

Pytorch for your Machine
~~~~~~~~~~~~~~~~~~~~~~~~

If no build tools are available on your machine, installation of ``torch``
and ``torchvision`` (:code:`python -m pip install torchvision torch`) may fail
with build error.
The latest stable version of ``torch`` and ``torchvision`` can be manually
installed by selecting the corresponding ``pip`` install command in the
Quick start section of the `pytorch homepage <https://pytorch.org/>`_.


Windows: pycocotools for COCO datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The COCO dataset handles use the package ``pycocotools>=2.0``.
For Linux, simply proceed with the next section, as ``pycocotools`` can
be installed from the repositories via

.. code:: bash

    python -m pip install pycocotools

For Windows, make sure to have installed

- the ``C++`` *Build Tools for Visual Studio Code* from
  `here <https://visualstudio.microsoft.com/downloads/>`_
- ``numpy>=1.18.2`` (:code:`python -m pip install numpy`)
- ``Cython>=0.29.16`` (:code:`python -m pip install Cython`)

Then one can build the python3 pycocotools for Windows e.g. using the
following port:

.. code:: bash

    python -m pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

Installation of other packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the requirements for deployment simply install them via the
provided ``requirements.txt`` file:

.. code:: bash

    python -m pip install -r requirements.txt

For development (test execution, linting, documentation generation),
install from ``requirements-dev.txt``:

.. code:: bash

    python -m pip install -r requirements-dev.txt

If you encounter an memory error you could also disable pip caching by

.. code:: bash

    python -m pip --no-cache-dir install -r requirements.txt


Create and Install as Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create an installable wheel, make sure ``setuptools`` and ``wheel`` are installed
and up-to-date (usually pre-installed in virtual environments):

.. code:: bash

    python -m pip install --upgrade setuptools wheel

Build the wheel into the directory ``dist``:

.. code:: bash

    python setup.py bdist_wheel -d dist

Now the built wheel package can be installed into any environment:

.. code:: bash

    python -m pip install /path/to/dist/hybrid_learning-VERSION.whl

If any installation issues occur due to missing torch or torchvision dependencies,
manually ensure a current version of ``torch`` and ``torchvision`` is installed
(see Preliminaries section).



For Developers
---------------

Documentation Generation
^^^^^^^^^^^^^^^^^^^^^^^^^

To generate the `sphinx <https://www.sphinx-doc.org/>`_ documentation,
make sure the following packages are installed (included in development requirements):

- ``sphinx``
- ``sphinx_automodapi``
- ``autoclasstoc``
- ``sphinx_rtd_theme``

Then call:

.. code:: bash

    python -m sphinx docs/source docs/build

The entry point for the resulting documentation then is ``docs/build/index.html``.
Note that you will need an internet connection to successfully download the
object inventories for cross-referencing external documentations.

For a clean build remove the directories

- ``docs/build``: The built HTML documentation as well as build artifacts
- ``docs/source/apiref/generated``: The auto-generated API documentation files

One can also use the provided Makefile at ``docs/Makefile``.
For this, ensure the shell command :code:`python -m sphinx` can be executed in the
command line. Then call one of

.. code:: bash

    make -f docs/Makefile clean                        # clean artifacts from previous builds
    make -f docs/Makefile build                        # normal sphinx html build
    make -f docs/Makefile build SPHINXOPTS="-b latex"  # build with additional options for sphinx


Code checks
^^^^^^^^^^^^^^^^^^^^^^^^^

**Preliminaries**: The train and test images and the ``pytest`` python package.

For mini training and testing, example images need to be downloaded.
The needed images are specified in the items of the ``images`` list in the JSON annotation files.
The online sources of the images can be found in the ``flickr_url`` field of the items in the ``images`` list,
the required filenames can be found in the ``file_name`` field.
Find the annotation files in ``dataset/coco_test/annotations`` and put

- images listed in the training annotation file into ``dataset/coco_test/images/train2017``,
- images listed in the validation annotation file into ``dataset/coco_test/images/val2017``.

For running the tests, ensure ``pytest`` is installed (included in development requirements),
and call from within the project root directory:

.. code:: bash

    python -m pytest -c pytest.ini test/

For running `doctest <https://docs.python.org/3/library/doctest.html>`_ on the
docstrings run

.. code:: bash

    python -m pytest -c pytest.ini hybrid_learning/ docs/source/

For all at once

.. code:: bash

    python -m pytest -c pytest.ini hybrid_learning/ docs/source/ test/


.. entry-point: contributing

Contributing
^^^^^^^^^^^^^^^^^^^^^^^^^

See the project's ``CONTRIBUTING.md``.



.. entry-point: license

License
-------------

Copyright (c) 2020 Continental Corporation. All rights reserved.

This repository is licensed under the MIT license.
See ``LICENSE.txt`` for the full license text.
