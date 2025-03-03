PyCoMSA |Stars|
===============

.. .. |Logo| image:: /_images/logo.png
..    :scale: 40%
..    :class: dark-light

.. |Stars| image:: https://img.shields.io/github/stars/althonos/pycomsa.svg?style=social&maxAge=3600&label=Star
   :target: https://github.com/althonos/pycomsa/stargazers
   :class: dark-light

*Cython bindings and Python interface to* `CoMSA <https://github.com/refresh-bio/CoMSA/>`_,
*a compressor for multiple sequence alignments*.

|Actions| |Coverage| |PyPI| |Bioconda| |AUR| |Wheel| |Versions| |Implementations| |License| |Source| |Mirror| |Issues| |Docs| |Changelog| |Downloads|

.. |Actions| image:: https://img.shields.io/github/actions/workflow/status/althonos/pycomsa/test.yml?branch=main&logo=github&style=flat-square&maxAge=300
   :target: https://github.com/althonos/pycomsa/actions
   :class: dark-light

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/althonos/pycomsa?style=flat-square&maxAge=600
   :target: https://codecov.io/gh/althonos/pycomsa/
   :class: dark-light

.. |PyPI| image:: https://img.shields.io/pypi/v/pycomsa.svg?style=flat-square&maxAge=3600
   :target: https://pypi.python.org/pypi/pycomsa
   :class: dark-light

.. |Bioconda| image:: https://img.shields.io/conda/vn/bioconda/pycomsa?style=flat-square&maxAge=3600
   :target: https://anaconda.org/bioconda/pycomsa
   :class: dark-light

.. |AUR| image:: https://img.shields.io/aur/version/python-pycomsa?logo=archlinux&style=flat-square&maxAge=3600
   :target: https://aur.archlinux.org/packages/python-pycomsa
   :class: dark-light

.. |Wheel| image:: https://img.shields.io/pypi/wheel/pycomsa?style=flat-square&maxAge=3600
   :target: https://pypi.org/project/pycomsa/#files
   :class: dark-light

.. |Versions| image:: https://img.shields.io/pypi/pyversions/pycomsa.svg?style=flat-square&maxAge=3600
   :target: https://pypi.org/project/pycomsa/#files
   :class: dark-light

.. |Implementations| image:: https://img.shields.io/pypi/implementation/pycomsa.svg?style=flat-square&maxAge=3600&label=impl
   :target: https://pypi.org/project/pycomsa/#files
   :class: dark-light

.. |License| image:: https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square&maxAge=3600
   :target: https://choosealicense.com/licenses/gpl-3.0/
   :class: dark-light

.. |Source| image:: https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=3600&style=flat-square
   :target: https://github.com/althonos/pycomsa/
   :class: dark-light

.. |Mirror| image:: https://img.shields.io/badge/mirror-LUMC-003EAA.svg?maxAge=3600&style=flat-square
   :target: https://git.lumc.nl/mflarralde/pycomsa/
   :class: dark-light

.. |Issues| image:: https://img.shields.io/github/issues/althonos/pycomsa.svg?style=flat-square&maxAge=600
   :target: https://github.com/althonos/pycomsa/issues
   :class: dark-light

.. |Docs| image:: https://img.shields.io/readthedocs/pycomsa?style=flat-square&maxAge=3600
   :target: http://pycomsa.readthedocs.io/en/stable/?badge=stable
   :class: dark-light

.. |Changelog| image:: https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=3600&style=flat-square
   :target: https://github.com/althonos/pycomsa/blob/main/CHANGELOG.md
   :class: dark-light

.. |Downloads| image:: https://img.shields.io/pypi/dm/pycomsa?style=flat-square&color=303f9f&maxAge=86400&label=downloads
   :target: https://pepy.tech/project/pycomsa
   :class: dark-light


Overview
--------

PyCoMSA is a Python module that provides bindings to Prodigal using
`Cython <https://cython.org/>`_. It directly interacts with the CoMSA
internals, which has the following advantages:


.. grid:: 1 2 3 3
   :gutter: 1

   .. grid-item-card:: :fas:`battery-full` Batteries-included

      Just add ``pycomsa`` as a ``pip`` or ``conda`` dependency, no need
      for the CoMSA binary or any external dependency.

   .. grid-item-card:: :fas:`screwdriver-wrench` Sans I/O

      Build alignments to compress as Python `str` or byte-like objects, no 
      need for intermediate files.

   .. grid-item-card:: :fas:`text-slash` Flexible format
      
      Pick the file format based on your needs, and not based on the 
      file format of the original aligmnment.

   .. grid-item-card:: :fas:`arrow-right-arrow-left` Better portability

      Support reading and writing files for any architecture, and not
      just native one as the original CoMSA.

   .. grid-item-card:: :fas:`check` Compatible

      Load files generated with PyCoMSA using CoMSA, as both are using 
      the same compression method.

   .. grid-item-card:: :fas:`toolbox` Feature-complete

      Access all the features of the original CLI through the :doc:`Python API <api/index>`.



Setup
-----

Run ``pip install pycomsa`` in a shell to download the latest release and all
its dependencies from PyPi, or have a look at the
:doc:`Installation page <guide/install>` to find other ways to install 
``pycomsa``.


Library
-------

Check the following pages of the user guide or the API reference for more
in-depth reference about library setup, usage, and rationale:

.. toctree::
   :maxdepth: 2

   User Guide <guide/index>
   API Reference <api/index>


Related Projects
----------------

The following Python libraries may be of interest for bioinformaticians.

.. include:: related.rst


License
-------

This library is provided under the `GNU General Public License v3.0 <https://choosealicense.com/licenses/gpl-3.0/>`_.
The Prodigal code was written by `Sebastian Deorowicz <https://github.com/sebastiandeorowicz>`_ and is distributed under the
terms of the GPLv3 as well. See the :doc:`Copyright Notice <guide/copyright>` section
for the full GPLv3 license.

*This project is in no way not affiliated, sponsored, or otherwise endorsed by
the original* `Prodigal`_ *authors. It was developed by* `Martin Larralde <https://github.com/althonos>`_ 
*during his PhD project at the* `Leiden University Medical Center <https://www.lumc.nl/en/>`_
*in the* `Zeller team <https://github.com/zellerlab>`_.

