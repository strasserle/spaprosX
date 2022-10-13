spapros
==========

|PyPI| |Python Version| |License| |Read the Docs| |Build| |Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/spapros.svg
   :target: https://pypi.org/project/spapros/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/spapros
   :target: https://pypi.org/project/spapros
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/theislab/spapros
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/spapros/latest.svg?label=Read%20the%20Docs
   :target: https://spapros.readthedocs.io/
   :alt: Read the documentation at https://spapros.readthedocs.io/
.. |Build| image:: https://github.com/theislab/spapros/workflows/Build%20spapros%20Package/badge.svg
   :target: https://github.com/theislab/spapros/workflows/Build%20spapros%20Package/badge.svg
   :alt: Build package Status
.. |Tests| image:: https://github.com/theislab/spapros/actions/workflows/run_tests.yml/badge.svg
   :target: https://github.com/theislab/spapros/actions/workflows/run_tests.yml/badge.svg
   :alt: Tests
.. |Codecov| image:: https://codecov.io/gh/theislab/spapros/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/theislab/spapros
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black

|logo|

Features
--------

* Select probe sets for targeted spatial transcriptomics
* Evaluate probe sets with an extensive pipeline


Installation
------------

You can install *spapros* via pip_ from PyPI_:

.. code:: console

   $ pip install spapros


Note! Currently the pip installation into an environment without `leidenalg` installed can lead to problems when running
Spapros' knn-based evaluation (see `issue #234 <https://github.com/theislab/spapros/issues/234>`_). To solve this, run
Spapros in a conda environment and install leidenalg before installing spapros via:

.. code:: console

    $ conda install -c conda-forge leidenalg


Usage
-----

Visit our `documentation`_ for installation, tutorials, examples and more.


Credits
-------

This package was created with cookietemple_ using Cookiecutter_ based on Hypermodern_Python_Cookiecutter_.

.. |logo| image:: https://user-images.githubusercontent.com/21954664/111175015-409d9080-85a8-11eb-9055-f7452aed98b2.png
.. _cookietemple: https://cookietemple.com
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _MIT: http://opensource.org/licenses/MIT
.. _PyPI: https://pypi.org/
.. _Hypermodern_Python_Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _pip: https://pip.pypa.io/
.. _Contributor Guide: CONTRIBUTING.rst
.. _Usage: https://spapros.readthedocs.io/en/latest/usage.html
.. _documentation: https://spapros.readthedocs.io/en/latest/
