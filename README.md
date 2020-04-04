# franckcondon
Classes for Franck-Condon fitting of photoluminescene and absorption spectra using one or more vibrational modes.

The fitting is based on the equation from e.g. Ho _et al J. Chem. Phys._ 2001, **115** (6), 2709–2720

Thanks to Stefan Wedler and Anna Köhler for help with the underlying equation.

### How to Install
You need a python installation. I would recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

Clone the repository, navigate to the repository folder and run:
```sh
python setup.py install
```
You can then access the fitting class in your python scripts like so:
```python
from franckcondon import MultiModeFCFitting
```
### How to Use
Take a look at **example.py** for intended usage. I will add proper documentation at some point.
