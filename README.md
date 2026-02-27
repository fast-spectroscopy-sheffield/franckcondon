# franckcondon
Classes for Franck-Condon fitting of photoluminescene and absorption spectra using one or more vibrational modes.

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
Take a look at **example.py** for intended usage.

### Documentation
Open **doc/\_build/html/index.html** in your browser.

### Notes
For PL spectra, a photon density of states correction has to be applied; phonon is just a misspelling in the code.

Fitting should be done on an energy $x$-axis (not wavelength). Given that, please pre-apply the Jacobian correction to spectroscopic data, if required (intensity, PL, ...). It's not required for ratiometric data (absorbance, TA, ...). See, for example, [this paper](https://pubs.acs.org/doi/full/10.1021/jz401508t) for details.
