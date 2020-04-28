franckcondon documentation
==========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   _pages/installation
   _pages/calculations
   _pages/fitting
   
About
=====

This package contains classes for calculating Franck-Condon progressions involving 1 or more vibrational modes and fitting them to experimental data.

The calculation is based on the equation found in (for example) `this paper <https://aip.scitation.org/doi/abs/10.1063/1.1372508>`_, which says that:

.. math::

   I_{PL}(E)\propto n^3E^3\sum_{(m_i)}\left\{\left(\prod_{i}\frac{S_i^{m_i}e^{-S_i}}{m_i!}\right)\Gamma\left[E-\left(E_0-\sum_{i}m_iE_i^{vib}\right)\right]\right\}

Where :math:`n` is refractive index, :math:`i` indexes the vibrational modes, :math:`m` indexes the vibrational sublevels, :math:`S_i` are Huang-Rhys parameters, :math:`E_0` is the energy of the 0-0 transition, :math:`E_i^{vib}` are the energies of the vibrational modes and :math:`\Gamma` is the lineshape function, taken here to be gaussian. Note that :math:`(m_i)` refers to all possible permutations of the :math:`m_i`.
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`