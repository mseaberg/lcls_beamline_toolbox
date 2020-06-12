# lcls_beamline_toolbox
 
### Dependencies:
Requires Python 3.7 to already be installed.

All package dependencies are installed automatically during installation.

### How to install:

From a command line run the following:

<code>git clone https://github.com/mseaberg/lcls_beamline_toolbox </code>  
<code>cd lcls_beamline_toolbox </code>  
<code>python3 -m pip install -e .</code>(this step will install the package as an editable package)

### Examples:

Open a jupyter notebook and navigate to the lcls_beamline_toolbox/scripts directory. This directory includes a 
number of examples that will eventually be well-documented.

### Acknowledgments

Propagation method inspired by SRW<sup>1</sup>. Crystal reflections use xrt raycing backend<sup>2</sup>.

1. Chubar, O. & Celestre, R. Memory and CPU efficient computation of the Fresnel free-space propagator in 
Fourier optics simulations. Opt. Express 27, 28750 (2019).
2. K. Klementiev and R. Chernikov, “Powerful scriptable ray tracing package xrt”, Proc. SPIE 9209, Advances
 in Computational Methods for X-Ray Optics III, 92090A; doi:10.1117/12.2061400. Online documentation at 
 xrt.readthedocs.io; doi:10.5281/zenodo.1252468.