# driftcorrection
Drift correction method for SPM images, implemented in python, with an example file using real STM data. (c) Maxime Le Ster, 2023

## Introduction

Full paper describing the model can be found elsewhere (DOI: to be disclosed).
The python library (driftcorrection.py) allows to load and distort SPM images according to a measured and target lattice.
The SPM data can be of the following formats:

- spm
- mtrx (Omicron)
- par  (Omicron)
- jpg
- png
- txt
- npy (numpy.array)

More formats may be added at a later date.

## How to use?

1) Loading the data
The data can be loaded into a DriftCorrection object as follows:
DriftCorrection(path)
If format does not contain metadata (specifying dimensions of the image), it is essential to use the following:
DriftCorrection(path, Lx, Ly) with Lx, Ly the x and y dimensions of the image.
2) Displaying raw data
The raw data (along with the FFT) can be displayed using .ShowRaw()
3) Setting the raw lattice
The measured lattice can be specified using two methods:
- .SetRawReci(K1, K2)  with K1 and K2 both tuples or list or np.array (2,)
- .SetRawReal(r1, r2, omega, theta) with r1, r2 sides of parallelogram, omega angle between R1 and R2, and theta angle between x-axis and R1.

4) Setting the target lattice
5) Performing the transformation
6) Displaying the results
7) Saving the data





