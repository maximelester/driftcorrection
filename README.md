# driftcorrection
Drift correction method for SPM images, implemented in python, with an example file using real STM data. (c) Maxime Le Ster, 2023

## Introduction

Full paper describing the model can be found elsewhere (DOI: to be disclosed).
The python library (``driftcorrection.py``) allows to load and distort SPM images according to a measured and target lattice.
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

Using the driftcorrection library is really straightforward. Typically it is convenient to use the following steps.
For more detail, check the example file ``exec.py``, which corrects an atomically-resolved STM image of highly-oriented pyrolithic graphite.

### 1) Loading the data

The data can be loaded into a DriftCorrection object as follows:
``DriftCorrection(path)``
If format does not contain metadata (specifying dimensions of the image), it is essential to use the following:
``DriftCorrection(path, Lx, Ly)`` with ``Lx`` and ``Ly`` the x and y dimensions of the image.

### 2) Displaying raw data

The raw data (along with the FFT) can be displayed using .ShowRaw()

### 3) Setting the raw lattice

The measured lattice can be specified using two methods:
- ``.SetRawReci(K1, K2)``  with ``K1`` and ``K2`` both passed as ``tuple``, ``list`` or ``np.array`` (shape: (2,))
- ``.SetRawReal(r1, r2, omega, theta)`` with ``r1``, ``r2`` sides of parallelogram, ``omega`` angle between R1 and R2, and ``theta`` angle between x-axis and R1.

### 4) Setting the target lattice

The target lattice is specified using similar methods:
- ``.SetTargetReci(K1, K2)``
- ``.SetTargetReal(r1, r2, omega, theta)``

### 5) Performing the transformation

The ``warpAffine`` function is embedded in a function and the transformation is achieved as follows:
``.Transform()``

### 6) Displaying the results

Once the transformation is done, the results can be plot as follows:
``.ShowAll()``
The details of the transformation (real and reciprocal matrices, dimensions, ...) can be printed into the console with the following command:
``.PrintResults()``

### 7) Saving the data

Lastly, if satisfied with the warped image, the warped real space image and/or the warped FFT can be saved with:
``.SaveData(which)``, with ``which='real'`` for real space image only, ``which='fft'`` or ``which='both'``.





