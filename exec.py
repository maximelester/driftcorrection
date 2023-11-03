# exec.py

import driftcorrection as dc
import numpy as np


# 1) data path and data structure

path = 'graphiteterrace.spm'

D = dc.DriftCorrection(path=path, unit='a')

# 2) check raw data and raw FFT

D.ShowRaw()

# 3) defining the raw reciprocal parameters (from FFT inspection)

D.SetRawReci(K1=(0.4191, 0.1692), K2=(0.1176, 0.5127))

# 4) defining the ideal real space parameters 

D.SetTargetReal(r1=2.461, r2=2.461, omega=120*np.pi/180, theta=-11.1*np.pi/180)

# 5) performing the warpAffine operation

D.Transform()

# 6) show results

D.PrintResults()

D.ShowAll(titles=True)

# 7) save data

D.SaveData()