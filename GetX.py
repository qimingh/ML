# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 12:49:06 2018

@author: Minot
"""


import pandas as pd
import numpy as np
#from test_unit import _grad
p = pd.read_csv('outputP.txt', delimiter=" ", header = None)
f = pd.read_csv('outputF.txt', delimiter=" ", header = None)

print ("p.shape", p.shape)
print ("f.shape", f.shape)
p = np.matrix(p)
f = np.matrix(f)
#W=_grad(f)
M = np.dot(f,p.T)
m = pd.DataFrame(M)
print ("m.shape", m.shape)

