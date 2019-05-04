#!/usr/bin/env python
from ga4py import gain
# Disable deprecated warning
import warnings
warnings.simplefilter("ignore")

A = gain.asarray([[1, 2], [3, 4]])
B = gain.asarray([[5, 6], [7, 8]])
print(A)
print(B)
print(A.dot(B))
