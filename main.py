#!/usr/bin/env python
from global_array import GlobalArray

A = GlobalArray.array([[1, 2], [3, 4]])
A.disp()
B = GlobalArray.array([[5, 6], [7, 8]])
B.disp()
C = A.dot(B)
C.disp()
