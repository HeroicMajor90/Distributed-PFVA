#!/usr/bin/env python
from global_array import GlobalArray
import numpy as np
from mpi4py import MPI

print("TEST: Matrix RREF")
shape = np.empty(1, dtype=np.int32)
for i in range(1000):
    shape[:] = np.random.randint(1, 10, 1, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[0])
    MPI.COMM_WORLD.Bcast(A)
    if np.linalg.det(A) == 0:
        continue  # Skip for now
    B = np.eye(shape[0])
    X = np.concatenate((A, B), axis=1)
    print(X)
    X_ga = GlobalArray.array(X)
    X_ga.rref()
    X_ga.disp()

    #A_inv = np.linalg.inv(A)
    #C_ga = GlobalArray.array(C)
    #if C_ga != A_ga.dot(B_ga) and MPI.COMM_WORLD.Get_rank() == 0:
    #    (C_ga - A_ga.dot(B_ga)).disp()
    #    raise Exception("FAIL")
    #if C_ga == A_ga.dot(B_ga) and MPI.COMM_WORLD.Get_rank() == 0:
    #    print(i)

print("TEST: Matrix Multiplication")
shape = np.empty(3, dtype=np.int32)
for i in range(1000):
    shape[:] = np.random.randint(1, 1000, 3, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    B = 1000 * np.random.rand(shape[1], shape[2])
    MPI.COMM_WORLD.Bcast(A)
    MPI.COMM_WORLD.Bcast(B)
    A_ga = GlobalArray.array(A)
    B_ga = GlobalArray.array(B)
    C = A.dot(B)
    C_ga = GlobalArray.array(C)
    if C_ga != A_ga.dot(B_ga) and MPI.COMM_WORLD.Get_rank() == 0:
        (C_ga - A_ga.dot(B_ga)).disp()
        raise Exception("FAIL")
    if C_ga == A_ga.dot(B_ga) and MPI.COMM_WORLD.Get_rank() == 0:
        print(i)

