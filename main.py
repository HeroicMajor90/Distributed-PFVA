#!/usr/bin/env python
from global_array import GlobalArray
import numpy as np
from mpi4py import MPI

print("TEST: Matrix slicing")
shape = np.empty(2, dtype=np.int32)
for i in range(1000):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = GlobalArray.array(A)

    start = np.random.randint(0, shape[0], 1, np.int32)[0]
    stop = np.random.randint(start, max(start + 1, shape[0]), 1, np.int32)[0]
    step = np.random.randint(1, max(stop-start+1, 2), 1, np.int32)[0]

    start1 = np.random.randint(0, shape[1], 1, np.int32)[0]
    stop1 = np.random.randint(start, max(start1 + 1, shape[0]), 1, np.int32)[0]
    step1 = np.random.randint(1, max(stop1-start1+1, 2), 1, np.int32)[0]

    start = MPI.COMM_WORLD.bcast(start, root=0)
    stop = MPI.COMM_WORLD.bcast(stop, root=0)
    step = MPI.COMM_WORLD.bcast(step, root=0)

    start1 = MPI.COMM_WORLD.bcast(start1, root=0)
    stop1 = MPI.COMM_WORLD.bcast(stop1, root=0)
    step1 = MPI.COMM_WORLD.bcast(step1, root=0)

    AS = A[start:stop:step, start1:stop1:step1]

    AS_ga = GlobalArray.array(A[start:stop:step, start1:stop1:step1])

    Sliced_Array = A_ga[start:stop:step, start1:stop1:step1]

    if Sliced_Array != AS_ga:
        AS_ga.disp()
        Sliced_Array.disp()
        raise Exception("FAIL")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print(i)


print("TEST: Matrix indexing")
shape = np.empty(2, dtype=np.int32)
for i in range(1000):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = GlobalArray.array(A)

    index = np.random.randint(0, shape[0], 1, np.int32)[0]

    index = MPI.COMM_WORLD.bcast(index, root=0)

    AS_ga = GlobalArray.array(A[index][np.newaxis])

    Indexed_Array = A_ga[index]

    if Indexed_Array != AS_ga:
        AS_ga.disp()
        Indexed_Array.disp()
        raise Exception("FAIL")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print(i)


print("TEST: Matrix row slicing")
shape = np.empty(2, dtype=np.int32)
for i in range(1000):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = GlobalArray.array(A)

    start = np.random.randint(0, shape[0], 1, np.int32)[0]
    stop = np.random.randint(start, max(start + 1, shape[0]), 1, np.int32)[0]
    step = np.random.randint(1, max(stop-start+1, 2), 1, np.int32)[0]

    start = MPI.COMM_WORLD.bcast(start, root=0)
    stop = MPI.COMM_WORLD.bcast(stop, root=0)
    step = MPI.COMM_WORLD.bcast(step, root=0)

    AS_ga = GlobalArray.array(A[start:stop:step])

    Sliced_Array = A_ga[start:stop:step]

    if Sliced_Array != AS_ga:
        AS_ga.disp()
        Sliced_Array.disp()
        raise Exception("FAIL")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print(i)


print("TEST: Matrix Transpose")
shape = np.empty(2, dtype=np.int32)
for i in range(1000):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = GlobalArray.array(A)
    AT_ga = GlobalArray.array(A.transpose())

    if A_ga.transpose() != AT_ga:
        A_ga.disp()
        AT_ga.disp()
        A_ga.transpose().disp()
        raise Exception("FAIL")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print(i)

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
    X_ga = GlobalArray.array(X)
    X_ga.rref()

    Ainv = np.linalg.inv(A)
    Ainv_ga = GlobalArray.array(np.concatenate((B, Ainv), axis=1))
    if Ainv_ga != X_ga:
        X_ga.disp()
        raise Exception("FAIL")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print(i)

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
    if C_ga != A_ga.dot(B_ga):
        (C_ga - A_ga.dot(B_ga)).disp()
        raise Exception("FAIL")
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print(i)
