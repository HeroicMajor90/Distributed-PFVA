#!/usr/bin/env python
import global_array as ga
import numpy as np
from mpi4py import MPI

TRIES_PER_TEST = 100

def im_root():
    return MPI.COMM_WORLD.Get_rank() == 0


if im_root(): print("TEST: Greater than")
shape = np.empty(3, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 3, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[0])
    B = 1000 * np.random.rand(shape[0], shape[0])
    MPI.COMM_WORLD.Bcast(A)
    MPI.COMM_WORLD.Bcast(B)
    A_ga = ga.GlobalArray.array(A)
    B_ga = ga.GlobalArray.array(B)
    C = A > B
    C_ga = ga.GlobalArray.array(C)
    A_ga = A_ga > B_ga
    if (C_ga != A_ga):
        (C_ga - A_ga.dot(B_ga)).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)   


if im_root(): print("TEST: Lesser than")
shape = np.empty(3, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 3, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[0])
    B = 1000 * np.random.rand(shape[0], shape[0])
    MPI.COMM_WORLD.Bcast(A)
    MPI.COMM_WORLD.Bcast(B)
    A_ga = ga.GlobalArray.array(A)
    B_ga = ga.GlobalArray.array(B)
    C = A < B
    C_ga = ga.GlobalArray.array(C)
    A_ga = A_ga < B_ga
    if (C_ga != A_ga):
        (C_ga - A_ga.dot(B_ga)).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)        


if im_root(): print("TEST: QR Decomposition")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 100, 2, np.int32)
    shape[0] = shape.max()
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)

    Q_ga, R_ga = ga.qr(A_ga)
    Q_trans_ga = Q_ga.transpose()
    Eye_ga = ga.GlobalArray.eye(shape[0])
    if (Q_ga.dot(R_ga) != A_ga or
            Q_ga.dot(Q_trans_ga) != Eye_ga or
            Q_trans_ga.dot(Q_ga) != Eye_ga):
        Q_ga.disp()
        R_ga.disp()
        Q_ga.dot(R_ga).disp()
        A_ga.disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Sort by First Column")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    SA_ga = ga.GlobalArray.array(A[A[:, 0].argsort()])
    if ga.sort_by_first_column(A_ga) != SA_ga:
        SA_ga.disp()
        ga.sort_by_first_column(A_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix slice assignment")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)

    start = np.random.randint(0, shape[0], 1, np.int32)[0]
    stop = np.random.randint(start, max(start + 1, shape[0]), 1, np.int32)[0]
    step = np.random.randint(1, max(stop-start+1, 2), 1, np.int32)[0]

    start1 = np.random.randint(0, shape[1], 1, np.int32)[0]
    stop1 = np.random.randint(start1, max(start1 + 1, shape[1]), 1, np.int32)[0]
    step1 = np.random.randint(1, max(stop1-start1+1, 2), 1, np.int32)[0]

    offset = np.random.randint(-start, shape[0] - stop, 1, np.int32)[0]
    offset1 = np.random.randint(-start1, shape[1] - stop1, 1, np.int32)[0]

    start = MPI.COMM_WORLD.bcast(start)
    stop = MPI.COMM_WORLD.bcast(stop)
    step = MPI.COMM_WORLD.bcast(step)

    start1 = MPI.COMM_WORLD.bcast(start1)
    stop1 = MPI.COMM_WORLD.bcast(stop1)
    step1 = MPI.COMM_WORLD.bcast(step1)

    offset = MPI.COMM_WORLD.bcast(offset)
    offset1 = MPI.COMM_WORLD.bcast(offset1)

    A_ga[start+offset:stop+offset:step, start1+offset1:stop1+offset1:step1] = (
    	A_ga[start:stop:step, start1:stop1:step1])

    A[start+offset:stop+offset:step, start1+offset1:stop1+offset1:step1] = (
    	A[start:stop:step, start1:stop1:step1])

    AS_ga = ga.GlobalArray.array(A)

    if A_ga != AS_ga:
        AS_ga.disp()
        A_ga.disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix Std: Column Wise")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    C = np.std(A,axis=0)
    C = np.reshape(C,(-1,1))
    C_ga = ga.GlobalArray.array(C)
    AS_ga = A_ga.std(axis=0)
    if C_ga != AS_ga:
        (C_ga - AS_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix Std: Row Wise")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    C = np.std(A,axis=1)
    C = np.reshape(C,(-1,1))
    C_ga = ga.GlobalArray.array(C)
    AS_ga = A_ga.std(axis=1)
    if C_ga != AS_ga:
        (C_ga - AS_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix Std: Flat")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    C = np.std(A)
    C = np.reshape(C,(-1,1))
    C_ga = ga.GlobalArray.array(C)
    AS_ga = A_ga.std()
    if C_ga != AS_ga:
        C_ga.disp()
        (C_ga - AS_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)

if im_root(): print("TEST: Matrix Sum: Column Wise")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    C = np.sum(A,axis=0)
    C = np.reshape(C,(-1,1))
    C_ga = ga.GlobalArray.array(C)
    AS_ga = A_ga.sum(axis=0)
    if C_ga != AS_ga:
        (C_ga - AS_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix Sum: Row Wise")
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    C = np.sum(A,axis=1)
    C = np.reshape(C,(-1,1))
    C_ga = ga.GlobalArray.array(C)
    AS_ga = A_ga.sum(axis=1)
    if C_ga != AS_ga:
        (C_ga - AS_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix Sum: Flat")
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    C = np.sum(A,axis=None)
    C = np.reshape(C,(-1,1))
    C_ga = ga.GlobalArray.array(C)
    AS_ga = A_ga.sum(axis=None)
    if C_ga != AS_ga:
        (C_ga - AS_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)

if im_root(): print("TEST: Matrix Average: Column Wise")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    C = np.mean(A,axis=0)
    C = np.reshape(C,(-1,1))
    C_ga = ga.GlobalArray.array(C)
    AM_ga = A_ga.mean(axis=0)
    if C_ga != AM_ga:
        (C_ga - AM_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix Average: Row Wise")
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    C = np.mean(A,axis=1)
    C = np.reshape(C,(-1,1))
    C_ga = ga.GlobalArray.array(C)
    AM_ga = A_ga.mean(axis=1)
    if C_ga != AM_ga:
        (C_ga - AM_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix Average: Flat")
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    C = np.mean(A,axis=None)
    C = np.reshape(C,(-1,1))
    C_ga = ga.GlobalArray.array(C)
    AM_ga = A_ga.mean(axis=None)
    if C_ga != AM_ga:
        (C_ga - AM_ga).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: NP-2-GA-2-NP")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    if not np.allclose(A, A_ga.to_np()):
        A_ga.disp()
        ga.GlobalArray.array(A_ga.to_np()).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix slicing")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)

    start = np.random.randint(0, shape[0], 1, np.int32)[0]
    stop = np.random.randint(start, max(start + 1, shape[0]), 1, np.int32)[0]
    step = np.random.randint(1, max(stop-start+1, 2), 1, np.int32)[0]

    start1 = np.random.randint(0, shape[1], 1, np.int32)[0]
    stop1 = np.random.randint(start1, max(start1 + 1, shape[1]), 1, np.int32)[0]
    step1 = np.random.randint(1, max(stop1-start1+1, 2), 1, np.int32)[0]

    start = MPI.COMM_WORLD.bcast(start, root=0)
    stop = MPI.COMM_WORLD.bcast(stop, root=0)
    step = MPI.COMM_WORLD.bcast(step, root=0)

    start1 = MPI.COMM_WORLD.bcast(start1, root=0)
    stop1 = MPI.COMM_WORLD.bcast(stop1, root=0)
    step1 = MPI.COMM_WORLD.bcast(step1, root=0)

    AS = A[start:stop:step, start1:stop1:step1]

    AS_ga = ga.GlobalArray.array(A[start:stop:step, start1:stop1:step1])

    Sliced_Array = A_ga[start:stop:step, start1:stop1:step1]

    if Sliced_Array != AS_ga:
        AS_ga.disp()
        Sliced_Array.disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix indexing")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)

    index = np.random.randint(0, shape[0], 1, np.int32)[0]

    index = MPI.COMM_WORLD.bcast(index, root=0)

    AS_ga = ga.GlobalArray.array(A[index][np.newaxis])

    Indexed_Array = A_ga[index]

    if Indexed_Array != AS_ga:
        AS_ga.disp()
        Indexed_Array.disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix row slicing")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)

    start = np.random.randint(0, shape[0], 1, np.int32)[0]
    stop = np.random.randint(start, max(start + 1, shape[0]), 1, np.int32)[0]
    step = np.random.randint(1, max(stop-start+1, 2), 1, np.int32)[0]

    start = MPI.COMM_WORLD.bcast(start)
    stop = MPI.COMM_WORLD.bcast(stop)
    step = MPI.COMM_WORLD.bcast(step)

    AS_ga = ga.GlobalArray.array(A[start:stop:step])
    Sliced_Array = A_ga[start:stop:step]

    if Sliced_Array != AS_ga:
        AS_ga.disp()
        Sliced_Array.disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix Transpose")
shape = np.empty(2, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 2, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    MPI.COMM_WORLD.Bcast(A)
    A_ga = ga.GlobalArray.array(A)
    AT_ga = ga.GlobalArray.array(A.transpose())

    if A_ga.transpose() != AT_ga:
        A_ga.disp()
        AT_ga.disp()
        A_ga.transpose().disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix RREF")
shape = np.empty(1, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 100, 1, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[0])
    MPI.COMM_WORLD.Bcast(A)
    if np.linalg.det(A) == 0:
        continue  # Skip for now
    B = np.eye(shape[0])

    X = np.concatenate((A, B), axis=1)
    X_ga = ga.GlobalArray.array(X)
    X_ga.rref()

    Ainv = np.linalg.inv(A)
    Ainv_ga = ga.GlobalArray.array(np.concatenate((B, Ainv), axis=1))
    if Ainv_ga != X_ga:
        X_ga.disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)


if im_root(): print("TEST: Matrix Multiplication")
shape = np.empty(3, dtype=np.int32)
for i in range(TRIES_PER_TEST):
    shape[:] = np.random.randint(1, 1000, 3, np.int32)
    MPI.COMM_WORLD.Bcast(shape)
    A = 1000 * np.random.rand(shape[0], shape[1])
    B = 1000 * np.random.rand(shape[1], shape[2])
    MPI.COMM_WORLD.Bcast(A)
    MPI.COMM_WORLD.Bcast(B)
    A_ga = ga.GlobalArray.array(A)
    B_ga = ga.GlobalArray.array(B)
    C = A.dot(B)
    C_ga = ga.GlobalArray.array(C)
    if C_ga != A_ga.dot(B_ga):
        (C_ga - A_ga.dot(B_ga)).disp()
        raise Exception("FAIL")
    elif im_root():
        print(i)
