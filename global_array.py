#!/usr/bin/env python
from mpi4py import MPI
import numpy as np


class GloabalArray(object):

    def __init__(self, N, M=None, dtype=None):
        self.comm = MPI.COMM_WORLD
        self.nodes = self.comm.Get_size()
        self.node_id = self.comm.Get_rank()

        self.row_offset = ((N / self.nodes * self.node_id) +
                           min(self.node_id, N % self.nodes))
        self.rows = N / self.nodes + (self.node_id < N % self.nodes)
        M = N if M is None else M
        self.local = np.empty((self.rows, M), dtype)

