#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import math


class GlobalArray(object):


    def _get_rows_per_node(self, total_rows, nodes):
        return [total_rows / nodes + (node_id < total_rows % nodes)
                for node_id in range(nodes)]


    def _get_offsets_per_node(self, total_rows, nodes):
        return [total_rows / nodes * node_id + min(node_id, total_rows % nodes)
                for node_id in range(nodes)]


    def key2id(self, key, total_idxs=None, nodes=None):
        idxs = self.total_rows if not total_rows else total_rows
        nodes = self.nodes if not nodes else nodes

        max_id = key / (idxs / nodes)

        return max_id - (max_id > (key % nodes) < (idxs % nodes))


    def __init__(self, total_rows, total_cols=None, dtype=None, local=None):
        total_cols = total_rows if total_cols is None else total_cols
        self.total_rows = total_rows
        self.total_cols = total_cols

        self.comm = MPI.COMM_WORLD
        self.nodes = self.comm.Get_size()
        self.node_id = self.comm.Get_rank()

        self.rows = self._get_rows_per_node(
            total_rows, self.nodes)[self.node_id]
        self.offset = self._get_offsets_per_node(
            total_rows, self.nodes)[self.node_id]

        self.local = np.empty(
            (self.rows, total_cols), dtype) if local is None else local


    def __add__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows, self.total_cols, local=self.local + other.local)
        return GlobalArray(self.total_rows, self.total_cols, local=self.local + other)


    def __radd__(self, other):
        return self + other


    def __sub__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows, self.total_cols, local=self.local - other.local)
        return GlobalArray(self.total_rows, self.total_cols, local=self.local - other)


    def __rsub__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows, self.total_cols, local=other.local - self.local)
        return GlobalArray(self.total_rows, self.total_cols, local=other - self.local)


    def __mul__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows, self.total_cols, local=self.local * other.local)
        return GlobalArray(self.total_rows, self.total_cols, local=self.local * other)


    def __rmul__(self, other):
        return self * other


    def __div__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows, self.total_cols, local=self.local / other.local)
        return GlobalArray(self.total_rows, self.total_cols, local=self.local / other)


    def __rdiv__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows, self.total_cols, local=other.local / self.local)
        return GlobalArray(self.total_rows, self.total_cols, local=other / self.local)


    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.total_rows)
            assert step > 0, 'Negative steps are not currently supported'

            start = start + self.total_rows if start < 0 else start
            stop = stop + self.total_rows if stop < 0 else stop

            assert -1 < start and -1 < stop, 'Indices out of range'


        elif isinstance(key, int):
            id_with_local = self.key2id(key)
            if self.node_id == 0:
                local = np.empty((1, self.total_cols))
                self.comm.Recv([local, MPI.DOUBLE], source=id_with_local)
            else:
                local = None

            if self.node_id == id_with_local:
                self.comm.Send(self.local[key - self.row_offset, :], dest=0)

            return GlobalArray(1, self.total_cols, local=local)

        elif isinstance(key, tuple):
            raise TypeError, 'Fancy slicing hasn"t been implemented'
        else:
            raise TypeError, 'Global Arrays indices must be integers or slice'


    def disp(self):
        for n in range(self.nodes):
            if n == self.node_id:
                for r in range(self.rows):
                    print("nodeid " + str(n) + ": " + "rownum " +
                          str(r + self.offset) + ": " + str(self.local[r]))
            self.comm.Barrier()


    def dot(self, other):
        assert self.total_cols == other.total_rows
        res = GlobalArray(self.total_rows, other.total_cols)

        local_size = np.array([other.rows])
        sizes = np.empty(other.nodes, local_size.dtype)
        self.comm.Allgather(local_size, sizes)

        local_offset = np.array([other.offset])
        offsets = np.empty(other.nodes, local_offset.dtype)
        self.comm.Allgather(local_offset, offsets)

        current_col = np.empty(other.total_rows, np.float64)
        for c in range(other.total_cols):
            local_current_col = other.local[:, c].copy()
            print(local_current_col)
            self.comm.Allgatherv(
                local_current_col, [current_col, other.rows, other.offset, MPI.DOUBLE])
            print(current_col)
            self.comm.Barrier()

        return res


    def _global_to_local(self, y, x):
        for nodeloop in range(self.nodes):
            low_bound = ((self.total_rows / self.nodes * nodeloop) +
                         min(nodeloop, self.total_rows % self.nodes))
            high_bound = low_bound + self.total_rows / self.nodes + \
                         (nodeloop < self.total_rows % self.nodes)
            if (low_bound <= y and high_bound > y):
                node = nodeloop
                loc_y = y - low_bound
                loc_x = x
                return node, [loc_y, loc_x]
        raise Exception("y value: " + str(y) +
                        " out of bounds, higher than or eq to" + str(self.total_rows))


    def rref(self):
        eps = 1.0 / (10 ** 10)
        error = False

        for current_column in range(min(self.total_rows, self.total_cols)):
            mem = np.zeros(self.total_cols)

            current_pivot_node, pivotCoords = self._global_to_local(
                current_column, current_column)

            ############# SET MAX PIVOT START ###############
            if self.node_id < current_pivot_node or self.rows < 1:  # Node is irrelevant if above pivot
                senddata = np.array([-1, self.node_id])
            elif self.node_id == current_pivot_node:  # If node is pivot_node
                a = np.abs(self.local[pivotCoords[0]:self.rows, current_column])
                maxind = np.argmax(a) + pivotCoords[0]
                senddata = np.array([np.amax(a), self.node_id])
            elif self.node_id > current_pivot_node:  # If node is under pivot_node
                a = np.abs(self.local[:self.rows, current_column])
                maxind = np.argmax(a)
                senddata = np.array([np.amax(a), self.node_id])
            else:
                raise Exception("MPI rank error")

            self.comm.Barrier()

            recvdata = self.comm.allreduce(senddata, op=MPI.MAXLOC)

            if current_pivot_node == recvdata[1]:  # If exchange is local
                if self.node_id == recvdata[1] and pivotCoords[0] != maxind:
                    self.local[[maxind, pivotCoords[0]],
                    :] = self.local[[pivotCoords[0], maxind], :]
            else:  # If exchange is between nodes
                if self.node_id == recvdata[1]:  # If, maxrow node
                    sendrow = self.local[maxind, :]

                    self.comm.Sendrecv(
                        sendrow, dest=current_pivot_node, recvbuf=mem, source=current_pivot_node)

                    self.local[maxind, :] = mem
                if self.node_id == current_pivot_node:  # If, pivot node

                    sendrow = self.local[pivotCoords[0], :]

                    self.comm.Sendrecv(
                        sendrow, dest=recvdata[1], recvbuf=mem, source=recvdata[1])

                    self.local[pivotCoords[0], :] = mem
            self.comm.Barrier()
            ############# SET MAX PIVOT END ###############

            ############# CHECK SINGULAR START ###############

            if self.node_id == current_pivot_node:  # Check if singular
                if np.abs(self.local[pivotCoords[0], pivotCoords[1]]) <= eps:
                    print("SINGULAR")
                    error = True
            error = self.comm.bcast(error, root=current_pivot_node)
            if (error):
                return False

            ############# CHECK SINGULAR END ###############

            ############# ROW REDUCTION START ###############

            if self.node_id == current_pivot_node:
                mem = self.local[pivotCoords[0], :]

            reduction_row = self.comm.bcast(mem, root=current_pivot_node)

            if self.node_id == current_pivot_node:
                if pivotCoords[0] != self.rows:  # If there is local elimination to be done
                    for local_row in range(pivotCoords[0] + 1, self.rows):  # Repeat for each local row under pivot
                        c = self.local[local_row, current_column] / reduction_row[current_column]
                        for column in range(current_column, self.total_cols):
                            self.local[local_row, column] -= self.local[pivotCoords[0], column] * c

            if self.node_id > current_pivot_node:  # In progress
                for local_row in range(self.rows):
                    c = self.local[local_row, current_column] / reduction_row[current_column]
                    for column in range(current_column, self.total_cols):
                        self.local[local_row, column] -= reduction_row[column] * c
            self.comm.Barrier()
            ############# ROW REDUCTION END ###############

        ############# BACK SUBSTIUTION START ###############
        for current_column in range(min(self.total_rows, self.total_cols) - 1, -1, -1):

            current_pivot_node, pivotCoords = self._global_to_local(
                current_column, current_column)

            if self.node_id == current_pivot_node:
                mem = self.local[pivotCoords[0], :]

            reduction_row = self.comm.bcast(mem, root=current_pivot_node)

            if self.node_id == current_pivot_node:
                for row in range(pivotCoords[0]):  # Repeat for each local row over pivot
                    c = self.local[row, current_column] / reduction_row[current_column]
                    for column in range(current_column, self.total_cols):
                        self.local[row, column] -= self.local[pivotCoords[0], column] * c
                self.local[pivotCoords[0], :] /= self.local[pivotCoords[0], pivotCoords[1]]

            if self.node_id < current_pivot_node:  # In progress
                for local_row in range(self.rows):
                    c = self.local[local_row, current_column] / reduction_row[current_column]
                    for column in range(current_column, self.total_cols):
                        self.local[local_row, column] -= reduction_row[column] * c

        ############# BACK SUBSTIUTION END ###############

        if self.node_id == 0:
            print("")
        self.disp()
