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


    def _row2nodeid(self, key, total_rows=None, nodes=None):
        rows = self.total_rows if not total_rows else total_rows
        nodes = self.nodes if not nodes else nodes
        for node_id in range(nodes):
            if key < rows / nodes * node_id + min(node_id, rows % nodes):
                return node_id - 1

        return node_id


    def _cumsum(self, array):
        cumsum = np.zeros(len(array))
        cumsum[1:] = np.cumsum(array[:-1])
        return cumsum


    def _key2slice(self, key):
        assert (isinstance(key, slice) or isinstance(key, (int, np.integer)),
                'Keys must be slices or integers')
        
        return key if isinstance(key, slice) else slice(key, key+1)


    def _slice_array(self, slice_axis1, slice_axis2=None):
        start, stop, step = slice_axis1.indices(self.total_rows)
        slice_axis2 = slice_axis2 if slice_axis2 else slice(self.total_cols)
        assert step > 0, 'Negative steps are not currently supported'

        start = start + self.total_rows if start < 0 else max(start, 0)
        stop = (stop + self.total_rows
                if stop < 0 else
                min(stop, self.total_rows))

        s_amount = np.zeros(self.nodes).astype(int)
        r_amount = np.zeros(self.nodes).astype(int)

        new_total_rows = len(range(start, stop, step))
        new_total_cols = len(range(*slice_axis2.indices(self.total_cols)))
        rows_to_send = []

        for new_idx, idx in enumerate(range(start, stop, step)):
            s_node = self._row2nodeid(idx)
            r_node = self._row2nodeid(new_idx, new_total_rows)

            if s_node == self.node_id:
                rows_to_send.append(self.local[idx - self.offset, slice_axis2])
                s_amount[r_node] += 1

            if r_node == self.node_id:
                r_amount[s_node] += 1

        sendbuf = np.asarray(
            rows_to_send if rows_to_send else self.local, dtype=np.float64)

        recvbuf = np.empty(
            (max(r_amount.sum(), 1), new_total_cols), dtype=np.float64)

        s_offsets = self._cumsum(s_amount)*new_total_cols
        r_offsets = self._cumsum(r_amount)*new_total_cols
        s_amount *= new_total_cols
        r_amount *= new_total_cols

        self.comm.Alltoallv(
            [sendbuf, s_amount, s_offsets, MPI.DOUBLE],
            [recvbuf, r_amount, r_offsets, MPI.DOUBLE])

        return GlobalArray(new_total_rows, self.total_cols, local=recvbuf)


    def __init__(self, total_rows, total_cols=None, local=None):
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
            (self.rows, total_cols), np.float64) if local is None else local


    @classmethod
    def zeros(cls, total_rows, total_cols=None):
        ga = cls(total_rows, total_cols)
        ga.local[:, :] = 0
        return ga


    @classmethod
    def ones(cls, total_rows, total_cols=None):
        ga = cls(total_rows, total_cols)
        ga.local[:, :] = 1
        return ga


    @classmethod
    def eye(cls, total_rows):
        ga = cls.zeros(total_rows, total_rows)
        for row in range(ga.rows):
            ga.local[row, ga.offset + row] = 1
        return ga.local.ndim


    @classmethod
    def array(cls, total_array):
        total_array = np.array(total_array)
        assert total_array.ndim == 2
        ga = cls(total_array.shape[0], total_array.shape[1])
        ga.local[:] = total_array[ga.offset:ga.offset + ga.rows]
        return ga


    @classmethod
    def from_file(cls, filename, **kwargs):
        return cls.array(np.loadtxt(filename, **kwargs))


    def __add__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows,
                               self.total_cols,
                               local=self.local + other.local)
        return GlobalArray(
            self.total_rows, self.total_cols, local=self.local + other)


    def __radd__(self, other):
        return self + other


    def __sub__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows,
                               self.total_cols,
                               local=self.local - other.local)
        return GlobalArray(
            self.total_rows, self.total_cols, local=self.local - other)


    def __rsub__(self, other):
        return GlobalArray(
            self.total_rows, self.total_cols, local=other - self.local)


    def __mul__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows,
                               self.total_cols,
                               local=self.local * other.local)
        return GlobalArray(
            self.total_rows, self.total_cols, local=self.local * other)


    def __rmul__(self, other):
        return self * other


    def __div__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.total_rows,
                               self.total_cols,
                               local=self.local / other.local)
        return GlobalArray(
            self.total_rows, self.total_cols, local=self.local / other)


    def __rdiv__(self, other):
        return GlobalArray(
            self.total_rows, self.total_cols, local=other / self.local)


    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._slice_array(key)

        elif isinstance(key, (int, np.integer)):
            return self._slice_array(self._key2slice(key))

        elif isinstance(key, tuple):
            return self._slice_array(
                *(self._key2slice(axis_key) for axis_key in key))
        else:
            raise TypeError, 'Global Arrays indices must be integers or slice'


    def __setitem__(self, ):
        pass


    def __eq__(self, other):  # Total-wise
        local_eq = np.array([np.allclose(self.local, other.local)])
        global_eq = np.empty(1, dtype=bool)
        self.comm.Allreduce(local_eq, global_eq, op=MPI.LAND)
        return global_eq[0]


    def __ne__(self, other):  # Total-wise
        return not (self == other)


    def disp(self):
        for node_id in range(self.nodes):
            if node_id == self.node_id:
                for row in range(self.rows):
                    print("nodeid " + str(node_id) + ": " + "rownum " +
                          str(row + self.offset) + ": " + str(self.local[row]))
            self.comm.Barrier()


    def transpose(self):
        res = GlobalArray(self.total_cols, self.total_rows)
        self_rows = self._get_rows_per_node(self.total_rows, self.nodes)
        self_offsets = self._get_offsets_per_node(self.total_rows, self.nodes)

        current_col = np.empty(self.total_rows, np.float64)
        for col in range(self.total_cols):
            local_current_col = self.local[:, col].copy()
            self.comm.Allgatherv(
                local_current_col,
                [current_col, self_rows, self_offsets, MPI.DOUBLE])
            if res.offset <= col < res.offset + res.rows:
                res.local[col - res.offset, :] = current_col.T
        return res


    def dot(self, other):
        assert self.total_cols == other.total_rows
        res = GlobalArray(self.total_rows, other.total_cols)
        other_rows = self._get_rows_per_node(other.total_rows, other.nodes)
        other_offsets = self._get_offsets_per_node(other.total_rows,
                                                   other.nodes)

        current_col = np.empty(other.total_rows, np.float64)
        for col in range(other.total_cols):
            local_current_col = other.local[:, col].copy()
            self.comm.Allgatherv(
                local_current_col,
                [current_col, other_rows, other_offsets, MPI.DOUBLE])
            for row in range(self.rows):
                res.local[row, col] = self.local[row].dot(current_col)
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
        raise Exception("y value: "
                        + str(y)
                        + " out of bounds, higher than or eq to"
                        + str(self.total_rows))


    def rref(self):
        eps = 1.0 / (10 ** 10)
        error = np.array([False])

        for current_column in range(min(self.total_rows, self.total_cols)):
            mem = np.zeros(self.total_cols)

            current_pivot_node, pivotCoords = self._global_to_local(
                current_column, current_column)

            ############# SET MAX PIVOT START ###############
            if self.node_id < current_pivot_node or self.rows < 1:
                # Node is irrelevant if above pivot
                senddata = np.array([0, self.node_id],dtype=np.float64)
            elif self.node_id == current_pivot_node:  # If node is pivot_node
                a = np.abs(self.local[pivotCoords[0]:self.rows, current_column])
                maxind = np.argmax(a) + pivotCoords[0]
                senddata = np.array([np.amax(a), self.node_id])
            elif self.node_id > current_pivot_node:
                # If node is under pivot_node
                a = np.abs(self.local[:self.rows, current_column])
                maxind = np.argmax(a)
                senddata = np.array([np.amax(a), self.node_id])
            else:
                raise Exception("MPI rank error")

            recvdata = np.empty(2*self.nodes)
            self.comm.Barrier()
            self.comm.Allgather(senddata,recvdata)
            maxnode = np.argmax(recvdata[0::2])
            maxnode = recvdata[maxnode*2+1]

            if current_pivot_node == maxnode:  # If exchange is local
                if self.node_id == maxnode and pivotCoords[0] != maxind:
                    self.local[[maxind, pivotCoords[0]],
                    :] = self.local[[pivotCoords[0], maxind], :]
            else:  # If exchange is between nodes
                if self.node_id == maxnode:  # If, maxrow node
                    sendrow = self.local[maxind, :]

                    self.comm.Sendrecv(
                        sendrow, dest=current_pivot_node, recvbuf=mem,
                        source=current_pivot_node)

                    self.local[maxind, :] = mem
                if self.node_id == current_pivot_node:  # If, pivot node

                    sendrow = self.local[pivotCoords[0], :]

                    self.comm.Sendrecv(
                        sendrow, dest=maxnode, recvbuf=mem, source=maxnode)

                    self.local[pivotCoords[0], :] = mem
            self.comm.Barrier()
            ############# SET MAX PIVOT END ###############

            ############# CHECK SINGULAR START ###############

            if self.node_id == current_pivot_node:  # Check if singular
                if np.abs(self.local[pivotCoords[0], pivotCoords[1]]) <= eps:
                    print("SINGULAR")
                    error = np.array([True])
            self.comm.Bcast(error, root=current_pivot_node)
            if (error[0]):
                return False

            ############# CHECK SINGULAR END ###############

            ############# ROW REDUCTION START ###############

            reduction_row = np.empty(self.total_cols,dtype=np.float64)

            if self.node_id == current_pivot_node:
                reduction_row = self.local[pivotCoords[0], :] 

            self.comm.Bcast(reduction_row, root=current_pivot_node)

            if self.node_id == current_pivot_node:
                # If there is local elimination to be done
                if pivotCoords[0] != self.rows:
                    # Repeat for each local row under pivot
                    for local_row in range(pivotCoords[0] + 1, self.rows):
                        c = (self.local[local_row, current_column]
                             / reduction_row[current_column])
                        for column in range(current_column, self.total_cols):
                            self.local[local_row, column] -= (
                                self.local[pivotCoords[0], column] * c)

            if self.node_id > current_pivot_node:  # In progress
                for local_row in range(self.rows):
                    c = (self.local[local_row, current_column]
                         / reduction_row[current_column])
                    for column in range(current_column, self.total_cols):
                        self.local[local_row, column] -= (
                            reduction_row[column] * c)
            self.comm.Barrier()
            ############# ROW REDUCTION END ###############

        ############# BACK SUBSTIUTION START ###############
        for current_column in range(
                min(self.total_rows, self.total_cols) - 1, -1, -1):
            current_pivot_node, pivotCoords = self._global_to_local(
                current_column, current_column)

            reduction_row = np.empty(self.total_cols,dtype=np.float64)

            if self.node_id == current_pivot_node:
                reduction_row = self.local[pivotCoords[0], :]

            self.comm.Bcast(reduction_row, root=current_pivot_node)

            if self.node_id == current_pivot_node:
                # Repeat for each local row over pivot
                for row in range(pivotCoords[0]):
                    c = (self.local[row, current_column]
                         / reduction_row[current_column])
                    for column in range(current_column, self.total_cols):
                        self.local[row, column] -= (
                            self.local[pivotCoords[0], column] * c)
                self.local[pivotCoords[0], :] /= self.local[pivotCoords[0],
                                                            pivotCoords[1]]

            if self.node_id < current_pivot_node:  # In progress
                for local_row in range(self.rows):
                    c = (self.local[local_row, current_column]
                         / reduction_row[current_column])
                    for column in range(current_column, self.total_cols):
                        self.local[local_row, column] -= (
                            reduction_row[column] * c)
