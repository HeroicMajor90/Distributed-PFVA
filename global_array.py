#!/usr/bin/env python
from mpi4py import MPI
import numpy as np


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
        assert isinstance(key, slice) or isinstance(key, (int, np.integer)), (
                "Keys must be slices or integers")
        return key if isinstance(key, slice) else slice(key, key+1)


    def _slice_array(self, slice_axis1, slice_axis2=None):
        start, stop, step = slice_axis1.indices(self.total_rows)
        slice_axis2 = slice_axis2 if slice_axis2 else slice(self.total_cols)
        assert step > 0, "Negative steps are not currently supported"

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

        sendbuf = np.asarray(rows_to_send, dtype=np.float64)

        recvbuf = np.empty((r_amount.sum(), new_total_cols), dtype=np.float64)

        s_offsets = self._cumsum(s_amount)*new_total_cols
        r_offsets = self._cumsum(r_amount)*new_total_cols
        s_amount *= new_total_cols
        r_amount *= new_total_cols

        self.comm.Alltoallv(
            [sendbuf, s_amount, s_offsets, MPI.DOUBLE],
            [recvbuf, r_amount, r_offsets, MPI.DOUBLE])

        return GlobalArray(new_total_rows, new_total_cols, local=recvbuf)


    def _setslice_array(self, slice_axis1, slice_axis2=None, ga=None):
        start, stop, step = slice_axis1.indices(self.total_rows)
        slice_axis2 = slice_axis2 if slice_axis2 else slice(self.total_cols)
        assert step > 0, "Negative steps are not currently supported"

        start = start + self.total_rows if start < 0 else max(start, 0)
        stop = (stop + self.total_rows
                if stop < 0 else
                min(stop, self.total_rows))

        s_amount = np.zeros(self.nodes).astype(int)
        r_amount = np.zeros(self.nodes).astype(int)

        new_total_cols = len(range(*slice_axis2.indices(self.total_cols)))
        rows_to_send = []

        for s_idx, r_idx in enumerate(range(start, stop, step)):
            s_node = self._row2nodeid(s_idx, ga.total_rows)
            r_node = self._row2nodeid(r_idx)

            if s_node == self.node_id:
                rows_to_send.append(ga.local[s_idx - ga.offset, :])
                s_amount[r_node] += 1

            if r_node == self.node_id:
                r_amount[s_node] += 1

        sendbuf = np.asarray(rows_to_send, dtype=np.float64)

        recvbuf = np.empty((r_amount.sum(), new_total_cols), dtype=np.float64)

        s_offsets = self._cumsum(s_amount)*new_total_cols
        r_offsets = self._cumsum(r_amount)*new_total_cols
        s_amount *= new_total_cols
        r_amount *= new_total_cols

        self.comm.Alltoallv(
            [sendbuf, s_amount, s_offsets, MPI.DOUBLE],
            [recvbuf, r_amount, r_offsets, MPI.DOUBLE])

        return recvbuf


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


    def copy(self):
        return GlobalArray(
            self.total_rows, self.total_cols, local=self.local.copy())


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
        return ga


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


    def diagonal(self):
        assert self.total_rows == self.total_cols
        ga = GlobalArray(self.total_rows, 1)
        for row in range(self.rows):
            ga.local[row, 0] = self.local[row, self.offset + row] 
        return ga 


    def to_np(self):
        np_array = np.empty((self.total_rows, self.total_cols), np.float64)
        recv_elems = [self.total_cols * rows for rows in
                      self._get_rows_per_node(self.total_rows, self.nodes)]
        recv_offsets = [self.total_cols * offset for offset in
                        self._get_offsets_per_node(self.total_rows, self.nodes)]
        self.comm.Allgatherv(
            self.local, [np_array, recv_elems, recv_offsets, MPI.DOUBLE])
        return np_array


    def __add__(self, other):
        if isinstance(other, GlobalArray):
            other = other.to_np() if other.total_rows == 1 else other.local
        return GlobalArray(
            self.total_rows, self.total_cols, local=self.local + other)


    def __radd__(self, other):
        return self + other


    def __sub__(self, other):
        if isinstance(other, GlobalArray):
            other = other.to_np() if other.total_rows == 1 else other.local
        return GlobalArray(
            self.total_rows, self.total_cols, local=self.local - other)


    def __rsub__(self, other):
        return GlobalArray(
            self.total_rows, self.total_cols, local=other - self.local)


    def __mul__(self, other):
        if isinstance(other, GlobalArray):
            other = other.to_np() if other.total_rows == 1 else other.local
        return GlobalArray(
            self.total_rows, self.total_cols, local=self.local * other)


    def __rmul__(self, other):
        if isinstance(other, GlobalArray):
            other = other.to_np() if other.total_rows == 1 else other.local
        return GlobalArray(
            self.total_rows, self.total_cols, local=other * self.local)


    def __div__(self, other):
        if isinstance(other, GlobalArray):
            other = other.to_np() if other.total_rows == 1 else other.local
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
            raise TypeError("Global Arrays indices must be integers or slice")


    def __setitem__(self, key, value):
        if isinstance(value, (int, np.integer)):
            value = GlobalArray.array(np.array([value])[np.newaxis])
        if isinstance(key, slice):
            recvbuf = self._setslice_array(key, ga=value)
            idxs = [idx for idx in range(*key.indices(self.total_rows)) 
                    if self._row2nodeid(idx) == self.node_id]
            for idx, row in zip(idxs, recvbuf):
                self.local[idx - self.offset] = row

        elif isinstance(key, (int, np.integer)):
            recvbuf = self._setslice_array(self._key2slice(key), ga=value)
            if self._row2nodeid(key) == self.node_id:
                self.local[key - self.offset] = recvbuf

        elif isinstance(key, tuple):
            key = [self._key2slice(axis_key) for axis_key in key]
            recvbuf = self._setslice_array(*key, ga=value)
            idxs = [idx for idx in range(*key[0].indices(self.total_rows)) 
                    if self._row2nodeid(idx) == self.node_id]
            for idx, row in zip(idxs, recvbuf):
                self.local[idx-self.offset, key[1]] = row
        else:
            raise TypeError("Global Arrays indices must be integers or slice")


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

    def mean(self, axis=None):
        # axis =    None is average of flattened array
        #      =    0 is column wise
        #      =    1 is row wise
        if axis == 0:
            col_mean = GlobalArray(self.total_cols, 1)
            local_sum = np.sum(self.local, axis=0)
            global_sum = np.empty(self.total_cols, np.float64)
            self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
            mean_vec = global_sum / self.total_rows
            for col in range(col_mean.rows):
                col_mean.local[col, 0] = mean_vec[col + col_mean.offset]
            return col_mean
        else:
            row_mean = GlobalArray(self.total_rows, 1)
            row_mean.local[:, 0] = np.mean(self.local, axis=1)
            if axis == 1:
                return row_mean
            else:
                return row_mean.mean(axis=0)


    def std(self, axis=None, ddof=0, zero_default=0):
        if axis == 1:
            row_std = GlobalArray(self.total_rows, 1)
            row_std.local[:, 0] = np.std(self.local, axis=1, ddof=ddof)
            row_std.local = np.where(row_std.local == 0, zero_default,
                                     row_std.local)
            return row_std
        else:
            col_mean = self.mean(axis)
            col_mean = col_mean.to_np()
            global_sum = np.empty(self.total_cols, np.float64)
            local_copy = (self.local - col_mean.flatten()) ** 2
            local_sum = np.sum(local_copy, axis=0)
            self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
            if axis == 0:
                col_std = GlobalArray(self.total_cols, 1)
                std_vec = np.sqrt(global_sum / (self.total_rows - ddof))
                std_vec = np.where(std_vec == 0, zero_default, std_vec)
                for col in range(col_std.rows):
                    col_std.local[col, 0] = std_vec[col + col_std.offset]
                return col_std
            else:
                col_std = GlobalArray(1, 1)
                varis = np.sqrt(np.sum(global_sum) /
                                (self.total_rows * self.total_cols - ddof))
                varis = np.where(varis == 0, zero_default, varis)
                if self.node_id == 0:
                    col_std.local[0, 0] = varis
                return col_std


    def rref(self):
        eps = 1.0 / (10 ** 10)
        error = np.array([False])
        maxcol = min(self.total_rows, self.total_cols)
        
        for current_column in range(maxcol):
            mem = np.zeros(self.total_cols)
            current_pivot_node = self._row2nodeid(current_column)
            local_row = self._get_offsets_per_node(self.total_rows, self.nodes)
            pivot_coords = [current_column - local_row[current_pivot_node],
                            current_column]

            # Check Singular
            if self.node_id == current_pivot_node:  # Check if singular
                if np.abs(self.local[pivot_coords[0], pivot_coords[1]]) <= eps:
                    
                    # Set Max Pivot
                    if self.node_id < current_pivot_node or self.rows < 1:
                        # Node is irrelevant if above pivot
                        senddata = np.array([0, self.node_id], 
                                            dtype=np.float64)
                    elif self.node_id == current_pivot_node:  
                        # If node is pivot_node
                        a = np.abs(self.local[pivot_coords[0]:self.rows,
                                              current_column])
                        maxind = np.argmax(a) + pivot_coords[0]
                        senddata = np.array([np.amax(a), self.node_id])
                    elif self.node_id > current_pivot_node:
                        # If node is under pivot_node
                        a = np.abs(self.local[:self.rows, current_column])
                        maxind = np.argmax(a)
                        senddata = np.array([np.amax(a), self.node_id])
                    else:
                        raise Exception("MPI rank error")

                    recvdata = np.empty(2 * self.nodes)
                    self.comm.Allgather(senddata, recvdata)
                    maxnode = np.argmax(recvdata[0::2])
                    maxnode = recvdata[maxnode * 2 + 1]

                    if current_pivot_node == maxnode:  # If exchange is local
                        if (self.node_id == maxnode and
                                pivot_coords[0] != maxind):
                            self.local[[maxind, pivot_coords[0]], :] = (
                                self.local[[pivot_coords[0], maxind], :])
                    else:  # If exchange is between nodes
                        if self.node_id == maxnode:  # If, maxrow node
                            sendrow = self.local[maxind, :]

                            self.comm.Sendrecv(
                                sendrow, dest=current_pivot_node,
                                recvbuf=mem, source=current_pivot_node)

                            self.local[maxind, :] = mem
                        if self.node_id == current_pivot_node: 
                            # If, pivot node
                            sendrow = self.local[pivot_coords[0], :]
                            self.comm.Sendrecv(sendrow, dest=maxnode,
                                               recvbuf=mem, source=maxnode)
                            self.local[pivot_coords[0], :] = mem
                if np.abs(self.local[pivot_coords[0], pivot_coords[1]]) <= eps:
                    print("SINGULAR")
                    error = np.array([True])
            self.comm.Bcast(error, root=current_pivot_node)
            if error[0]:
                return False

            # Row Reduction
            reduction_row = np.empty(self.total_cols,dtype=np.float64)

            if self.node_id == current_pivot_node:
                reduction_row = self.local[pivot_coords[0], :]

            self.comm.Bcast(reduction_row, root=current_pivot_node)

            if self.node_id == current_pivot_node:
                # If there is local elimination to be done
                if pivot_coords[0] != self.rows:
                    # Repeat for each local row under pivot
                    c = (self.local[pivot_coords[0] + 1:self.rows,
                                    current_column] /
                         reduction_row[current_column])
                    c = c.reshape(-1, 1)
                    pivot_row = np.tile(reduction_row,(c.size,1))
                    self.local[pivot_coords[0] + 1:self.rows, :] -= (
                            pivot_row * c)

            if self.node_id > current_pivot_node:
                c = (self.local[:self.rows, current_column] /
                     reduction_row[current_column])
                c = c.reshape(-1, 1)
                pivot_row = np.tile(reduction_row,(c.size, 1))
                self.local[:self.rows, :] -= (pivot_row * c)

            # Back Substitution
            if self.node_id == current_pivot_node:
                # Repeat for each local row under pivot
                c = (self.local[:pivot_coords[0], current_column] /
                     reduction_row[current_column])
                c = c.reshape(-1, 1)
                pivot_row = np.tile(reduction_row, (c.size, 1))
                self.local[:pivot_coords[0], :] -= (pivot_row * c)

                self.local[pivot_coords[0], :] /= self.local[pivot_coords[0],
                                                             pivot_coords[1]]

            if self.node_id < current_pivot_node:
                c = (self.local[:self.rows,current_column] /
                     reduction_row[current_column])
                c = c.reshape(-1, 1)
                pivot_row = np.tile(reduction_row, (c.size, 1))
                self.local[:self.rows, :] -= (pivot_row * c)


def qr(A):
    assert A.total_rows >= A.total_cols
    R = A.copy()
    V = GlobalArray.zeros(R.total_rows, R.total_cols)
    for k in range(R.total_cols):
        y = R[k:, k]
        e = GlobalArray.zeros(y.total_rows, 1)
        e[0] = 1
        sign = np.sign(y[0].to_np()) if y[0].to_np() != 0 else 1
        w = y + float(sign) * float(np.sqrt(y.transpose().dot(y).to_np())) * e
        v = w / float(np.sqrt(w.transpose().dot(w).to_np()))
        V[k:, k] = v
        Rk = R[k:, k:]
        R[k:, k:] = Rk - 2 * v.dot(v.transpose().dot(Rk))
    Q = GlobalArray.eye(R.total_rows)
    for k in range(R.total_cols - 1, -1, -1):
        v = V[k:, k]
        Qk = Q[k:, k:]
        Q[k:, k:] = Qk - 2 * v.dot(v.transpose().dot(Qk))
    return Q, R


def sort_by_first_column(A):
    rows_per_node = A._get_rows_per_node(A.total_rows, A.nodes)
    local = A.local[A.local[:, 0].argsort()]
    d = 1
    while d < A.nodes:
        if (A.node_id - d) % (2 * d) == 0 and A.node_id - d >= 0:
            A.comm.Send(local, dest=A.node_id - d)
        elif A.node_id % (2 * d) == 0 and A.node_id + d < A.nodes:
            other = np.empty(
                (sum(rows_per_node[A.node_id + d:A.node_id + 2 * d]),
                 A.total_cols),
                np.float64)
            A.comm.Recv(other, source=A.node_id + d)
            new_local = np.empty(
                (local.shape[0] + other.shape[0], A.total_cols), np.float64)
            local_idx = 0
            other_idx = 0
            while local_idx < local.shape[0] and other_idx < other.shape[0]:
                if local[local_idx, 0] < other[other_idx, 0]:
                    new_local[local_idx + other_idx] = local[local_idx]
                    local_idx += 1
                else:
                    new_local[local_idx + other_idx] = other[other_idx]
                    other_idx += 1
            while local_idx < local.shape[0]:
                new_local[local_idx + other_idx] = local[local_idx]
                local_idx += 1
            while other_idx < other.shape[0]:
                new_local[local_idx + other_idx] = other[other_idx]
                other_idx += 1
            local = new_local
        d *= 2
    sorted_array = (local if A.node_id == 0
                    else np.empty((A.total_rows, A.total_cols), np.float64))
    A.comm.Bcast(sorted_array)
    return GlobalArray.array(sorted_array)
