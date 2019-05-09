#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import math


class GlobalArray(object):
    def __init__(self, N, M=None, dtype=None, local=None):
        M = N if M is None else M
        self.N = N
        self.M = M

        self.comm = MPI.COMM_WORLD
        self.nodes = self.comm.Get_size()
        self.node_id = self.comm.Get_rank()

        self.row_offset = ((N / self.nodes * self.node_id) +
                           min(self.node_id, N % self.nodes))
        self.rows = N / self.nodes + (self.node_id < N % self.nodes)

        self.local = np.empty(
            (self.rows, M), dtype) if local is None else local

    def __add__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.N, self.M, local=self.local + other.local)
        return GlobalArray(self.N, self.M, local=self.local + other)


    def __radd__(self, other):
        return self + other


    def __sub__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.N, self.M, local=self.local - other.local)
        return GlobalArray(self.N, self.M, local=self.local - other)


    def __rsub__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.N, self.M, local=other.local - self.local)
        return GlobalArray(self.N, self.M, local=other - self.local)


    def __mul__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.N, self.M, local=self.local * other.local)
        return GlobalArray(self.N, self.M, local=self.local * other)


    def __rmul__(self, other):
        return self * other


    def __div__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.N, self.M, local=self.local / other.local)
        return GlobalArray(self.N, self.M, local=self.local / other)


    def __rdiv__(self, other):
        if isinstance(other, GlobalArray):
            return GlobalArray(self.N, self.M, local=other.local / self.local)
        return GlobalArray(self.N, self.M, local=other / self.local)


    def __getitem__(self, key):
        if isinstance(key, slice):
            # Code for slicing
            pass
        elif isinstance(key, int):
            max_id = key / (self.N / self.nodes)
            id_with_local = max_id - (max_id > (key % self.nodes) < (self.N % self.nodes))

            if self.node_id == 0:
                local = np.empty((1, self.M))
                self.comm.Recv([local, MPI.DOUBLE], source=id_with_local)
            else:
                local = None

            if self.node_id == id_with_local:
                self.comm.Send(self.local[key - self.row_offset, :], dest=0)

            return GlobalArray(1, self.M, local=local)


    def disp(self):
        for n in range(self.nodes):
            if n == self.node_id:
                for r in range(self.rows):
                    print("nodeid " + str(n) + ": " + "rownum " +
                          str(r + self.row_offset) + ": " + str(self.local[r]))
            self.comm.Barrier()


    def dot(self, other):
        assert self.M == other.N
        res = GlobalArray(self.N, other.M)
        sizes = np.empty(self.nodes)
        self.comm.Allgather(np.array([self.rows]), sizes)
        offsets = np.empty(self.nodes)
        self.comm.Allgather(np.array([self.row_offset]), offsets)
        print(sizes, offsets)
        current_col = np.empty(other.N)
        # for c in range(other.M):
        #    self.comm.Allgather(other.local[:, c], [current_col, sizes, offsets])


    def _global_to_local(self, y, x):
        for nodeloop in range(self.nodes):
            low_bound = ((self.N / self.nodes * nodeloop) +
                         min(nodeloop, self.N % self.nodes))
            high_bound = low_bound + self.N / self.nodes + \
                (nodeloop < self.N % self.nodes)
            if(low_bound <= y and high_bound > y):
                node = nodeloop
                loc_y = y - low_bound
                loc_x = x
                return node, [loc_y, loc_x]
        raise Exception("y value: " + str(y) +
                        " out of bounds, higher than or eq to" + str(self.N))


    def rref(self):
        eps = 1.0/(10**10)
        error = False
        
        for current_column in range(min(self.N, self.M)):
            mem = np.zeros(self.M)

            current_pivot_node, pivotCoords = self._global_to_local(
                current_column, current_column)
        
            ############# SET MAX PIVOT START ###############
            if self.node_id < current_pivot_node or self.rows < 1: #Node is irrelevant if above pivot
                senddata = np.array([-1, self.node_id])
            elif self.node_id == current_pivot_node: #If node is pivot_node
                a = np.abs(self.local[pivotCoords[0]:self.rows, current_column])
                maxind = np.argmax(a)+pivotCoords[0]
                senddata = np.array([np.amax(a), self.node_id])
            elif self.node_id > current_pivot_node: #If node is under pivot_node
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

            if self.node_id == current_pivot_node: #Check if singular
                if np.abs(self.local[pivotCoords[0],pivotCoords[1]]) <= eps:
                    print("SINGULAR")
                    error = True
            error = self.comm.bcast(error, root=current_pivot_node)
            if(error):
                return False
            
            ############# CHECK SINGULAR END ###############

            ############# ROW REDUCTION START ###############

            if self.node_id == current_pivot_node: 
                mem = self.local[pivotCoords[0],:]

            reduction_row = self.comm.bcast(mem, root=current_pivot_node)

            if self.node_id == current_pivot_node: 
                if pivotCoords[0] != self.rows: # If there is local elimination to be done
                    for local_row in range(pivotCoords[0]+1,self.rows): # Repeat for each local row under pivot
                        c = self.local[local_row,current_column]/reduction_row[current_column]
                        for column in range(current_column,self.M):
                            self.local[local_row,column] -= self.local[pivotCoords[0],column] * c

            if self.node_id > current_pivot_node: # In progress
                for local_row in range(self.rows):
                    c = self.local[local_row,current_column]/reduction_row[current_column]
                    for column in range(current_column,self.M):
                            self.local[local_row,column] -= reduction_row[column] * c
            self.comm.Barrier()
            ############# ROW REDUCTION END ###############

        ############# BACK SUBSTIUTION START ###############
        for current_column in range(min(self.N, self.M)-1,-1,-1):

            current_pivot_node, pivotCoords = self._global_to_local(
                current_column, current_column)

            if self.node_id == current_pivot_node: 
                mem = self.local[pivotCoords[0],:]

            reduction_row = self.comm.bcast(mem, root=current_pivot_node)
            
            if self.node_id == current_pivot_node: 
                for row in range(pivotCoords[0]): # Repeat for each local row over pivot
                    c = self.local[row,current_column]/reduction_row[current_column]
                    for column in range(current_column,self.M):
                        self.local[row,column] -= self.local[pivotCoords[0],column] * c
                self.local[pivotCoords[0],:] /= self.local[pivotCoords[0],pivotCoords[1]]

            if self.node_id < current_pivot_node: # In progress
                for local_row in range(self.rows):
                    c = self.local[local_row,current_column]/reduction_row[current_column]
                    for column in range(current_column,self.M):
                            self.local[local_row,column] -= reduction_row[column] * c

        ############# BACK SUBSTIUTION END ###############       

        if self.node_id == 0:
            print("")
        self.disp()
        pass
