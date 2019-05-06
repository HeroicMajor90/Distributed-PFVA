#!/usr/bin/env python
from mpi4py import MPI
import numpy as np


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

        self.local = np.empty((self.rows, M), dtype) if local is None else local

        
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
          #Code for slicing
          pass
        elif isinstance(key, int):
          	key_in_local = self.row_offset <= key < self.row_offset + self.rows
	        return GlobalArray(
                1, self.M, local=self.local[key - row_offset, :] if key_in_local else None)
      
      
    def disp(self):
        for n in range(self.nodes):
            if n == self.node_id:
                for r in range(self.rows):
                  print(self.local[r])
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
        #for c in range(other.M):
        #    self.comm.Allgather(other.local[:, c], [current_col, sizes, offsets])
        

        
    def rref(self):
      	# lo hare local lol
        pass