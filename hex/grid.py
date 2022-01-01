import numpy as np

from hex.nodes import Node, ModuleNode


class Grid:
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n), dtype=Node)
        self.shape = self.grid.shape
        for i in range(self.n):
            for j in range(self.n):
                self.grid[i, j] = Node()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.grid[tuple(i)]

    def __setitem__(self, i, v):
        self.grid[tuple(i)] = v

    def __iter__(self):
        self.iter_i = 0
        return self

    def __next__(self):
        # 1d -> 2d iteration
        if self.iter_i < self.n*self.n:
            i,j = self.iter_i//self.n,self.iter_i%self.n
            res = self[i,j]
            self.iter_i += 1
            return res
        else:
            raise StopIteration

    def add_module(self, module):
        """
        Traverse the given module's nodes, adding ModuleNode's to the grid
            as placeholders where indicated.

        :param module: Module to add
        :return: None, grid is modified in place.
        """
        for node in module.nodes:
            self.grid[node] = ModuleNode()

    def in_bounds(self, node):
        """
        :param node: coordinate for a node.
        :return: Return if the given node is inside our given grid bounds.
        """
        return node[0] >= 0 and node[0] < self.n and node[1] >= 0 and node[1] < self.n
