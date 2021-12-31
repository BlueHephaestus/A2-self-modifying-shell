import numpy as np

from hex.nodes import Node, ModuleNode


class Grid:
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n), dtype=Node)
        for i in range(self.n):
            for j in range(self.n):
                self.grid[i, j] = Node()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # return self.grid[i]
        return self.grid[tuple(i)]

    def add_module(self, module):
        """
        Traverse the given module's nodes, adding ModuleNode's to the grid
            as placeholders where indicated.

        :param module: Module to add
        :return: None, grid is modified in place.
        """
        for node in module.nodes:
            self.grid[node] = ModuleNode()