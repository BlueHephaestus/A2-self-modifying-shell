import numpy as np


class Node:
    # inherently located at a cell in the grid
    def __init__(self):
        # Determines if this cell is empty or if this is an active node.
        self.exists = False

        # Output value for this node.
        self.output = None

        self.activation = self.sigmoid
        self.aggregation = sum
        self.bias = None
        self.output_weight = None  # not using this for now

        self.in_edges = []  # list of form (idx, weight) for source node and weight from it
        self.out_edges = []  # list of idx for dst node - used for reference when removing nodes.

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def get_input(self, grid):
        """
        Apply propagation from our source nodes to this node via edges,
        Then apply this node's aggregation function to combine them into one input.
        :param grid: Hex Grid this node exists on.
        :return: input value computed from this node's connections.
        """
        # Gonna make it a for loop for more clarity.
        # node_inputs = [grid[i].output * w for i, w in self.edges if grid[i].output is not None]

        node_inputs = []
        for input_idx, weight in self.in_edges:
            if grid[input_idx].output is not None:
                node_inputs.append(grid[input_idx].output * weight)
        return self.aggregation(node_inputs)


class ModuleNode(Node):
    def __init__(self):
        """
        Placeholder node with varying purposes, for use with a module at this location.
            Since it will never output anything, has different attributes:
            No output, only input
            No activation function, no bias
            No output weight.
            Always starts with exists=True

        Shares subclass for Grid's usage.
        """
        Node.__init__(self)
        self.exists = True

        # Input value for this node.
        self.input = None
        self.aggregation = sum