"""
Base file for all node definitions. If this extends further, will separate into a directory, like with modules.
    This is one of our highest level classes, since nodes make up modules.
"""
import numpy as np


class Node:
    # inherently located at a cell in the grid
    def __init__(self):
        # Determines if this cell is empty or if this is an active node.
        self.exists = False

        # Output value for this node.
        self.output = 0.0

        #self.activation = self.sigmoid
        #self.aggregation = np.sum
        #self.bias = None
        #self.output_weight = None  # not using this for now

        self.in_edges = np.empty((0,2),np.intp)  # list of inputs for incoming sources, (n,2)
        self.in_weights = np.empty((0,),np.float64)  # list of weights for incoming source weights, (n,)
        self.out_edges = np.empty((0,2),np.intp)  # list of idx for dst node - used for reference when removing nodes. (n,2)


    @staticmethod
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def remove_outgoing(self, node):
        """
        Remove any outgoing edges to the given node address
            Inefficient op on numpy arrays chosen b/c data structure is primarily utilised for it's speed
            in much more commonly used operations for this system.
        :param node: Given target node address
        :return: Nothing, modifies this object in place.
        """
        mask = np.all(self.out_edges!=node, axis=1)#elements to keep
        self.out_edges = self.out_edges[mask]

    def remove_incoming(self, node):
        """
        Remove any incoming edges and weights from the given node address
            Inefficient op on numpy arrays chosen b/c data structure is primarily utilised for it's speed
            in much more commonly used operations for this system.
        :param node: Given target node address
        :return: Nothing, modifies this object in place.
        """
        mask = np.all(self.in_edges!=node, axis=1)#elements to keep
        self.in_edges = self.in_edges[mask]
        self.in_weights = self.in_weights[mask]

    def add_outgoing(self, node):
        """
        Add new out edge to this node.
        :param node: Given target node address
        :return: Nothing, modifies this object in place.
        """
        self.out_edges = np.append(self.out_edges, [node], axis=0)

    def add_incoming(self, node, weight):
        """
        Add new in edge and weight to this node.
        :param node: Given target node address
        :param weight: Given weight value
        :return: Nothing, modifies this object in place.
        """
        self.in_edges = np.append(self.in_edges, [node], axis=0)
        self.in_weights = np.append(self.in_weights, weight)

    def edit_incoming(self, node, weight):
        """
        Edit edge from this node to have the new weight value
        :param node: Given target node address
        :param weight: Given weight value
        :return: Nothing, modifies this object in place.
        """
        mask = np.any(self.in_edges==node, axis=1)#relevant node
        self.in_weights[mask] = weight


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
