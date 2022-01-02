"""
Hex - is a weird RNN. This is because for starters i'm trying to maximize greppability
    greppable - to be easy to use, diagnose, understand, work with, handle, etc.

This means that there is a highly limited size of it, since there can only be 256 neurons total in a given Hex.
    it can of course be easily extended i'm just doing it this way so i can view the entire network easily,
    and because rather than extending the range of each dimension i'll probably add dimensions instead. not sure.
"""
import numpy as np
from scipy.special import expit as sigmoid
from tqdm import tqdm

from hex.modules.edge import EdgeModule
from hex.modules.io import Inputs, Outputs
from hex.modules.memory import MemoryNode, MemoryModule
from hex.modules.meta import MetaModule
from hex.modules.node import NodeModule
from hex.nodes import Node, ModuleNode
from hex.rendering import NetworkRenderer
from hex.rng import rng_hex_core, rng_hex_connect_core

"""
Underlying representation of RNNs in this system:
    Can be whatever, so long as we can convert it to NetworkX structure, since that is what we use to render.
    And since we are having to do a lil bit of a hacky thing to use NetworkX with our rendering, via naming each
        node as it's position in the grid, 
    I'm going to instead opt for making this a different more efficient representation, with a converter that can
        create a human-viewable networkx graph of it when rendering.
    We'll still use an NxN grid for all of this
    However we'll have a LOT of different types of given nodes in that grid.
    Not sure yet how we should do the node vs. module distinction, since multiple nodes make up a module. 
        We could always do our quadrant idea - no, this wouldn't work for when it creates its own.
        If it creates a new module in an area, that will only work if there is room. 
            It will try and create it with the given address being the top-left corner of the module.
                (yes, this means each module should have a spec saying what it's surface area is, so we can eval
                if a new addition will work)
        So nodes should obviously be related to modules that they are under.
        This highly relates to when we check for if a module is triggered.
            REMEMBER that modules only get triggered when their threshold is exceeded
        After we run propagate(), we check each module. This would involve looping through a list of existing
            modules in the grid, and checking their object to see if the threshold has been exceeded, and if it has,
            only then does it care about grabbing and putting together all the inputs it has to it's module.
"""


"""
HexNetwork
    Has objects as instance:
        Grid
        Module Subclasses
        Nodes
    Nodes being the main thing getting swapped between states, just becasue we are computing values.
    activate(inputs, think_t) - the main loop that occurs for every input to produce output
        for think_i in think_t:
            if think_i == 0:
                send input signal
            activate_nodes
            activate_modules
            activate_output
                if output threshold exceeded
                    return output
        return output (regardless)
"""

"""
Main Data structures:
    1d or 2d doesn't really make opt. difference
Nodes: nxn size matrix, type Node
    Each Node:
    x size tuple of indices
    x size tuple of weights
Values: nxn size matrix
biases: nxn size matrix


"""

class HexNetwork:

    def __init__(self, n):
        # For now, not extending past cartpole.
        self.grid = np.zeros((n,n), dtype=Node)
        for i in range(n):
            for j in range(n):
                self.grid[i,j] = Node()
        self.values = np.zeros((n,n), dtype=np.float64)
        self.biases = np.zeros((n,n), dtype=np.float64)

        self.core = []  # keeps track of non-special nodes for propagation
        self.renderer = NetworkRenderer(n)

        # Init input and output modules
        self.inputs = Inputs((0, 0), 5)
        self.outputs = Outputs((n - 4, n - 4), 2, threshold=1)

        # Init base memory bank
        self.memory = [
            MemoryNode((0, 12), threshold=0),
            MemoryNode((0, 14), threshold=0),
            MemoryNode((1, 12), threshold=0.5),
            MemoryNode((1, 14), threshold=0.5),
            MemoryNode((2, 12), threshold=1),
            MemoryNode((2, 14), threshold=1),
            MemoryNode((3, 12), threshold=2),
            MemoryNode((3, 14), threshold=2),
        ]
        # Init base modules
        self.modules = [
            NodeModule((0, 4), threshold=1),
            EdgeModule((4, 0), threshold=1),
            MemoryModule((0, 8), threshold=1),
            MetaModule((12, 0), threshold=2),
        ]
        # Add respective module nodes to grid
        self.add_module(self.inputs)
        self.add_module(self.outputs)
        for memory_node in self.memory:
            self.add_module(memory_node)
        for module in self.modules:
            self.add_module(module)

        # RNG Initialize a core
        rng_hex_core(self.grid, self.biases, self.core)

        # RNG Connect the core to the rest of our network.
        rng_hex_connect_core(self.grid, self.core, self.inputs, self.outputs, self.memory, self.modules)

        self.render()

    # REMEMBER that the only nodes with outputs are normal nodes, and we don't have to handle
    # the cases where there are modules here.
    def add_module(self, module):
        """
        Traverse the given module's nodes, adding ModuleNode's to the grid
            as placeholders where indicated.

        :param module: Module to add
        :return: None, grid is modified in place.
        """
        for node in module.nodes:
            self.grid[node] = ModuleNode()

    # TODO separate into propagate and aggregate again if we only call this in 2 different ways
    def propagate(self, node, bias=True, activation=True):
        # Compute WX + b for a given node
        # each node has different numbers of inputs so we have to compute each separately
        # Each node also may differ in which functions are activated, so this has varying functionality,
        # and can do WX, WX + b, activ(WX), or activ(WX + b) depending on usecase.

        # W = vector of node input weights
        # X = vector of node inputs
        # WX = dotted together, (n,1) x (1,n) -> 1
        idxs = self.grid[node].in_edges
        w = self.grid[node].in_weights
        x = self.values[idxs[:, 0], idxs[:, 1]]
        z = np.dot(w,x)
        if bias:
            z += self.biases[node]
        if activation:
            z = sigmoid(z)
        return z

    def activate(self, input_values, think_t):
        """
        activate(inputs, think_t) - the main loop that occurs for every input to produce output
            for think_i in think_t:
                if think_i == 0:
                    send input signal
                activate_nodes
                activate_modules ( including memory nodes)
                activate_output
                    if output threshold exceeded
                        return output
            return output (regardless)
        :param input_values:
        :param think_t: Timesteps allowed for thinking loop.
            Must be AT LEAST 1 or this will not be able to produce any outputs.
        :return: outputs, regardless of if obtained via threshold or think_t reached.
        """
        prev_len = len(self.core) + len(self.memory) + len(self.modules)
        for think_i in tqdm(range(think_t), disable=True):

            ### INPUT NODES ###
            # first step, get input node values into curr state
            # note: removed next[i]=v since it seemed useless
            if think_i == 0:
                for input_idx, input_value in zip(self.inputs, input_values):
                    self.values[input_idx] = input_value
                self.render()

            ### CORE NODES ###
            # remember that this is ONE state transfer.
            # We iterate through every NORMAL (aka core) node, and propagate its signal forward.
            # then afterwards we handle modules, and check outputs
            # activate_nodes and propagate into next state
            for node in self.core:
                self.values[node] = self.propagate(node)

            ### MEMORY NODES ###
            # activate_memory_nodes
            for memory_node in self.memory:
                # Only overwrite the value if threshold is exceeded.
                # This handles the majority of the functions for memory, since this is where
                # their special behavior comes into play.

                ## THRESHOLD ##
                # computed like normal modules, where we don't go past aggregation.
                threshold_node_input = self.propagate(memory_node[0], bias=False, activation=False)

                # We now employ the resistance-to-overwrites that memory has, s.t. it will only
                # update it's storage/output if the total threshold node input exceeds its initialized
                # threshold value.
                if threshold_node_input > memory_node.threshold:
                    ## STORAGE OUTPUT ##
                    # Threshold exceeded; update storage value.
                    # TODO: decide if we will allow the storage node to have normal biases and activations in future.
                    self.values[memory_node[1]] = self.propagate(memory_node[1], bias=False, activation=False)

                # If threshold not exceeded, its output will remain what it was last set to.
                # Thus retaining its storage.

            ### MODULE NODES ###
            # activate_modules
            for module in self.modules:
                for node in module:
                    # Compute full inputs for each module node
                    # Recall their outputs will never be set,
                    # And they use a different node class.
                    self.values[node] = self.propagate(node, bias=False, activation=False)

                # Now that all module nodes have their full inputs computed and stored,
                # We see if the total inputs for this module can produce a valid activation.
                # Logic for thresholds, addresses, etc. is left up to module.
                if module.is_valid_activation(self.grid, self.values, self.core, self.inputs, self.outputs, self.memory, self.modules):
                    # If so, we do it and update the grid.
                    module.activate(self.grid, self.values, self.biases, self.core, self.inputs, self.outputs, self.memory, self.modules)

            if len(self.core) + len(self.memory) + len(self.modules) != prev_len:
                #print("\nNetwork Change Detected, Rendering...")
                self.render()
                prev_len = len(self.core) + len(self.memory) + len(self.modules)

            ### OUTPUT NODES ###
            # Compute all outputs, again without activations and biases. Will check threshold
            # node and if threshold exceeded, will return output and end thinking loop.
            # activate_output
            ## OUTPUT ##
            for node in self.outputs[:-1]:
                self.values[node] = self.propagate(node, bias=False, activation=False)

            ## THRESHOLD ##
            if self.propagate(self.outputs[-1], bias=False, activation=False) > self.outputs.threshold:
                # Output threshold exceeded, return output and end thinking loop.
                #print(f"\nNetwork Thresholded Outputs: {[self.net[node].output for node in self.outputs[:-1]]}")
                return [self.values[node] for node in self.outputs[:-1]]

        # Thinking loop end, take what has been output so far.
        return [self.values[node] for node in self.outputs[:-1]]

    def render(self):
        #self.renderer.render_hex_network(self)
        pass


# time to improve rendering so we can check this bad boy out
if __name__ == "__main__":
    while True:
        hexnet = HexNetwork(16)
        hexnet.render()
