"""
Hex - is a weird RNN. This is because for starters i'm trying to maximize greppability
    greppable - to be easy to use, diagnose, understand, work with, handle, etc.

This means that there is a highly limited size of it, since there can only be 256 neurons total in a given Hex.
    it can of course be easily extended i'm just doing it this way so i can view the entire network easily,
    and because rather than extending the range of each dimension i'll probably add dimensions instead. not sure.
"""
from hex.grid import Grid
from hex.modules.edge import EdgeModule
from hex.modules.io import Inputs, Outputs
from hex.modules.memory import MemoryNode, MemoryModule
from hex.modules.meta import MetaModule
from hex.modules.node import NodeModule
from hex.rng import rng_hex_core, rng_hex_connect_core
from hex.rendering import NetworkRenderer
from tqdm import tqdm

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


class HexNetwork:

    def __init__(self, grid_n):
        # For now, not extending past cartpole.

        self.net = [Grid(grid_n), Grid(grid_n)]
        self.state = 0
        self.core = []  # keeps track of non-special nodes for propagation
        self.renderer = NetworkRenderer(grid_n)

        # Init input and output modules
        self.inputs = Inputs((0, 0), 5)
        self.outputs = Outputs((grid_n - 4, grid_n - 4), 2, threshold=1)

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
        self.net[self.state].add_module(self.inputs)
        self.net[self.state].add_module(self.outputs)
        for memory_node in self.memory:
            self.net[self.state].add_module(memory_node)
        for module in self.modules:
            self.net[self.state].add_module(module)

        # RNG Initialize a core
        rng_hex_core(self.net[self.state], self.core)

        # RNG Connect the core to the rest of our network.
        rng_hex_connect_core(self.net[self.state], self.core, self.inputs, self.outputs, self.memory, self.modules)

        self.render()
        # curr, next - only needed for nodes only
        # guess this means we should add those nodes when we add the modules?
        """
        We initialize the nodes to have random connections TO the modules, which means that yea 
            we do need those to have nodes in the grid.
            
        How do we proper handle the copying?
            Well, from the grids perspective it is only made up of nodes.
            These nodes have their properties limited but still only nodes.
            
        So we start on the first state, and can just initialize that one with all our weights,
            and set the second state to be empty since it will be filled when we propagate.
        """

    # REMEMBER that the only nodes with outputs are normal nodes, and we don't have to handle
    # the cases where there are modules here.

    # REMEMBER that connections is what we were calling node evals. But I like our new form better.
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
            # swap grid objects (does nothing on first iteration)
            curr = self.net[self.state]
            #next = self.net[1 - self.state]
            #self.state = 1 - self.state

            ### INPUT NODES ###
            # first step, get input node values into curr state
            # note: removed next[i]=v since it seemed useless
            if think_i == 0:
                for input_idx, input_value in zip(self.inputs, input_values):
                    curr[input_idx].output = input_value
                self.render()

            ### CORE NODES ###
            # remember that this is ONE state transfer.
            # We iterate through every NORMAL (aka core) node, and propagate its signal forward.
            # then afterwards we handle modules, and check outputs
            # activate_nodes and propagate into next state
            for node in self.core:
                node_input = curr[node].get_input(curr)

                # Compute output value for this node.
                # would be node.bias + node.output_weight * agg if we were using that
                curr[node].output = curr[node].activation(curr[node].bias + node_input)

            ### MEMORY NODES ###
            # activate_memory_nodes
            for memory_node in self.memory:
                # Only overwrite the value if threshold is exceeded.
                # This handles the majority of the functions for memory, since this is where
                # their special behavior comes into play.
                threshold_node, storage_node = curr[memory_node[0]], curr[memory_node[1]]

                ## THRESHOLD ##
                # computed like normal modules, where we don't go past aggregation.
                threshold_node_input = threshold_node.get_input(curr)

                # We now employ the resistance-to-overwrites that memory has, s.t. it will only
                # update it's storage/output if the total threshold node input exceeds its initialized
                # threshold value.
                if threshold_node_input > memory_node.threshold:
                    ## STORAGE OUTPUT ##
                    # Threshold exceeded; update storage value.
                    # TODO: decide if we will allow the storage node to have normal biases and activations in future.
                    #print(f"Updating Memory at {memory_node[0]}")
                    storage_node_input = storage_node.get_input(curr)
                    curr[memory_node[1]].output = storage_node_input

                # If threshold not exceeded, its output will remain what it was last set to.
                # Thus retaining its storage.

            ### MODULE NODES ###
            # activate_modules
            for module in self.modules:
                for node in module:
                    # Compute full inputs for each module node
                    # Recall their outputs will never be set,
                    # And they use a different node class.
                    node_input = curr[node].get_input(curr)
                    curr[node].input = node_input

                # Now that all module nodes have their full inputs computed and stored,
                # We see if the total inputs for this module can produce a valid activation.
                # Logic for thresholds, addresses, etc. is left up to module.
                if module.is_valid_activation(curr, self.core, self.inputs, self.outputs, self.memory, self.modules):
                    # If so, we do it and update the grid.
                    module.activate(curr, self.core, self.inputs, self.outputs, self.memory, self.modules)

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
                node_input = curr[node].get_input(curr)
                curr[node].output = node_input

            ## THRESHOLD ##
            threshold_node = curr[self.outputs[-1]]
            threshold_node_input = threshold_node.get_input(curr)
            if threshold_node_input > self.outputs.threshold:
                # Output threshold exceeded, return output and end thinking loop.
                #print(f"\nNetwork Thresholded Outputs: {[curr[node].output for node in self.outputs[:-1]]}")
                return [curr[node].output for node in self.outputs[:-1]]

        # Thinking loop end, take what has been output so far.
        return [curr[node].output for node in self.outputs[:-1]]

    def render(self):
        self.renderer.render_hex_network(self)
        pass


# time to improve rendering so we can check this bad boy out
if __name__ == "__main__":
    while True:
        hexnet = HexNetwork(16)
        hexnet.render()
