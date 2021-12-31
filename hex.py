"""
Hex - is a weird RNN. This is because for starters i'm trying to maximize greppability
    greppable - to be easy to use, diagnose, understand, work with, handle, etc.

This means that there is a highly limited size of it, since there can only be 256 neurons total in a given Hex.
    it can of course be easily extended i'm just doing it this way so i can view the entire network easily,
    and because rather than extending the range of each dimension i'll probably add dimensions instead. not sure.
"""
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
import numpy as np
from rng import rng_hex_core, rng_hex_connect_core

class Module():
    """
    Base Abstract class for all modules in our Hex structure.
    Any given new module types must follow this layout.
    """
    def __init__(self, location: (int,int), threshold: int):
        """
        Creates new module, with common attributes.
        Nodes used in full grid is hardcoded for each module.

        :param location: Location of this module in the grid, via (row, col) coordinate.
        :param threshold: Threshold value for firing of this module. integer.
        """
        self.i, self.j = location
        self.location = location
        self.threshold = threshold
        self.nodes = ""# TO BE OVERRIDDEN

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, i):
        return self.nodes[i]

    def __iter__(self):
        self.iter_i = 0
        return self

    def __next__(self):
        if self.iter_i < len(self):
            res = self[self.iter_i]
            self.iter_i += 1
            return res
        else:
            raise StopIteration

    def get_address(self, grid, addr_nodes):
        """
        Given a grid and the nodes on it which contain a raw address input for fully indicating
            another node in the grid:

            Get the length of each address (log2 of grid size, or half of len(addr_nodes))
                We use half because it's simpler.
            Convert to binary: <=0 is 0, >0 is 1
            Split into two binary strings
            Convert each to decimal
            Return both.

        e.g. 3,3,-1,2 for a grid of 4x4 -> 1,1,0,1 -> 11, 01 -> Node at (3, 1)

        :param grid: Usual grid at this timestep with addr values.
        :param addr_nodes: Address nodes with values indicating another node on the grid.
            Assumed to be an even number.
        :return: Coordinate of node on grid indicated by addr_nodes, of form (i, j)
            Note: will always be within the grid bounds.
        """
        n = len(addr_nodes)//2
        addrs = ""
        for node in addr_nodes:
            addr_val = grid[node].input
            addrs += "0" if addr_val <= 0 else "1"
        return (int(addrs[:n], 2), int(addrs[n:], 2))

    def is_valid_activation(self, grid, core, inputs, outputs, memory, modules):
        """
        If the module, at this timestep, has met all criteria for a valid activation.

        These are usually:
            if threshold value > threshold
            if address and required node(s) are all in a valid empty space
            if options are valid and in range.

        :param grid: Usual hex grid where all nodes and modules are stored.
        :param core: List of core node idxs.
        :param inputs: Module subclass denoting input locations and nodes.
        :param outputs: Module subclass denoting output locations and nodes.
        :param memory: List of MemoryNode objects denoting Hex Memory bank.
        :param modules: List of Module subclasses denoting various Hex Self-Modification Modules.
        :return: True/False depending on if this is a valid activation of the given module.
        """
        pass

    def activate(self, grid, core, inputs, outputs, memory, modules):
        """
        Assuming is_valid_activation == True, perform this module's function.
            This may be adding, editing, or deleting given on the inputs it receives.

        :param grid: Usual hex grid where all nodes and modules are stored.
        :param core: List of core node idxs.
        :param inputs: Module subclass denoting input locations and nodes.
        :param outputs: Module subclass denoting output locations and nodes.
        :param memory: List of MemoryNode objects denoting Hex Memory bank.
        :param modules: List of Module subclasses denoting various Hex Self-Modification Modules.
        :return: None, modifies grid and given objects in place.
        """
        pass


# could use 1d -> 2d to do this more elegantly but it'd be just as verbose and harder to read
# TODO use 1d instead, that is more general for higher size grids.
# TODO have attribute for the grid size???
class NodeModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i+i,self.j+j) for i in range(2) for j in range(4)]
        self.nodes.extend([(self.i+2, self.j), (self.i+2, self.j+1)])

        self.epsilon = 0.05 # if value_node is less than this away from 0, delete the given node.
        self.addr_nodes = self.nodes[:8]
        self.threshold_node = self.nodes[-2]
        self.value_node = self.nodes[-1]

    def is_valid_activation(self, grid, core, inputs, outputs, memory, modules):
        """
        Determine if this is a valid activation for our Node Modules
        This occurs if all of the following are true:
            Threshold nodes exceeded
            Address evaluates to valid address

        Recall that although many activations may be valid, they may do different things
            depending on the values given.
        """

        # If not exceeded, we don't do anything
        if grid[self.threshold_node].input <= self.threshold:
            return False

        # If address points to a ModuleNode of any type (this includes inputs and outputs)
        # Yes, this means that this module can't overwrite storage nodes - those are meant
        # to only be writable via accessing the memory and exceeding that threshold.
        self.addr = self.get_address(grid, self.addr_nodes)
        node = grid[self.addr]
        if isinstance(node == ModuleNode):
            return False

        # Otherwise this is valid, proceed with activation
        return True


    def activate(self, grid, core, inputs, outputs, memory, modules):
        """
        Execute activation for NodeModule.
        This will perform the following based on the value in value_node:
            Value is within `epsilon` of zero: Delete the node at address, if one exists.
                This will prune any connections to and from this node, as well.
                This is why we keep track of edges going both ways for a given node.
                If empty cell, we do nothing.

            If Value is within commit range:
                Address points to an empty cell: Create a new node at address, w/ given value.
                    No connections.
                Address points to an existing node: Update bias value to match given value.
        """
        value = grid[self.value_node]
        node = grid[self.addr]

        # Delete if any node exists, otherwise leave empty.
        if abs(value) < self.epsilon and node.exists:
            # Remove all edges and references
            # TODO: consider changing to a dict structure to save complexity.
            # src (out_edges) <-> node (in edges, out edges) <-> dst (in edges)
            for in_node,_weight in node.in_edges:
                # Remove references from any incoming nodes' out_edges lists.
                grid[in_node].out_edges.remove(self.addr)

            for out_node in node.out_edges:
                # Remove references from any outgoing nodes' in_edges lists.
                grid[out_node].in_edges = [in_edge for in_edge in grid[out_node].in_edges if in_edge[0] != self.addr]

            # Finally remove the node via full reset, from both grid and core.
            # TODO: consider adding grid.delete method for a given addr to be replaced with new Node() object.
            grid[self.addr] = Node()
            core.remove(self.addr)

        # Non-delete cases - edit existing or add new if empty
        else:
            if not node.exists:
                # Add
                grid[self.addr] = Node()
                core.append(self.addr)
            # Edit bias regardless of if pre-existing or new
            grid[self.addr].bias = value

class EdgeModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i+i,self.j+j) for i in range(4) for j in range(4)]
        self.nodes.extend([(self.i+4, self.j), (self.i+4, self.j+1)])

        self.epsilon = 0.05 # if value_node is less than this away from 0, delete the given edge.
        self.src_addr_nodes = self.nodes[:8]
        self.dst_addr_nodes = self.nodes[8:16]
        self.threshold_node = self.nodes[-2]
        self.value_node = self.nodes[-1]

    def is_valid_activation(self, grid, core, inputs, outputs, memory, modules):
        """
        Determine if this is a valid activation for our Edge Modules
        This occurs if all of the following are true:
            Threshold nodes exceeded
            Addresses evaluate to non-empty cells (can be ModuleNodes, in some cases)
            Indicated edge is valid, meaning BOTH:
                The source node:
                    Is either core, input, or memory storage.
                The destination node:
                    Is either core, output, or a module node (anything other than inputs)

        Recall that although many activations may be valid, they may do different things
            depending on the values given.
        """

        # If not exceeded, we don't do anything
        if grid[self.threshold_node].input <= self.threshold:
            return False

        self.src_addr = self.get_address(grid, self.src_addr_nodes)
        self.dst_addr = self.get_address(grid, self.src_addr_nodes)
        src = grid[self.src_addr]
        dst = grid[self.dst_addr]

        # Both must exist
        if not src.exists or not dst.exists:
            return False

        # Must be either core, input, or memory storage (second in each memory node)
        if self.src_addr not in core\
            and self.src_addr not in inputs\
            and self.src_addr not in [memory_node[1] for memory_node in memory]:
            return False

        # Must be anything other than an input node.
        if self.dst_addr in inputs:
            return False

        # Otherwise this is valid, proceed with activation
        return True

    def activate(self, grid, core, inputs, outputs, memory, modules):
        """
        Execute activation for EdgeModule.
        This will perform the following based on the value in value_node:
            Value is within `epsilon` of zero: Delete the edge between addresses, if one exists.
                If no edge, we do nothing.

            If Value is within commit range:
                If no edge exists yet, create new one with given value as weight.
                If edge exists, update weight to match given value.
        """
        value = grid[self.value_node]
        edge_exists = self.dst_addr in grid[self.src_addr].out_edges

        # Delete if any edge exists, otherwise leave empty.
        if abs(value) < self.epsilon and edge_exists:
            grid[self.src_addr].out_edges.remove(self.dst_addr)
            grid[self.dst_addr].in_edges = [in_edge for in_edge in grid[self.dst_addr].in_edges if in_edge[0] != self.src_addr]

        # Non-delete cases - edit existing or add new if not
        else:
            if not edge_exists:
                # Add
                grid[self.src_addr].out_edges.append(self.dst_addr)
                grid[self.dst_addr].in_edges.append((self.src_addr, value))
            else:
                # Edit (via remove and re-add because tuples)
                grid[self.dst_addr].in_edges = [in_edge for in_edge in grid[self.dst_addr].in_edges if
                                                in_edge[0] != self.src_addr]
                grid[self.dst_addr].in_edges.append((self.src_addr, value))


class MemoryModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i+i,self.j+j) for i in range(2) for j in range(4)]
        self.nodes.extend([(self.i+2, self.j), (self.i+2, self.j+1)])

        self.epsilon = -1 # if value_node is less than this, delete the given memory node.
        self.addr_nodes = self.nodes[:8]
        self.threshold_node = self.nodes[-2]
        self.value_node = self.nodes[-1]

    def is_valid_activation(self, grid, core, inputs, outputs, memory, modules):
        """
        Determine if this is a valid activation for our Memory Modules
        This occurs if all of the following are true:
            Threshold nodes exceeded
            Address evaluates to valid address, with either
                existing memory node
                or enough room for new memory node

        Recall that although many activations may be valid, they may do different things
            depending on the values given.
        """

        # If not exceeded, we don't do anything
        if grid[self.threshold_node].input <= self.threshold:
            return False

        self.addr = self.get_address(grid, self.addr_nodes)

        # By definition this can't be on the farthest column and have room for full memory node.
        if self.addr[1] >= grid.shape[1]:
            return False

        # Threshold meaning the new memory node's threshold
        self.threshold_addr = self.addr
        self.storage_addr = self.addr + (0,1)
        threshold_node = grid[self.threshold_addr]
        storage_node = grid[self.storage_addr]

        # Determine if we already have a memory node here.
        self.memory_node_exists = False
        self.memory_i = -1
        for memory_i, memory_node in enumerate(memory):
            if memory_node[0] == self.threshold_addr:
                self.memory_node_exists = True
                self.memory_i = memory_i
                break

        # If we don't, determine if we have free cells for a new one.
        if not self.memory_node_exists:
            if threshold_node.exists or storage_node.exists:
                return False

        # Otherwise a memory node either already exists or there is room for one,
        # proceed with activation.
        return True

    def activate(self, grid, core, inputs, outputs, memory, modules):
        """
        Execute activation for MemoryModule.
        This will perform the following based on the value in value_node:
            Value is less than `epsilon`: Delete the node at address, if one exists.
                This will prune any connections to and from this node, as well.
                This is why we keep track of edges going both ways for a given node.
                If empty cell, we do nothing.

            If Value is within commit range (greater than `epsilon`):
                Address points to an empty cell: Create a new node at address, w/ given value
                    as the new threshold.
                    No connections, no values in storage.
                Address points to an existing node: Update threshold value to match given value.
        """
        value = grid[self.value_node]
        threshold_node = grid[self.threshold_addr]
        storage_node = grid[self.storage_addr]

        # Delete if any node exists, otherwise leave empty.
        if value < self.epsilon and self.memory_node_exists:
            # Remove all edges and references
            # Thresholds can only have inputs, Storage can have inputs and outputs.
            # src (out_edges) <-> node (in edges, out edges) <-> dst (in edges)

            # Remove threshold inputs
            for in_node,_weight in threshold_node.in_edges:
                # Remove references from any incoming nodes' out_edges lists.
                grid[in_node].out_edges.remove(self.threshold_addr)

            # Remove storage inputs
            for in_node,_weight in storage_node.in_edges:
                # Remove references from any incoming nodes' out_edges lists.
                grid[in_node].out_edges.remove(self.storage_addr)

            # Remove storage outputs
            for out_node in storage_node.out_edges:
                # Remove references from any outgoing nodes' in_edges lists.
                grid[out_node].in_edges = [in_edge for in_edge in grid[out_node].in_edges if in_edge[0] != self.storage_addr]

            # Finally remove the node via full reset, from both grid and memory.
            grid[self.threshold_addr] = Node()
            grid[self.storage_addr] = Node()
            del memory[self.memory_i]

        # Non-delete cases - edit existing or add new if empty
        else:
            if not self.memory_node_exists:
                # Add
                memory_node = MemoryNode(self.threshold_addr, threshold=value)
                memory.append(memory_node)
                grid.add_module(memory_node)
            else:
                # Edit threshold (it's already in grid and memory)
                memory[self.memory_i].threshold = value


class MetaModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i+i,self.j+j) for i in range(3) for j in range(4)]

        self.epsilon = -1 # if value_node is less than this, delete the given module.
        self.addr_nodes = self.nodes[:8]
        self.threshold_node = self.nodes[-4]
        self.module_type_nodes = self.nodes[-3:-1]
        self.value_node = self.nodes[-1]

        # Index == key, maps the binary num indicated by module type nodes to the type of
        # module this will be editing. Same logic for numerical parsing as addresses.
        # e.g. node vals of [-3.43, 1.21] -> [0, 1] -> 01 -> 1 -> map[1] -> EdgeModule.
        self.module_type_mapping = [NodeModule, EdgeModule, MemoryModule, MetaModule]

    @staticmethod
    def get_module_type(grid, type_nodes, type_mapping):

        """
        Given a grid and the nodes on it which contain a raw module type input for
            fully indicating one of four modules for this metamodule to reference,

            Convert to binary: <=0 is 0, >0 is 1
            Convert to decimal
            Use as index mapping to get module type
        # e.g. node vals of [-3.43, 1.21] -> [0, 1] -> 01 -> 1 -> map[1] -> EdgeModule.

        :param grid: Usual grid at this timestep with addr values.
        :param type_nodes: Type nodes with values indicating which module to use.
        :param type_mapping: Mapping of value -> Module subclass
        :return: one of the given modules in the mapping, a subclass of Module.
        """
        module_type = ""
        for node in type_nodes:
            type_val = grid[node].input
            module_type += "0" if type_val <= 0 else "1"
        return type_mapping[int(module_type, 2)]

    def is_valid_activation(self, grid, core, inputs, outputs, memory, modules):
        """
        Determine if this is a valid activation for our Meta Module
        This occurs if all of the following are true:
            Threshold nodes exceeded
            Address evaluates to valid address, with either
                existing module or
                enough room for new module
                    New module determined by type node input.

        Recall that although many activations may be valid, they may do different things
            depending on the values given.

        Also note that this is the most powerful module, and subsequently has very stringent
            conditions for working - it's meant to be "with great power comes great responsibility"
        """

        # If not exceeded, we don't do anything
        if grid[self.threshold_node].input <= self.threshold:
            return False

        self.addr = self.get_address(grid, self.addr_nodes)
        self.module_type = self.get_module_type(grid, self.module_type_nodes, self.module_type_mapping)

        ###
        # We always care about both address and module type.
        # If we are adding, we need enough room for the given module type
        # If we are editing, it has to match the given module type
        # If we are deleting, it has to match the given module type

        # Determine if we already have a matching module here
        self.module_exists = False
        self.module_i = -1
        for module_i, module in enumerate(modules):
            if module[0] == self.addr:
                # If it matches we're good
                if isinstance(module, self.module_type):
                    self.module_exists = True
                    self.module_i = module_i
                    break
                # If it doesn't then this isn't valid and we do nothing
                else:
                    return False

        # If we don't, determine if we have enough room for a new one of our given type
        # Via if any of the needed nodes are already in use.
        # I'm really glad I made these all subclasses of Module now, and i'm really glad
        # that i have a separate method to actually add an instantiated module.
        if not self.module_exists:
            module = self.module_type(self.addr, -1) # get node positions via dummy module
            for node in module.nodes:
                if grid[node].exists:
                    return False


        # Otherwise either a matching module already exists or we have room for it,
        # proceed with activation.
        return True

    def activate(self, grid, core, inputs, outputs, memory, modules):
        """
        Execute activation for MetaModule.
        This will perform the following based on the value in value_node:
            Value is less than `epsilon`: Delete the module at address, if one exists.
                This will prune any connections to and from ALL of this module's nodes, as well.
                This is why we keep track of edges going both ways for a given node.
                If empty area, we do nothing.

            If Value is within commit range (greater than `epsilon`):
                (we already know it must match the module type)
                Address points to an empty area: Create a new module at address, w/ given value
                    as the new threshold.
                    No connections, nothing else.
                Address points to an existing module: Update threshold value to match given value.
        """
        value = grid[self.value_node]

        # Delete if any module exists, otherwise leave empty.
        if value < self.epsilon and self.module_exists:
            # Remove all edges and references to all nodes in module.
            # Modules can only have inputs, for all their nodes.
            # src (out_edges) -> node (in edges)
            module = modules[self.module_i]

            # Iteratively remove each node.
            for node in module.nodes:
                # Remove all outgoing connections to these nodes.
                for in_node,_weight in grid[node].in_edges:
                    # Remove references from any incoming nodes' out_edges lists.
                    grid[in_node].out_edges.remove(node)

                # Remove the node
                grid[node] = Node()

            # Finally remove the module.
            del modules[self.module_i]

        # Non-delete cases - edit existing or add new if empty
        else:
            if not self.module_exists:
                # Add
                module = self.module_type(self.addr, threshold=value)
                modules.append(module)
                grid.add_module(module)
            else:
                # Edit threshold (it's already in grid and modules)
                modules[self.module_i].threshold = value

class Node(object):
    # inherently located at a cell in the grid
    def __init__(self):
        # Determines if this cell is empty or if this is an active node.
        self.exists = False

        # Output value for this node.
        self.output = None

        self.activation = self.sigmoid
        self.aggregation = sum
        self.bias = None
        self.output_weight = None # not using this for now

        self.in_edges = [] # list of form (idx, weight) for source node and weight from it
        self.out_edges = [] # list of idx for dst node - used for reference when removing nodes.

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
        #node_inputs = [grid[i].output * w for i, w in self.edges if grid[i].output is not None]

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

class MemoryNode(Module, Node):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)
        Node.__init__(self)

        # Nodes used in full grid (locations, in this case only a pair)
        self.nodes = [(self.i, self.j), (self.i, self.j+1)]

class Grid(object):
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n), dtype=Node)
        for i in range(self.n):
            for j in range(self.n):
                self.grid[i,j] = Node()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        #return self.grid[i]
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
# Keep as module subclass for the required interface
class Inputs(Module):
    def __init__(self, location, input_n):
        # Keeps in a 4x4 quadrant from initial location.
        Module.__init__(self, location, None)
        self.i, self.j = location
        self.nodes = [(self.i+i//4,self.i+i%4) for i in range(input_n)]

# Keep as module subclass for the required interface
class Outputs(Module):
    def __init__(self, location, output_n, threshold):
        # Keeps in a 4x4 quadrant from initial location.
        # Threshold node not included in number of outputs, is the last output node.
        Module.__init__(self, location, threshold)
        self.i, self.j = location
        self.nodes = [(self.i+i//4,self.i+i%4) for i in range(output_n+1)]

class HexNetwork(object):

    def __init__(self, grid_n):
        # For now, not extending past cartpole.

        self.net = [Grid(grid_n), Grid(grid_n)]
        self.state = 0
        self.core = [] # keeps track of non-special nodes for propagation
        #self.renderer = NetworkRenderer(grid_n)

        # Init input and output modules
        self.inputs = Inputs((0,0), 5)
        self.outputs = Outputs((grid_n-4,grid_n-4), 2, threshold=1)

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
            NodeModule((0,4), threshold=1),
            EdgeModule((4,0), threshold=1),
            MemoryModule((0,8), threshold=1),
            MetaModule((12,0), threshold=2),
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
        :param inputs: Input state of the simulation
        :param think_t: Timesteps allowed for thinking loop.
        :return: outputs, regardless of if obtained via threshold or think_t reached.
        """
        for think_i in range(think_t):
            # swap grid objects (does nothing on first iteration)
            curr = self.net[self.state]
            next = self.net[1 - self.state]
            self.state = 1 - self.state

            ### INPUT NODES ###
            # first step, get input node values into curr state
            # note: removed next[i]=v since it seemed useless
            if think_i == 0:
                for input_idx, input_value in zip(self.inputs, input_values):
                    curr[input_idx].output = input_value

            ### CORE NODES ###
            # remember that this is ONE state transfer.
            # We iterate through every NORMAL (aka core) node, and propagate its signal forward.
            # then afterwards we handle modules, and check outputs
            # activate_nodes and propagate into next state
            for node in self.core:
                node_input = curr[node].get_input(curr)

                # Compute output value for this node.
                # would be node.bias + node.output_weight * agg if we were using that
                next[node].output = node.activation(node.bias + node_input)

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
                    # TODO: decide if storage values should be initialized to something other than None
                    storage_node_input = storage_node.get_input(curr)
                    next[storage_node].output = storage_node_input

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
                    next[node].input = node_input

                # Now that all module nodes have their full inputs computed and stored,
                # We see if the total inputs for this module can produce a valid activation.
                # Logic for thresholds, addresses, etc. is left up to module.
                if module.is_valid_activation(next, self.core, self.inputs, self.outputs, self.memory, self.modules):
                    # If so, we do it and update the grid.
                    module.activate(next, self.core, self.inputs, self.outputs, self.memory, self.modules)

            ### OUTPUT NODES ###
            # Compute all outputs, again without activations and biases. Will check threshold
            # node and if threshold exceeded, will return output and end thinking loop.
            # activate_output
            ## OUTPUT ##
            for node in self.outputs[:-1]:
                node_input = curr[node].get_input(curr)
                next[node].output = node_input

            ## THRESHOLD ##
            threshold_node = curr[self.outputs[-1]]
            threshold_node_input = threshold_node.get_input(curr)
            if threshold_node_input > self.outputs.threshold:
                # Output threshold exceeded, return output and end thinking loop.
                return [next[node].output for node in self.outputs[:-1]]

        # Thinking loop end, take what has been output so far.
        return [next[node].output for node in self.outputs[:-1]]

    def render(self):
        self.renderer.render_hex_network()

# time to improve rendering so we can check this bad boy out
if __name__ == "__main__":
    hex = HexNetwork(16)
    #hex.render()