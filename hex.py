"""
Hex - is a weird RNN. This is because for starters i'm trying to maximize greppability
    greppable - to be easy to use, diagnose, understand, work with, handle, etc.

This means that there is a highly limited size of it, since there can only be 256 neurons total in a given Hex.
    it can of course be easily extended i'm just doing it this way so i can view the entire network easily,
    and because rather than extending the range of each dimension i'll probably add dimensions instead. not sure.
"""
"""
OI FUTURE SELF

i'm going to leave this here, we are modifying the network's non-neat components, since it offers a simple
    and really extendable interface for us to build the hexnet's core out of - taking the inputs and thinking 
    until produced outputs happen - we can also specify the selfmod functions here. 
    
SO THE CREATE() FUNCTION WILL BE AVOIDED FOR NOW


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

    def is_valid_activation(self):
        """
        If the module, at this timestep, has met all criteria for a valid activation.

        These are usually:
            if threshold value > threshold
            if address and required node(s) are all in a valid empty space
            if options are valid and in range.
        """
        pass

    def activate(self):
        """
        Assuming is_valid_activation == True, perform this module's function.
            This may be adding, editing, or deleting given on the inputs it receives.
        """
        pass


# could use 1d -> 2d to do this more elegantly but it'd be just as verbose and harder to read
class NodeModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i+i,self.j+j) for i in range(2) for j in range(4)]
        self.nodes.extend([(self.i+2, self.j), (self.i+2, self.j+1)])

class EdgeModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i+i,self.j+j) for i in range(4) for j in range(4)]
        self.nodes.extend([(self.i+4, self.j), (self.i+4, self.j+1)])

class MemoryModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i+i,self.j+j) for i in range(2) for j in range(4)]
        self.nodes.extend([(self.i+2, self.j), (self.i+2, self.j+1)])

class MetaModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i+i,self.j+j) for i in range(3) for j in range(4)]

# TODO set activation and response to suitable defaults since we aren't adding modifiers for these yet
class Node(object):
    # inherently located at a cell in the grid
    def __init__(self):
        # Determines if this cell is empty or if this is an active node.
        self.exists = False

        self.activation = None # type - function
        self.bias = None
        self.response = None# aka output function: type - function
        self.edges = [] # list of form (idx, weight) for source node and weight from it

class ModuleNode(Node):
    def __init__(self):
        """
        Placeholder node with varying purposes, for use with a module at this location.
        """
        Node.__init__(self)
        self.exists = True

class MemoryNode(Module, Node):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)
        Node.__init__(self)

        # Nodes used in full grid (locations, in this case only a pair)
        self.nodes = [(self.i, self.j), (self.i, self.j+1)]

# TODO we should probably make this class subscriptable
class Grid(object):
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n), dtype=Node)
        for i in range(self.n):
            for j in range(self.n):
                self.grid[i,j] = Node()

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
    btw time_to_think can be an attribute or just a value passed to activate
        I think having it be passed to activate is much better.
    Has functions:
        init(size_of_grid)
            creates initial modules
            creates initial rng core
            creates initial rng weights
            
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

        # Init input and output modules
        self.inputs = Inputs((0,0), 5)
        self.outputs = Outputs((grid_n-4,grid_n-4), 2, threshold=1)

        # Init base memory bank
        self.memory_bank = [
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
        for mem_node in self.memory_bank:
            self.net[self.state].add_module(mem_node)
        for module in self.modules:
            self.net[self.state].add_module(module)

        """
        So, RNG init.
        Lotta RNG.
        
        We RNG the core, with Node objects.
        Then, with the core nodes, we RNG connections to all other nodes in hex.
        These connections are RNG'd after the core for both simplicity and modularity.
        We call them secondary edges.
        We generate a large list of all valid secondary edges:
            input -> core
            core -> output
            core -> modules
            core <-> memory nodes
        And then select a random number of them to become secondary edges.
        """

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
        # NOTE ignoring initialization of state 'next' since it gets overwritten by curr propagation

        # initialize initial connections and random weights
        # inputs -> mods, inputs -> outputs. two separate copies sent to mods and outputs
        for input_i in self.input_idxs:
            for mod_i in self.mod_idxs:
                w = np.random.normal()
                # dst has marked down that src is connected to it with weight w
                self.net[self.state][mod_i].connections.append((input_i, w))
            for output_i in self.output_idxs:
                w = np.random.normal()
                # dst has marked down that src is connected to it with weight w
                self.net[self.state][output_i].connections.append((output_i, w))


            # node is just the key - the INDEX, remember
            for node, _activation, _aggregation, _bias, _response, links in self.connections:
                self.net[self.state][node] = 0.0
                for i, w in links:
                    # changing this to be = w, not sure why it started as = 0.0
                    self.net[self.state][i] = w


    # REMEMBER that the only nodes with outputs are normal nodes, and we don't have to handle
    # the cases where there are modules here.

    # REMEMBER that connections is what we were calling node evals. But I like our new form better.
    def activate(self, inputs, think_t):
        """
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
        :param inputs: Input state of the simulation
        :param think_t: Timesteps allowed for thinking loop.
        :return: outputs, regardless of if obtained via threshold or think_t reached.
        """
        for think_i in range(think_t):
            # swap grid objects
            curr = self.net[self.state]
            next = self.net[1 - self.state]
            self.state = 1 - self.state

            # first step, get input node values into curr state
            # note: removed next[i]=v since it seemed useless
            if think_i == 0:
                for i, v in zip(self.input_nodes, inputs):
                    curr[i] = v

            # activate_nodes and propagate into next state
            for (node, activation, aggregation, bias, response), links in self.connections:
                node_inputs = [curr[i] * w for i, w in links]
                s = aggregation(node_inputs)
                next[node] = activation(bias + response * s)

            # activate_modules
            for module in self.modules:
                if module.is_valid_activation():
                    module.activate()

            # activate_output
            # if output threshold exceeded
            # return output

            #return [next[i] for i in self.output_nodes]

hex = HexNetwork(16)
g = hex.net[hex.state].grid
for i in range(16):
    print(i, g[i])

"""
CODE FROM TEST-FEEDFORWARD THAT RUNS THE NETWORK
    WE NEED TO MOD THIS FOR OUR CARTPOLE VERSION
    AND MOD THE HEX CLASS FOR THIS CALLING FUNCTION
"""
"""
#net = neat.nn.FeedForwardNetwork.create(c, config)
net = HexNetwork()
sim = CartPole()

# Run the given simulation for up to 120 seconds.
balance_time = 0.0
while sim.t < 120.0:
    inputs = sim.get_scaled_state()
    action = net.activate(inputs)

    # Apply action to the simulated cart-pole
    force = discrete_actuator_force(action)
    sim.step(force)

    # Stop if the network fails to keep the cart within the position or angle limits.
    # The per-run fitness is the number of time steps the network can balance the pole
    # without exceeding these limits.
    if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
        break

    balance_time = sim.t
"""

"""
So how do we do the normal propagation?
Just like normal, and then we apply any modules or actions afterwards.
So we'd compute the overall inputs into every node just like normal
    the difference with moduels is that their nodes don't have outputs,
    and we make sure of that via the backend which controls outputs of the net.
We could specify what type of node each one is inside of it, but this would
    be unnecessary complex.

In reality this network is not far off from a default RNN, except we have steps
    where we take the results of several choice nodes (modules) and perform meta ops
    with them. 

    And of course we have time for cycles before the output is produced.

Question - what do we do with the memory nodes? 
    They can have outputs
    They require two spaces
    They can have inputs
    They will output as normal (if anything is connected to them) at each timestep.
    They will have their value overwritten, if threshold is exceeded at a given timestamp.

    Definitely it's own custom class.
        And at each timestep we are determining if the value gets overwritten, which will be
        an attr of the node.
"""
