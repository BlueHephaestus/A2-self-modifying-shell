"""
Hex - is a weird RNN. This is because for starters i'm trying to maximize greppability
    greppable - to be easy to use, diagnose, understand, work with, handle, etc.

So i'm going to make it so I can view the entire network as a 16x16 grid, with hexadecimal coordinates,
and special nodes in the top-left corner for input, output, and selfmod nodes.
    net[0,1:] is outputs
    net[1:8,0] is selfmods
    net[(0,0),(1,1),...] is inputs

    i.e. rows, columns, and diagonals from top left.

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



class NodeModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.i, self.j = location
        self.nodes = [(self.i+i,self.j+j) for i in range(2) for j in range(4)]
        self.nodes.extend([(self.i+2, self.j), (self.i+2, self.j+1)])

class EdgeModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.i, self.j = location
        self.nodes = [(self.i+i,self.j+j) for i in range(4) for j in range(4)]
        self.nodes.extend([(self.i+4, self.j), (self.i+4, self.j+1)])

class MemoryModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.i, self.j = location
        self.nodes = [(self.i+i,self.j+j) for i in range(2) for j in range(4)]
        self.nodes.extend([(self.i+2, self.j), (self.i+2, self.j+1)])

class MetaModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.i, self.j = location
        self.nodes = [(self.i+i,self.j+j) for i in range(3) for j in range(4)]

class Grid(object):
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n), dtype=Node)
        for i in self.n:
            for j in self.n:
                self.grid[i,j] = Node()


class HexGrid(object):
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size), dtype=HexNode)
        for i in self.size:
            for j in self.size:
                self.grid[i,j] = HexNode()

class HexNode(object):
    # inherently located at a node in the hexgrid
    def __init__(self):
        self.activation = None
        self.bias = None
        self.response = None#aka output function
        self.connections = []
        self.enabled = False

    def init_weights(self):
        pass



class HexNetwork(object):

    def __init__(self, input_n, output_n, connections):


        #TODO: add weight initialization
        #self.input_nodes = list(range(input_n))
        #self.output_nodes = list(range(output_n))
        #self.mod_nodes = list(range(MOD_NODES))

        # indices
        self.input_idxs = np.diag_indices(input_n)
        self.mod_idxs = (range(1,MOD_NODES+1), 0)
        self.output_idxs = (0, range(1,output_n+1))

        # where we would put our initial connections in the network
        self.connections = connections

        # curr, next
        self.net = [HexGrid(NET_SIZE), HexGrid(NET_SIZE)]

        # given we have hexnet, we no longer have node keys as indices infinitely
        # they are just their location in the hex grid.
        # see this is why i love it
        # simple matrices, bouis

        """
        # realized this is unnecessary since we literally initialize the entire thing to zeros
        for state in self.net:
            self.net[state][self.input_i] = 0.0
            self.net[state][self.mod_i] = 0.0
            self.net[state][self.output_i] = 0.0
        """

        # initialize initial connections and random weights
        # inputs -> mods, inputs -> outputs. two separate copies sent to mods and outputs
        for state in self.net:
            for input_i in self.input_idxs:
                for mod_i in self.mod_idxs:
                    w = np.random.normal()
                    # dst has marked down that src is connected to it with weight w
                    self.net[state][mod_i].connections.append((input_i, w))
                for output_i in self.output_idxs:
                    w = np.random.normal()
                    # dst has marked down that src is connected to it with weight w
                    self.net[state][output_i].connections.append((output_i, w))

        """
        OI FUTURE SELF
        THIS IS WHERE WE LEFT OFF, BECAUSE IT'S 1214 AND I NEED TO GO TO BED
        REALLY REALLY THIS IS THE ONE WHERE WE LEFT OFF
        """

    for v in self.values:

            # node is just the key, remember
            for node, _activation, _aggregation, _bias, _response, links in self.connections:
                v[node] = 0.0
                for i, w in links:
                    # changing this to be = w, not sure why it started as = 0.0
                    v[i] = w

        # state flag
        self.active = 0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    # TODO need to add in timesteps for when it doesn't activate the potential of the outputs, and
    # therefore doesn't have new inputs and is still thinking until it produces an output.
    # meaning that this function will run UNTIL output
    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        # while outputs activation threshold not met and wait_period not exceeded: advance

        # misleading; not input / output, but state transition implementation
        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        # first step, get input node values
        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, aggregation, bias, response, links in self.connections:
            node_inputs = [ivalues[i] * w for i, w in links]
            s = aggregation(node_inputs)
            ovalues[node] = activation(bias + response * s)

        return [ovalues[i] for i in self.output_nodes]

    # TODO AVOIDING THIS FOR NOW; WE WILL IMPLEMENT NEAT LATER - FOCUSING ON BASE DESIGN OF NETWORK FOR NOW
    @staticmethod
    def create(genome, config):
        pass
        """
        # Receives a genome and returns its phenotype (a RecurrentNetwork). 
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in genome.connections.values():
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = []
        for node_key, inputs in node_inputs.items():
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals.append((node_key, activation_function, aggregation_function, node.bias, node.response, inputs))

        return HexNetwork(genome_config.input_keys, genome_config.output_keys, node_evals)
        """

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

