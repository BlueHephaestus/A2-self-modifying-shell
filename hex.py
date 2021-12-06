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
from neat.graphs import required_for_output


class HexNetwork(object):

    def __init__(self, inputs, outputs, mods, node_evals):
        #TODO: add weight initialization
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.mod_nodes = mods
        self.node_evals = node_evals


        self.values = [{}, {}]
        for v in self.values:
            for k in list(inputs) + list(outputs) + list(mods):
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links in self.node_evals:
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

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, aggregation, bias, response, links in self.node_evals:
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
net = neat.nn.FeedForwardNetwork.create(c, config)
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

