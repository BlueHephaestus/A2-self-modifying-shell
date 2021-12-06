"""
My own implementation based on this, to be customised to load in existing structures
"""

"""Handles the continuous-time recurrent neural network implementation."""
from __future__ import division

from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems


class CTRNNNodeEval(object):
    def __init__(self, time_constant, activation, aggregation, bias, response, links):
        self.time_constant = time_constant
        self.activation = activation
        self.aggregation = aggregation
        self.bias = bias
        self.response = response
        self.links = links


class CTRNN(object):
    """Sets up the ctrnn network itself."""

    def __init__(self, inputs, outputs, node_evals):
        # pin nodes; constant for the given environment usually
        self.input_nodes = inputs
        self.output_nodes = outputs

        # node evals = the dest. nodes from when a given node fires?
        self.node_evals = node_evals

        # nice way to have the curr/next state setup
        self.values = [{}, {}]

        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            # init of the meat of the graph.
            """
            node_evals:

            [
                node: key for the node itself
                ne: "node eval" or the dest. nodes from this one.
                    .links attribute:
                    [
                        i: index of dest node
                        w: weight of connection to it
                    ]
            ]


            So.
            How do they make the one for the init of this example? 
                oh they use the above class: CTRNNNodeEval().links
            Because then we can OI FUTURE SELF

            modify this class slightly so the constructor can take a pre-existing CTRNN
                THIS MIGHT ALREADY BE HAPPENING IN THE create() FUNCTION, NEED MORE RESEARCH
                probs best for us to do it ourselves though, lest we delve too far into forces we don't understand
            and then map our existing one to a more greppable format.
            """

            for node, ne in iteritems(self.node_evals):
                v[node] = 0.0
                for i, w in ne.links:
                    # not using w, interesting
                    v[i] = 0.0

        self.active = 0
        self.time_seconds = 0.0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0
        self.time_seconds = 0.0

    def set_node_value(self, node_key, value):
        for v in self.values:
            v[node_key] = value

    def get_max_time_step(self):  # pragma: no cover
        # TODO: Compute max time step that is known to be numerically stable for
        # the current network configuration.
        # pylint: disable=no-self-use
        raise NotImplementedError()

    def advance(self, inputs, advance_time, time_step=None):
        """
        Advance the simulation by the given amount of time, assuming that inputs are
        constant at the given values during the simulated time.
        """
        final_time_seconds = self.time_seconds + advance_time

        # Use half of the max allowed time step if none is given.
        if time_step is None:  # pragma: no cover
            time_step = 0.5 * self.get_max_time_step()

        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0} inputs, got {1}".format(len(self.input_nodes), len(inputs)))

        while self.time_seconds < final_time_seconds:
            dt = min(time_step, final_time_seconds - self.time_seconds)

            ivalues = self.values[self.active]
            ovalues = self.values[1 - self.active]
            self.active = 1 - self.active

            for i, v in zip(self.input_nodes, inputs):
                ivalues[i] = v
                ovalues[i] = v

            for node_key, ne in iteritems(self.node_evals):
                node_inputs = [ivalues[i] * w for i, w in ne.links]
                s = ne.aggregation(node_inputs)
                z = ne.activation(ne.bias + ne.response * s)
                ovalues[node_key] += dt / ne.time_constant * (-ovalues[node_key] + z)

            self.time_seconds += dt

        ovalues = self.values[1 - self.active]
        return [ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config, time_constant):
        """ Receives a genome and returns its phenotype (a CTRNN). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in itervalues(genome.connections):
            if not cg.enabled:
                continue

            i, o = cg.key
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = {}
        for node_key, inputs in iteritems(node_inputs):
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
            node_evals[node_key] = CTRNNNodeEval(time_constant,
                                                 activation_function,
                                                 aggregation_function,
                                                 node.bias,
                                                 node.response,
                                                 inputs)

        return CTRNN(genome_config.input_keys, genome_config.output_keys, node_evals)
