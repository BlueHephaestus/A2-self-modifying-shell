from hex.nodes import Node, ModuleNode
from hex.modules.module import Module


# could use 1d -> 2d to do this more elegantly but it'd be just as verbose and harder to read
# TODO use 1d instead, that is more general for higher size grids.
# TODO have attribute for the grid size???
class NodeModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i + i, self.j + j) for i in range(2) for j in range(4)]
        self.nodes.extend([(self.i + 2, self.j), (self.i + 2, self.j + 1)])

        self.epsilon = 0.05  # if value_node is less than this away from 0, delete the given node.
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
        if isinstance(node, ModuleNode):
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
        value = grid[self.value_node].input
        node = grid[self.addr]

        # Delete if any node exists, otherwise leave empty.
        if abs(value) < self.epsilon and node.exists:
            # Remove all edges and references
            # TODO: consider changing to a dict structure to save complexity.
            # src (out_edges) <-> node (in edges, out edges) <-> dst (in edges)
            for in_node, _weight in node.in_edges:
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
        elif abs(value) >= self.epsilon:
            if not node.exists:
                # Add
                grid[self.addr] = Node()
                grid[self.addr].exists = True
                core.append(self.addr)
            # Edit bias regardless of if pre-existing or new
            grid[self.addr].bias = value
