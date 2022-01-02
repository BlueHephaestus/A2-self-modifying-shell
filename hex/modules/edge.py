from hex.modules.module import Module


class EdgeModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i + i, self.j + j) for i in range(4) for j in range(4)]
        self.nodes.extend([(self.i + 4, self.j), (self.i + 4, self.j + 1)])

        self.epsilon = 0.05  # if value_node is less than this away from 0, delete the given edge.
        self.src_addr_nodes = self.nodes[:8]
        self.dst_addr_nodes = self.nodes[8:16]
        self.threshold_node = self.nodes[-2]
        self.value_node = self.nodes[-1]

    def is_valid_activation(self, grid, values, core, inputs, outputs, memory, modules):
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
        if values[self.threshold_node] <= self.threshold:
            return False

        self.src_addr = self.get_address(values, self.src_addr_nodes)
        self.dst_addr = self.get_address(values, self.src_addr_nodes)
        src = grid[self.src_addr]
        dst = grid[self.dst_addr]

        # Both must exist
        if not src.exists or not dst.exists:
            return False

        # Must be either core, input, or memory storage (second in each memory node)
        if self.src_addr not in core \
                and self.src_addr not in inputs \
                and self.src_addr not in [memory_node[1] for memory_node in memory]:
            return False

        # Must be anything other than an input node.
        if self.dst_addr in inputs:
            return False

        # Otherwise this is valid, proceed with activation
        return True

    def activate(self, grid, values, biases, core, inputs, outputs, memory, modules):
        """
        Execute activation for EdgeModule.
        This will perform the following based on the value in value_node:
            Value is within `epsilon` of zero: Delete the edge between addresses, if one exists.
                If no edge, we do nothing.

            If Value is within commit range:
                If no edge exists yet, create new one with given value as weight.
                If edge exists, update weight to match given value.
        """
        value = values[self.value_node]
        edge_exists = self.dst_addr in grid[self.src_addr].out_edges

        # Delete if any edge exists, otherwise leave empty.
        if abs(value) < self.epsilon and edge_exists:
            grid[self.src_addr].remove_outgoing(self.dst_addr)
            grid[self.dst_addr].remove_incoming(self.src_addr)

        # Non-delete cases - edit existing or add new if not
        elif abs(value) >= self.epsilon:
            if not edge_exists:
                # Add
                grid[self.src_addr].add_outgoing(self.dst_addr)
                grid[self.dst_addr].add_incoming(self.src_addr, value)
            else:
                # Edit
                grid[self.dst_addr].in_edges.edit_incoming(self.src_addr, value)
