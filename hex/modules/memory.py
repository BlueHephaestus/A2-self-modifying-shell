from hex.modules.module import Module
from hex.nodes import Node


class MemoryNode(Module, Node):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)
        Node.__init__(self)

        # Nodes used in full grid (locations, in this case only a pair)
        self.nodes = [(self.i, self.j), (self.i, self.j + 1)]


class MemoryModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i + i, self.j + j) for i in range(2) for j in range(4)]
        self.nodes.extend([(self.i + 2, self.j), (self.i + 2, self.j + 1)])

        self.epsilon = -1  # if value_node is less than this, delete the given memory node.
        self.addr_nodes = self.nodes[:8]
        self.threshold_node = self.nodes[-2]
        self.value_node = self.nodes[-1]

    def is_valid_activation(self, grid, values, core, inputs, outputs, memory, modules):
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
        if values[self.threshold_node] <= self.threshold:
            return False

        self.addr = self.get_address(values, self.addr_nodes)

        # By definition this can't be on the farthest column and have room for full memory node.
        if self.addr[1] == grid.shape[1]-1:
            return False

        # Threshold meaning the new memory node's threshold
        self.threshold_addr = self.addr
        self.storage_addr = (self.addr[0],self.addr[1]+1)

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

    def activate(self, grid, values, biases, core, inputs, outputs, memory, modules):
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
        value = outputs[self.value_node]
        threshold_node = grid[self.threshold_addr]
        storage_node = grid[self.storage_addr]

        # Delete if any node exists, otherwise leave empty.
        if value < self.epsilon and self.memory_node_exists:
            # Remove all edges and references
            # Thresholds can only have inputs, Storage can have inputs and outputs.
            # src (out_edges) <-> node (in edges, out edges) <-> dst (in edges)

            # Remove threshold inputs
            for in_node, _weight in threshold_node.in_edges:
                # Remove references from any incoming nodes' out_edges lists.
                grid[in_node].remove_outgoing(self.threshold_addr)

            # Remove storage inputs
            for in_node, _weight in storage_node.in_edges:
                # Remove references from any incoming nodes' out_edges lists.
                grid[in_node].remove_outgoing(self.storage_addr)

            # Remove storage outputs
            for out_node in storage_node.out_edges:
                # Remove references from any outgoing nodes' in_edges lists.
                grid[out_node].remove_incoming(self.storage_addr)

            # Finally remove the node via full reset, from both grid and memory.
            grid[self.threshold_addr] = Node()
            grid[self.storage_addr] = Node()
            del memory[self.memory_i]

        # Non-delete cases - edit existing or add new if empty
        elif value >= self.epsilon:
            if not self.memory_node_exists:
                # Add
                memory_node = MemoryNode(self.threshold_addr, threshold=value)
                memory.append(memory_node)
                grid.add_module(memory_node)
            else:
                # Edit threshold (it's already in grid and memory)
                memory[self.memory_i].threshold = value
