class Module:
    """
    Base Abstract class for all modules in our Hex structure.
    Any given new module types must follow this layout.
    """

    def __init__(self, location: (int, int), threshold: int):
        """
        Creates new module, with common attributes.
        Nodes used in full grid is hardcoded for each module.

        :param location: Location of this module in the grid, via (row, col) coordinate.
        :param threshold: Threshold value for firing of this module. integer.
        """
        self.i, self.j = location
        self.location = location
        self.threshold = threshold
        self.nodes = ""  # TO BE OVERRIDDEN

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

    @staticmethod
    def in_bounds(node, grid):
        """
        :param node: coordinate for a node.
        :param grid: grid to check bounds
        :return: Return if the given node is inside our given grid bounds.
        """
        return (0 <= node).all() and (node < grid.shape[0]).all()

    @staticmethod
    def get_address(values, addr_nodes):
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

        :param values: Usual values at this timestep for all addrs.
        :param addr_nodes: Address nodes with values indicating another node on the grid.
            Assumed to be an even number.
        :return: Coordinate of node on grid indicated by addr_nodes, of form (i, j)
            Note: will always be within the grid bounds.
        """
        n = len(addr_nodes) // 2
        addrs = ""
        for node in addr_nodes:
            addr_val = values[node]
            addrs += "0" if addr_val <= 0 else "1"
        return int(addrs[:n], 2), int(addrs[n:], 2)

    def is_valid_activation(self, grid, values, core, inputs, outputs, memory, modules):
        """
        If the module, at this timestep, has met all criteria for a valid activation.

        These are usually:
            if threshold value > threshold
            if address and required node(s) are all in a valid empty space
            if options are valid and in range.

        :param grid: Usual hex grid where all nodes and modules are stored.
        :param grid: Usual grid of values
        :param core: List of core node idxs.
        :param inputs: Module subclass denoting input locations and nodes.
        :param outputs: Module subclass denoting output locations and nodes.
        :param memory: List of MemoryNode objects denoting Hex Memory bank.
        :param modules: List of Module subclasses denoting various Hex Self-Modification Modules.
        :return: True/False depending on if this is a valid activation of the given module.
        """
        pass

    def activate(self, grid, values, biases, core, inputs, outputs, memory, modules):
        """
        Assuming is_valid_activation == True, perform this module's function.
            This may be adding, editing, or deleting given on the inputs it receives.

        :param grid: Usual hex grid where all nodes and modules are stored.
        :param values: Usual grid of values
        :param biases: Usual grid of biases
        :param core: List of core node idxs.
        :param inputs: Module subclass denoting input locations and nodes.
        :param outputs: Module subclass denoting output locations and nodes.
        :param memory: List of MemoryNode objects denoting Hex Memory bank.
        :param modules: List of Module subclasses denoting various Hex Self-Modification Modules.
        :return: None, modifies grid and given objects in place.
        """
        pass
