from hex.modules.edge import EdgeModule
from hex.modules.memory import MemoryModule
from hex.modules.module import Module
from hex.modules.node import NodeModule
from hex.nodes import Node


class MetaModule(Module):
    def __init__(self, location, threshold):
        Module.__init__(self, location, threshold)

        # Nodes used in full grid (locations)
        self.nodes = [(self.i + i, self.j + j) for i in range(3) for j in range(4)]

        self.epsilon = -1  # if value_node is less than this, delete the given module.
        self.addr_nodes = self.nodes[:8]
        self.threshold_node = self.nodes[-4]
        self.module_type_nodes = self.nodes[-3:-1]
        self.value_node = self.nodes[-1]

        # Index == key, maps the binary num indicated by module type nodes to the type of
        # module this will be editing. Same logic for numerical parsing as addresses.
        # e.g. node vals of [-3.43, 1.21] -> [0, 1] -> 01 -> 1 -> map[1] -> EdgeModule.
        self.module_type_mapping = [NodeModule, EdgeModule, MemoryModule, MetaModule]

    @staticmethod
    def get_module_type(values, type_nodes, type_mapping):

        """
        Given a grid and the nodes on it which contain a raw module type input for
            fully indicating one of four modules for this metamodule to reference,

            Convert to binary: <=0 is 0, >0 is 1
            Convert to decimal
            Use as index mapping to get module type
        # e.g. node vals of [-3.43, 1.21] -> [0, 1] -> 01 -> 1 -> map[1] -> EdgeModule.

        :param values: Usual values at this timestep for all addrs.
        :param type_nodes: Type nodes with values indicating which module to use.
        :param type_mapping: Mapping of value -> Module subclass
        :return: one of the given modules in the mapping, a subclass of Module.
        """
        module_type = ""
        for node in type_nodes:
            type_val = values[node]
            module_type += "0" if type_val <= 0 else "1"
        return type_mapping[int(module_type, 2)]

    def is_valid_activation(self, grid, values, core, inputs, outputs, memory, modules):
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
        if values[self.threshold_node] <= self.threshold:
            return False

        self.addr = self.get_address(values, self.addr_nodes)
        self.module_type = self.get_module_type(values, self.module_type_nodes, self.module_type_mapping)

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
            module = self.module_type(self.addr, -1)  # get node positions via dummy module
            for node in module.nodes:
                if not module.in_bounds(node, grid):
                    return False

                if grid[node].exists:
                    return False

        # Otherwise either a matching module already exists or we have room for it,
        # proceed with activation.
        return True

    def activate(self, grid, values, biases, core, inputs, outputs, memory, modules):
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
        value = values[self.value_node]

        # Delete if any module exists, otherwise leave empty.
        if value < self.epsilon and self.module_exists:
            # Remove all edges and references to all nodes in module.
            # Modules can only have inputs, for all their nodes.
            # src (out_edges) -> node (in edges)
            module = modules[self.module_i]

            # Iteratively remove each node.
            for node in module.nodes:
                # Remove all outgoing connections to these nodes.
                for in_node, _weight in grid[node].in_edges:
                    # Remove references from any incoming nodes' out_edges lists.
                    grid[in_node].remove_outgoing(node)

                # Remove the node
                grid[node] = Node()

            # Finally remove the module.
            del modules[self.module_i]

        # Non-delete cases - edit existing or add new if empty
        elif value >= self.epsilon:
            if not self.module_exists:
                # Add
                module = self.module_type(self.addr, threshold=value)
                modules.append(module)
                grid.add_module(module)
            else:
                # Edit threshold (it's already in grid and modules)
                modules[self.module_i].threshold = value
