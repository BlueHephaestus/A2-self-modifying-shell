from hex.modules.module import Module


class Inputs(Module):
    def __init__(self, location, input_n):
        # Keeps in a 4x4 quadrant from initial location.
        Module.__init__(self, location, -1)
        self.i, self.j = location
        self.nodes = [(self.i + i // 4, self.i + i % 4) for i in range(input_n)]


class Outputs(Module):
    def __init__(self, location, output_n, threshold):
        # Keeps in a 4x4 quadrant from initial location.
        # Threshold node not included in number of outputs, is the last output node.
        Module.__init__(self, location, threshold)
        self.i, self.j = location
        self.nodes = [(self.i + i // 4, self.i + i % 4) for i in range(output_n + 1)]
