import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from random import randint

class Network():
    def __init__(self):
        self.net = np.zeros((16, 16))
        self.net[np.diag_indices(4)] = -10
        self.net[0, 1] = -10
        self.net[:7, 0] = -10

        self.fig, self.ax = plt.subplots()
        self.mat = self.ax.matshow(self.net)
        self.cb = plt.colorbar(self.mat)
        self.frames_n = 128

    def update(self, frame_i):
        # each timestep
        i,j = randint(0,15), randint(0,15)
        self.net[i,j] = randint(0,10)
        self.mat.set_data(self.net)
        #update colorbar limits each time
        self.mat.set_clim(np.amin(self.net), np.amax(self.net))

    def main(self):
        _ = animation.FuncAnimation(self.fig, self.update, frames=self.frames_n, interval=50, repeat=False)#needs ref
        plt.show()

if __name__ == "__main__":
    n = Network()
    n.main()