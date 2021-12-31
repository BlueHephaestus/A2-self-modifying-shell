import gym
import time
from hex.hex import HexNetwork

# TODO maybe make these encapsulated into an abstract base class so we can more readily throw nets at large amts of sims
sim = gym.make('CartPole-v1')
net = HexNetwork(sim.action_space.n, sim.observation_space.shape[0])
sim.reset()
sim.render()
# random action to get initial state values
state, reward, done, info = sim.step(sim.action_space.sample())

for _ in range(1000):
    #print(state)
    action = net.activate(state)
    sim.render()
    sim.step(action)
    if done:
        break
    time.sleep(.1)
sim.close()