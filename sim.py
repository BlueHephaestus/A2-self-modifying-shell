import gym
import time
from hex import HexNetwork

sim = gym.make('CartPole-v0')
net = HexNetwork(sim.action_space.n, sim.observation_space.shape[0])
sim.reset()
sim.render()
# random action to get initial state values
state, reward, done, info = sim.step(sim.action_space.sample())

for _ in range(1000):
    print(state)
    action = net.activate(state)
    if done:
        break
    time.sleep(.5)
sim.close()