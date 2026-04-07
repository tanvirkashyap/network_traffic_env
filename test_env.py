import NetworkTrafficEnvironment, NetworkAction from server.environment
env = NetworkTrafficEnvironment()
obs = env.reset()

while not obs.done:
    action = NetworkAction(action_id=np.random.choice([0,1,2]))
    obs = env.step(action)

print(env.state())