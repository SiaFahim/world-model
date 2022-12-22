# Building the Reinforcement Learning Model

obs = env.reset()
h = mdnrnn.initial_state()
done = False
comulative_reward = 0

while not done:
    z = cnnvae(obs) # get the latent representation of the observation
    a = controller([z, h]) # get the action from the controller
    obs, r, done = env.step(a) # take the action
    comulative_reward += r
    h = mdnrnn([a, z, h]) # get the next hidden state










