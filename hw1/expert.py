import load_policy
import numpy as np


class Expert(object):
  def __init__(self, env_fn, expert_fpath):
    self.env = env_fn()
    self.policy_fn = load_policy.load_policy(expert_fpath)

  def policy(self, obs):
    return self.policy_fn(obs[None, :])

  def evaluate(self):
    print('\n Evaluating Expert...')
    num_episodes = 10
    rewards = []
    env = self.env

    for _ in range(num_episodes):
      total_rewards = 0.
      obs = env.reset()
      while True:
        action = self.policy(obs)
        new_obs, reward, done, _ = env.step(action)
        obs = new_obs
        total_rewards += reward
        if done:
          break
      rewards.append(total_rewards)    

    print(f'reward: {np.mean(rewards)}+-{np.std(rewards)}')


if __name__ == '__main__':
  import gym
  import tensorflow as tf
  env_fn = lambda: gym.make('Hopper-v2')
  with tf.Session():
    expert = Expert(env_fn, 'experts/Hopper-v2.pkl')
    expert.evaluate()
