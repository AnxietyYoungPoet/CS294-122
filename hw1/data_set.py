import numpy as np
import pickle


class Expert_data(object):
  def __init__(self, fpath):
    self.load_data(fpath)
    self.pointer = 0

  def load_data(self, fpath):
    with open(fpath, 'rb') as f:
      data = pickle.load(f)
    self.obs = data['observations']
    self.acts = data['actions']
    self.acts = self.acts.reshape(len(self.acts), -1)
    self.rewards = data['ep_ret']
    self.size = len(self.obs)
    print(f'total transitions: {self.size}, rewards: {np.mean(self.rewards)}+-{np.std(self.rewards)}')
    indexes = [i for i in range(self.size)]
    np.random.shuffle(indexes)
    self.obs = self.obs[indexes]
    self.acts = self.acts[indexes]
    self.eval_size = int(0.3 * self.size)
    self.obs_eval = self.obs[: self.eval_size]
    self.acts_eval = self.acts[: self.eval_size]
    self.bc_obs = self.obs[self.eval_size:]
    self.bc_acts = self.acts[self.eval_size:]
    self.bc_size = self.size - self.eval_size

  def bc_sample_batch(self, batch_size=32):
    if self.pointer + batch_size > self.bc_size:
      self.pointer = 0
    batch_obs = self.bc_obs[self.pointer: self.pointer + batch_size]
    batch_acts = self.bc_acts[self.pointer: self.pointer + batch_size]
    self.pointer += batch_size
    return batch_obs, batch_acts


if __name__ == '__main__':
  expert_data = Expert_data('expert_data/Hopper-v2.pkl')
  obs, acts = expert_data.bc_sample_batch()
  print(obs.shape, acts.shape)
