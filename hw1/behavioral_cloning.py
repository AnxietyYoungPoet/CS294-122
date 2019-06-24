import gym
import tensorflow as tf
import numpy as np
import pickle
import core
from data_set import Expert_data
import time


class BC(object):
  def __init__(self, env_fn, fpath, epochs=50, batch_size=64, lr=1e-3,
               max_ep_len=1000, ac_kwargs=dict(), save_freq=50, exp_name=''):
    self.env = env_fn()
    self.test_env = env_fn()
    self.expert_data = Expert_data(fpath)
    self.epochs = epochs
    self.batch_size = batch_size
    self.lr = lr
    self.max_ep_len = max_ep_len
    self.save_freq = save_freq
    self.exp_name = exp_name

    self.obs_dim = self.env.observation_space.shape[0]
    self.act_dim = self.env.action_space.shape[0]
    self.act_limit = self.env.action_space.high[0]
    self.ac_kwargs = ac_kwargs
    self.ac_kwargs['action_space'] = self.env.action_space
    self.build()

  def add_place_holders(self):
    self.x_ph, self.a_ph = core.placeholders(self.obs_dim, self.act_dim)

  def build(self):
    self.add_place_holders()
    self.pi = core.mlp_policy(self.x_ph, self.a_ph, **self.ac_kwargs)

    self.loss = tf.reduce_mean(tf.reduce_sum((self.pi - self.a_ph)**2, axis=1))
    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.train_pi_op = optimizer.minimize(self.loss)

  def initialize(self):
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver(max_to_keep=None)

  def get_action(self, o):
    a = self.sess.run(self.pi, feed_dict={self.x_ph: o.reshape(1, -1)})[0]
    return a

  def test_agent(self, n=10):
    total_rewards = []
    total_lens = []
    for _ in range(n):
      o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
      while not(d or (ep_len) == self.max_ep_len):
        o, r, d, _ = self.env.step(self.get_action(o))
        ep_ret += r
        ep_len += 1
      total_rewards.append(ep_ret)
      total_lens.append(ep_len)
    return np.mean(total_rewards), np.sqrt(np.var(total_rewards)), np.mean(total_lens), np.sqrt(np.var(total_lens))

  def save_model(self):
    import os
    model_output = os.path.join('models', 'bc', self.exp_name)
    if not os.path.exists(model_output):
      os.makedirs(model_output)
    model_path = os.path.join(model_output, 'model.ckpt')
    self.saver.save(self.sess, model_path)

  def train(self):
    start_time = time.time()
    steps_per_epoch = self.expert_data.bc_size // self.batch_size + 1
    min_loss = np.inf
    for ep in range(self.epochs):
      ep_loss = 0.
      test_losses = []
      for t in range(steps_per_epoch):
        bacth_obs, batch_acts = self.expert_data.bc_sample_batch(self.batch_size)
        loss, _ = self.sess.run(
          [self.loss, self.train_pi_op], feed_dict={self.x_ph: bacth_obs, self.a_ph: batch_acts})
        ep_loss += loss
        if (t > 0 and t % self.save_freq == 0) or t == steps_per_epoch - 1:
          obs = self.expert_data.obs_eval
          acts = self.expert_data.acts_eval
          test_loss = self.sess.run(self.loss, feed_dict={self.x_ph: obs, self.a_ph: acts})
          test_losses.append(test_loss)
          if test_loss < min_loss:
            min_loss = test_loss
            self.save_model()
      rewards_mean, rewards_std, ep_len_mean, ep_len_std = self.test_agent()
      print(f'time: {time.time() - start_time}, trainning loss: {ep_loss / steps_per_epoch}, test loss: {np.mean(test_losses)}'
            f'  test rewards: {rewards_mean}+-{rewards_std}, ep len: {ep_len_mean}+-{ep_len_std}')

  def eval(self):
    import os
    model_output = os.path.join('models', 'bc', self.exp_name)
    ckpt = tf.train.get_checkpoint_state(model_output)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    rewards_mean, rewards_std, ep_len_mean, ep_len_std = self.test_agent()
    print(f'final evaluation: test rewards: {rewards_mean}+-{rewards_std}, ep len: {ep_len_mean}+-{ep_len_std}')

  def run(self):
    self.initialize()
    self.train()
    self.eval()


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('expert_data', type=str)
  parser.add_argument('env_name', type=str)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--hid', type=str, default='[256, 256]')
  parser.add_argument('--activation', type=str, default='tf.tanh')
  args = parser.parse_args()

  ac_kwargs = dict()
  ac_kwargs['hidden_size'] = eval(args.hid)
  ac_kwargs['activation'] = eval(args.activation)

  env_fn = lambda: gym.make(args.env_name)
  bc = BC(env_fn, args.expert_data, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, ac_kwargs=ac_kwargs, exp_name=args.env_name)
  bc.run()


if __name__ == '__main__':
  main()
