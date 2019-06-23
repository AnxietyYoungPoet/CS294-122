import tensorflow as tf
import pickle


def placeholder(dim=None):
  return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
  return [placeholder(dim) for dim in args]


def mlp(x, hidden_size=(32,), activation=tf.tanh, output_activation=None):
  for h in hidden_size[:-1]:
    x = tf.layers.dense(x, units=h, activation=activation)
  return tf.layers.dense(x, units=hidden_size[-1], activation=output_activation)


def mlp_policy(x, a, hidden_size=(128, 128), activation=tf.nn.relu,
               output_activation=tf.tanh, action_space=None):
  act_dim = a.shape.as_list()[-1]
  act_limit = action_space.high[0]
  with tf.variable_scope('pi'):
    pi = act_limit * mlp(x, list(hidden_size) + [act_dim], activation, output_activation)
  return pi
