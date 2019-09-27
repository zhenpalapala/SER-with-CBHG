# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from attention import attention
import numpy as np
epsilon = 1e-3
#input:
def encoder_cbhg(inputs, is_training, depth):
  input_channels = inputs.get_shape()[2]
 # input_channels = 40
  return cbhg(
    inputs,
    is_training,
    scope='encoder_cbhg',
    K=8,
    projections=[300, input_channels],
    depth=depth)
def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=channels,
      kernel_size=kernel_size,
      activation=activation,
      padding='same')
    return tf.layers.batch_normalization(conv1d_output, training=is_training)
def highwaynet(inputs, scope, depth):
  with tf.variable_scope(scope):
    H = tf.layers.dense(
      inputs,
      units=depth,
      activation=tf.nn.relu,
      name='H')
    T = tf.layers.dense(
      inputs,
      units=depth,
      activation=tf.nn.sigmoid,
      name='T',
      bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)

def cbhg(inputs, is_training, scope, K, projections, depth):
  with tf.variable_scope(scope):
    with tf.variable_scope('conv_bank'):
      # Convolution bank: concatenate on the last axis to stack channels from all convolutions
      conv_bank=[]
      for k in range(1,K+1):
         # x=tf.Session().run(inputs)
         # print(x)
         # con1d_output=conv1d(inputs,k,128,tf.nn.relu,is_training,'conv1d_%d' %k)
          con1d_output=tf.layers.conv1d(inputs,128, k, padding='same',name='conv1d_%d' %k)
          con1d_output_bn=tf.keras.layers.BatchNormalization(name='conv1d_%d_bn' %k)(con1d_output,training=is_training)
          #pool=tf.layers.max_pooling1d(con1d_output_bn,pool_size=128,strides=1,padding='valid')
          conv_bank.append(con1d_output_bn)
      conv_outputs=tf.concat(conv_bank,axis=-1)


      #conv_outputs = tf.concat(
       # [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in filter_size],
        #axis=-1
     # )
    '''
    n_in=40
    filter_size=(2,3,4,5,8)
    convs=[]
    for rsz in filter_size:
      conv=conv1d(inputs,rsz,128,tf.nn.relu,is_training,'conv1d_%d' %rsz)
      pool=tf.layers.max_pooling1d(conv,pool_size=n_in-rsz+1,strides=1,
                    padding='same')
      flatten=tf.layers.Flatten(pool)
      convs.append(flatten)
    if len(filter_size)>1:
      conv_outputs=tf.concat(convs)
    else:
      conv_outputs=convs[0]
    '''
    # Maxpooling:
    maxpool_output = tf.layers.max_pooling1d(
      conv_outputs,
      pool_size=2,
      strides=1,
      padding='same')

    # Two projection layers:
    proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
    proj2_output = conv1d(proj1_output, 3, projections[1], tf.nn.relu, is_training, 'proj_2')
    #proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
    #proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')
    # Residual connection:
    highway_input = proj2_output + inputs

    half_depth = depth // 2
    assert half_depth*2 == depth, 'encoder and postnet depths must be even.'

    # Handle dimensionality mismatch:
    if highway_input.shape[2] != half_depth:
      highway_input = tf.layers.dense(highway_input, half_depth)
    # 4-layer HighwayNet:
    for i in range(3):
      highway_input = highwaynet(highway_input, 'highway_%d' % (i+1), half_depth)
    rnn_input = highway_input

    # Bidirectional RNN
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      GRUCell(half_depth),
      GRUCell(half_depth),
      rnn_input,
      time_major=False,
      dtype=tf.float32,
      scope=('LSTM'))
    outputs=tf.concat(outputs,axis=2)
    print('outputs',outputs,'type of outputs :',type(outputs))
    #return tf.concat(outputs, axis=2)  # Concat forward and backward
    return outputs



def leaky_relu(x, leakiness=0.0):
  return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def batch_norm_wrapper(inputs, is_training, decay=0.999):
  scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
  beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
  pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
  pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

  if is_training is not None:
    batch_mean, batch_var = tf.nn.moments(inputs, [0])
    train_mean = tf.assign(pop_mean,
                           pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var,
                          pop_var * decay + batch_var * (1 - decay))
    with tf.control_dependencies([train_mean, train_var]):
      return tf.nn.batch_normalization(inputs,
                                       batch_mean, batch_var, beta, scale, epsilon)
  else:
    return tf.nn.batch_normalization(inputs,
                                     pop_mean, pop_var, beta, scale, epsilon)


def acrnn(inputs, num_classes=4,
          is_training=True,
          L1=128,
          L2=256,
          cell_units=128,
          depth=256,
          num_linear=768,
          p=10,
          time_step=150,
          F1=64,
          dropout_keep_prob=1):

  fully1_weight = tf.get_variable('fully1_weight', shape=[2 * cell_units, F1], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
  fully1_bias = tf.get_variable('fully1_bias', shape=[F1], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.1))
  fully2_weight = tf.get_variable('fully2_weight', shape=[F1, num_classes], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
  fully2_bias = tf.get_variable('fully2_bias', shape=[num_classes], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.1))
  outputs1=encoder_cbhg(inputs, is_training,depth)
  # Attention layer
  gru, alphas = attention(outputs1, 1, return_alphas=True)

  fully1 = tf.matmul(gru, fully1_weight) + fully1_bias
  fully1 = leaky_relu(fully1, 0.01)
  fully1 = tf.nn.dropout(fully1, dropout_keep_prob)

  Ylogits = tf.matmul(fully1, fully2_weight) + fully2_bias
  print('Ylogits:',Ylogits)
  # Ylogits = tf.nn.softmax(Ylogits)
  return Ylogits
