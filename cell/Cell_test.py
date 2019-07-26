import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops, array_ops
from subprocess import Popen
from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn import *
import numpy as np
import time
from logging import getLogger
from opennmt.decoders.rnn_decoder import _build_attention_mechanism, RNNDecoder
from interpolated_nmt.cell.cell import *

logger = getLogger()

class Interpolated_AttentionalRNNDecoder(RNNDecoder):
  
  def __init__(self,
               num_layers,
               num_units,
               bridge = None,
               attention_mechanism_class = tf.contrib.seq2seq.LuongAttention,
               output_is_attention = True,
               cell_class = Interpolated_GRUCell,
               dropout = 0.3,
               residual_connections = False):
    
    super(Interpolated_AttentionalRNNDecoder, self).__init__(
        num_layers,
        num_units,
        bridge = bridge,
        cell_class = cell_class,
        dropout = dropout,
        residual_connections = residual_connections)
    self.attention_mechanism_class = attention_mechanism_class
    self.output_is_attention = output_is_attention

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None,
                  dtype=None,
                  alignment_history=False):

    attention_mechanism = _build_attention_mechanism(
        self.attention_mechanism_class,
        self.num_units,
        memory,
        memory_sequence_length = memory_sequence_length)

    cell, initial_cell_state = build_cell(
        mode,
        batch_size,
        initial_state = initial_state,
        dtype = memory.dtype)

    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size = self.num_units,
        alignment_history = alignment_history,
        output_attention = self.output_is_attention,
        initial_cell_state = initial_cell_state)

    if mode == tf.estimator.ModeKeys.TRAIN and self.dropout > 0.0:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob = 1.0 - self.dropout)

    initial_state = cell.zero_state(batch_size, memory.dtype)

    return cell, initial_state

seq_length = tf.placeholder(name="seq_length", dtype=tf.int32)
input_data = tf.placeholder(name="input_data", shape=[None,None], dtype=tf.int32)
batch_size = tf.shape(input_data)[0]          

vocab_size = 10
embedding_size = 6
hidden_size = 12
encoder = "LSTM"
     
with tf.device("/cpu:0"):
    embedding = tf.get_variable(name = "word_embeddings", shape = [vocab_size, embedding_size-2], initializer = tf.ones_initializer() , dtype=tf.float32)
    embedding = tf.pad(embedding,[[0,0],[2,0]])
    ind = tf.ones(shape=[vocab_size, hidden_size-4], dtype = tf.float32)
    ind = tf.pad(ind,[[0,0],[4,0]])
    input = tf.concat([tf.nn.embedding_lookup(embedding, input_data, name="input_matrix"), tf.nn.embedding_lookup(ind, input_data, name="domain_matrix")],axis=2)
    # build RNN encoder of target sentence:
    
    with tf.variable_scope("encoder"):
        if encoder == "GRU":
            cell_fn = Interpolated_GRUCell
        elif encoder == "LSTM":
            cell_fn = rnn.BasicLSTMCell
            
        input_num_unit = [2,2,2]
        output_num_unit = [4,4,4]
        fw_cell = cell_fn(12, input_num_unit, output_num_unit, 2)     
        bw_cell = cell_fn(12, input_num_unit, output_num_unit, 2)

        initial_state_fw = fw_cell.zero_state(batch_size, dtype=tf.float32)
        initial_state_bw = bw_cell.zero_state(batch_size, dtype=tf.float32)

        (output_fw, output_bw), (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input, sequence_length = seq_length, initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)

        encoded = tf.concat([output_state_fw, output_state_bw], axis=1)
    
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    feed_dict = {input_data:[[0,1,2,0,0,0]], seq_length:[3]}
    print sess.run([input, output_fw], feed_dict)









