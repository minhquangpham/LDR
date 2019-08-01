"""Define the self-attention encoder."""

import tensorflow as tf

from opennmt.layers import transformer

from opennmt.encoders.encoder import Encoder
from opennmt.layers.position import SinusoidalPositionEncoder

def transform(inputs, ldr_inputs, name="ldr_bias",type="bias"):
    num_units = int(inputs.get_shape()[-1])
    if type=="bias":
        ldr = tf.layers.dense(ldr_inputs, num_units, name=name, use_bias=False)
        ldr = tf.Print(ldr, [tf.reduce_max(tf.abs(ldr))], message=name+"_bias :", first_n=3, summarize=100)
        return inputs + ldr
    elif type=="scale_bias":
        ldr_bias = tf.layers.dense(ldr_inputs, num_units, name=name+"_bias", use_bias=False)
        ldr_scale = tf.layers.dense(ldr_inputs, num_units, name=name+"_scale", use_bias=False, activation=tf.keras.activations.exponential)
        ldr_bias = tf.Print(ldr_bias, [tf.reduce_max(tf.abs(ldr_bias))], message=name+"_bias :", first_n=3, summarize=100)
        ldr_bias = tf.Print(ldr_scale, [tf.reduce_max(tf.abs(ldr_scale))], message="_scale :"%l, first_n=3, summarize=100)
        return tf.multiply(inputs + ldr_bias, ldr_scale)

class SelfAttentionEncoder(Encoder):
  
  def __init__(self,
               num_layers,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.0,
               attention_dropout=0.0,
               relu_dropout=0.0,
               position_encoder=SinusoidalPositionEncoder(),
               ldr_attention_type="bias"):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
      attention_dropout: The probability to drop units from the attention.
      relu_dropout: The probability to drop units from the ReLU activation in
        the feed forward layer.
      position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
        apply on inputs or ``None``.
    """
    self.ldr_attention_type = ldr_attention_type
    self.num_layers = num_layers
    self.num_units = num_units
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.relu_dropout = relu_dropout
    self.position_encoder = position_encoder

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    dim = int(inputs.get_shape()[-1])
    print("total dim", dim)
    inputs, ldr_inputs = tf.split(value=inputs, num_or_size_splits=[self.num_units, dim - self.num_units], axis=-1)
    inputs *= self.num_units**0.5
    #print("inputs",inputs)
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = tf.layers.dropout(
        inputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    ldr_inputs = tf.Print(ldr_inputs, [tf.reduce_mean(ldr_inputs)], message="ldr_encoder_input: ", first_n=3, summarize=100)
    inputs = transform(inputs, ldr_inputs, name="ldr_layer_0", type=self.ldr_attention_type)
    #inputs = tf.Print(inputs,[inputs],message="ldr_encoder_0: ", summarize=200000)
    mask = transformer.build_sequence_mask(
        sequence_length,
        num_heads=self.num_heads,
        maximum_length=tf.shape(inputs)[1])

    state = ()    
    for l in range(self.num_layers):
      with tf.variable_scope("layer_{}".format(l)):
        with tf.variable_scope("multi_head"):
          context = transformer.multi_head_attention(
              self.num_heads,
              transformer.norm(inputs),
              None,
              mode,
              num_units=self.num_units,
              mask=mask,
              dropout=self.attention_dropout)
          context = transformer.drop_and_add(
              inputs,
              context,
              mode,
              dropout=self.dropout)

        with tf.variable_scope("ffn"):
          transformed = transformer.feed_forward(
              transformer.norm(context),
              self.ffn_inner_dim,
              mode,
              dropout=self.relu_dropout)
          transformed = transformer.drop_and_add(
              context,
              transformed,
              mode,
              dropout=self.dropout)

        inputs = transformed
        
        inputs = transform(inputs, ldr_inputs, name="ldr_layer_%d"%l, type=self.ldr_attention_type)
        state += (tf.reduce_mean(inputs, axis=1),)

    outputs = transformer.norm(inputs)
    return (outputs, state, sequence_length)
