# pylint: disable=W0223

"""Define RNN-based decoders."""

import inspect

import tensorflow as tf

from tensorflow.python.estimator.util import fn_args

from opennmt.decoders import decoder
from opennmt.utils.cell import build_cell
from opennmt.layers.reducer import align_in_time
from opennmt.layers.transformer import build_sequence_mask, multi_head_attention
from opennmt.utils import beam_search

def get_embedding_fn(embedding):
  """Returns the embedding function.

  Args:
    embedding: The embedding tensor or a callable that takes word ids.

  Returns:
    A callable that takes word ids.
  """
  if callable(embedding):
    return embedding
  else:
    return lambda ids: tf.nn.embedding_lookup(embedding, ids)

class RNNDecoder_v2(decoder.Decoder):
  """A basic RNN decoder."""

  def __init__(self,
               num_layers,
               num_units,
               bridge=None,
               cell_class=tf.nn.rnn_cell.LSTMCell,
               dropout=0.3,
               residual_connections=False):    
    self.num_layers = num_layers
    self.num_units = num_units
    self.bridge = bridge
    self.cell_class = cell_class
    self.dropout = dropout
    self.residual_connections = residual_connections

  @property
  def output_size(self):
    """Returns the decoder output size."""
    return self.num_units 

  def _init_state(self, zero_state, initial_state=None):
    if initial_state is None:
      return zero_state
    elif self.bridge is None:
      raise ValueError("A bridge must be configured when passing encoder state")
    else:
      return self.bridge(initial_state, zero_state)

  def _get_attention(self, state, step=None):  # pylint: disable=unused-argument
    return None

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None,
                  dtype=None):
    _ = memory_sequence_length

    if memory is None and dtype is None:
      raise ValueError("dtype argument is required when memory is not set")

    cell = build_cell(
        self.num_layers,
        self.num_units,
        mode,
        dropout=self.dropout,
        residual_connections=self.residual_connections,
        cell_class=self.cell_class)

    initial_state = self._init_state(
        cell.zero_state(batch_size, dtype or memory.dtype if not isinstance(memory,list) else memory[0].dtype ), initial_state=initial_state)

    return cell, initial_state

  def decode(self,
             inputs,
             sequence_length,
             vocab_size=None,
             initial_state=None,
             sampling_probability=None,
             embedding=None,
             output_layer=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None,
             return_alignment_history=False):
    _ = memory
    _ = memory_sequence_length

    batch_size = tf.shape(inputs)[0]

    if (sampling_probability is not None
        and (tf.contrib.framework.is_tensor(sampling_probability)
             or sampling_probability > 0.0)):
      if embedding is None:
        raise ValueError("embedding argument must be set when using scheduled sampling")

      tf.summary.scalar("sampling_probability", sampling_probability)
      helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
          inputs,
          sequence_length,
          embedding,
          sampling_probability)
      fused_projection = False
    else:
      helper = tf.contrib.seq2seq.TrainingHelper(inputs, sequence_length)
      fused_projection = True  # With TrainingHelper, project all timesteps at once.

    cell, initial_state = self._build_cell(
        mode,
        batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=inputs.dtype)

    if output_layer is None:
      output_layer = decoder.build_output_layer(
          2 * self.output_size + inputs.get_shape()[-1], vocab_size, dtype=inputs.dtype)

    basic_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell,
        helper,
        initial_state,
        output_layer=output_layer if not fused_projection else None)

    outputs, state, length = tf.contrib.seq2seq.dynamic_decode(basic_decoder,scope="decoder")
    inputs_len = tf.shape(inputs)[1]    
    if fused_projection and output_layer is not None:
      logits = output_layer(tf.concat([align_in_time(outputs.rnn_output, inputs_len), inputs],-1))
    else:
      logits = tf.concat([align_in_time(outputs.rnn_output, inputs_len), inputs],-1) #outputs.rnn_output
    # Make sure outputs have the same time_dim as inputs
    #inputs_len = tf.shape(inputs)[1]
    #logits = align_in_time(logits, inputs_len)

    if return_alignment_history:
      alignment_history = self._get_attention(state)
      if alignment_history is not None:
        alignment_history = align_in_time(alignment_history, inputs_len)
      return (logits, state, length, alignment_history)
    return (logits, state, length)

  def step_fn(self,
              mode,
              batch_size,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              dtype=tf.float32):
    cell, initial_state = self._build_cell(
        mode,
        batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=dtype)

    def _fn(step, inputs, state, mode):
      _ = mode
      # This scope is defined by tf.contrib.seq2seq.dynamic_decode during the
      # training.
      with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        outputs, state = cell(inputs, state)
        if self.support_alignment_history:
          return tf.concat([outputs, inputs],-1), state, self._get_attention(state, step=step)
        return tf.concat([outputs, inputs],-1), state
    return _fn, initial_state

  def dynamic_decode_and_search(self,
                                embedding,
                                start_tokens,
                                end_token,
                                vocab_size=None,
                                initial_state=None,
                                output_layer=None,
                                beam_width=5,
                                length_penalty=0.0,
                                maximum_iterations=250,
                                minimum_length=0,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=None,
                                memory_sequence_length=None,
                                dtype=None,
                                return_alignment_history=False,
                                sample_from=1):
    """Decodes dynamically from :obj:`start_tokens` with beam search.

    Usually used for inference.

    Args:
      embedding: The embedding tensor or a callable that takes word ids.
      start_tokens: The start token ids with shape :math:`[B]`.
      end_token: The end token id.
      vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
        is not set.
      initial_state: The initial state as a (possibly nested tuple of...) tensors.
      output_layer: Optional layer to apply to the output prior sampling.
        Must be set if :obj:`vocab_size` is not set.
      beam_width: The width of the beam.
      length_penalty: The length penalty weight during beam search.
      maximum_iterations: The maximum number of decoding iterations.
      minimum_length: The minimum length of decoded sequences (:obj:`end_token`
        excluded).
      mode: A ``tf.estimator.ModeKeys`` mode.
      memory: (optional) Memory values to query.
      memory_sequence_length: (optional) Memory values length.
      dtype: The data type. Required if :obj:`memory` is ``None``.
      return_alignment_history: If ``True``, also returns the alignment
        history from the attention layer (``None`` will be returned if
        unsupported by the decoder).
      sample_from: Sample predictions from the :obj:`sample_from` most likely
        tokens. If 0, sample from the full output distribution.

    Returns:
      A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
      ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
      if :obj:`return_alignment_history` is ``True``.
    """
    if sample_from != 1 and beam_width > 1:
      raise ValueError("Sampling decoding is not compatible with beam search, "
                       "set beam_width to 1 instead.")
    batch_size = tf.shape(start_tokens)[0] * beam_width
    if dtype is None:
      if memory is None:
        raise ValueError("dtype argument is required when no memory is set")
      dtype = tf.contrib.framework.nest.flatten(memory)[0].dtype

    if beam_width > 1:
      if initial_state is not None:
        initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier=beam_width)
      if memory is not None:
        memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
      if memory_sequence_length is not None:
        memory_sequence_length = tf.contrib.seq2seq.tile_batch(
            memory_sequence_length, multiplier=beam_width)

    embedding_fn = get_embedding_fn(embedding)
    step_fn, initial_state = self.step_fn(
        mode,
        batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=dtype)
    if output_layer is None:
      if vocab_size is None:
        raise ValueError("vocab_size must be known when the output_layer is not set")
      output_layer = decoder.build_output_layer(2 * self.output_size + embedding.get_shape()[-1], vocab_size, dtype=dtype)

    state = {"decoder": initial_state}
    if self.support_alignment_history and not tf.contrib.framework.nest.is_sequence(memory):
      state["attention"] = tf.zeros([batch_size, 0, tf.shape(memory)[1]], dtype=dtype)

    def _symbols_to_logits_fn(ids, step, state):
      if ids.shape.ndims == 2:
        ids = ids[:, -1]
      inputs = embedding_fn(ids)
      returned_values = step_fn(step, inputs, state["decoder"], mode)
      if self.support_alignment_history:
        outputs, state["decoder"], attention = returned_values
        if "attention" in state:
          state["attention"] = tf.concat([state["attention"], tf.expand_dims(attention, 1)], 1)
      else:
        outputs, state["decoder"] = returned_values
      logits = output_layer(outputs)
      return logits, state

    if beam_width == 1:
      outputs, lengths, log_probs, state = greedy_decode(
          _symbols_to_logits_fn,
          start_tokens,
          end_token,
          decode_length=maximum_iterations,
          state=state,
          return_state=True,
          min_decode_length=minimum_length,
          last_step_as_input=True,
          sample_from=sample_from)
    else:
      outputs, log_probs, state = beam_search.beam_search(
          _symbols_to_logits_fn,
          start_tokens,
          beam_width,
          maximum_iterations,
          vocab_size,
          length_penalty,
          states=state,
          eos_id=end_token,
          return_states=True,
          tile_states=False,
          min_decode_length=minimum_length)
      lengths = tf.not_equal(outputs, 0)
      lengths = tf.cast(lengths, tf.int32)
      lengths = tf.reduce_sum(lengths, axis=-1) - 1  # Ignore </s>
      outputs = outputs[:, :, 1:]  # Ignore <s>.

    attention = state.get("attention")
    if beam_width == 1:
      # Make shape consistent with beam search.
      outputs = tf.expand_dims(outputs, 1)
      lengths = tf.expand_dims(lengths, 1)
      log_probs = tf.expand_dims(log_probs, 1)
      if attention is not None:
        attention = tf.expand_dims(attention, 1)

    if return_alignment_history:
      return (outputs, state["decoder"], lengths, log_probs, attention)
    return (outputs, state["decoder"], lengths, log_probs)

def _build_attention_mechanism(attention_mechanism,
                               num_units,
                               memory,
                               memory_sequence_length=None):
  """Builds an attention mechanism from a class or a callable."""
  if inspect.isclass(attention_mechanism):
    kwargs = {}
    if "dtype" in fn_args(attention_mechanism):
      # For TensorFlow 1.5+, dtype should be set in the constructor.
      kwargs["dtype"] = memory.dtype if not isinstance(memory,list) else memory[0].dtype      
       
    return attention_mechanism(
        num_units, memory, memory_sequence_length=memory_sequence_length, **kwargs)
  elif callable(attention_mechanism):
    return attention_mechanism(
        num_units, memory, memory_sequence_length)
  else:
    raise ValueError("Unable to build the attention mechanism")


class AttentionalRNNDecoder_v2(RNNDecoder_v2):
  """A RNN decoder with attention.

  It simple overrides the cell construction to add an attention wrapper.
  """

  def __init__(self,
               num_layers,
               num_units,
               bridge=None,
               attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
               output_is_attention=True,
               cell_class=tf.nn.rnn_cell.LSTMCell,
               dropout=0.3,
               residual_connections=False):
    
    super(AttentionalRNNDecoder_v2, self).__init__(
        num_layers,
        num_units,
        bridge=bridge,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections)
    self.attention_mechanism_class = attention_mechanism_class
    self.output_is_attention = output_is_attention

  @property
  def support_alignment_history(self):
    return True

  def _get_attention(self, state, step=None):
    alignment_history = state.alignment_history
    if step is not None:
      return alignment_history.read(step)
    return tf.transpose(alignment_history.stack(), perm=[1, 0, 2])

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None,
                  dtype=None):
    with tf.variable_scope("attention_mechanism",reuse=tf.AUTO_REUSE):
      attention_mechanism = _build_attention_mechanism(
        self.attention_mechanism_class,
        self.num_units,
        memory,
        memory_sequence_length=memory_sequence_length)
    
    cell, initial_cell_state = RNNDecoder_v2._build_cell(
        self,
        mode,
        batch_size,
        initial_state=initial_state,
        dtype=memory.dtype if not isinstance(memory,list) else memory[0].dtype)

    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=self.num_units,
        alignment_history=True,
        output_attention=self.output_is_attention,
        initial_cell_state=initial_cell_state,
        name="Attention_layer")

    if mode == tf.estimator.ModeKeys.TRAIN and self.dropout > 0.0:
      cell = tf.nn.rnn_cell.DropoutWrapper(
          cell, output_keep_prob=1.0 - self.dropout)

    initial_state = cell.zero_state(batch_size, memory.dtype if not isinstance(memory,list) else memory[0].dtype )

    return cell, initial_state
