import tensorflow as tf
import opennmt as onmt
from opennmt.utils.optim import *
from utils.dataprocess import *
from utils.utils_ import *
import argparse
import sys
import numpy as np
from opennmt.inputters.text_inputter import load_pretrained_embeddings
from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.evaluator import *
from opennmt.utils.parallel import GraphDispatcher
from opennmt import constants
import os
import ipdb
import copy
import yaml
import io

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.layers.base import InputSpec
from tensorflow.python.layers.base import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.util.tf_export import tf_export
import cell.rnn_decoder as rnn_decoder
import cell.self_attention_decoder_LDR as self_attention_decoder_LDR
import cell.self_attention_encoder_LDR as self_attention_encoder_LDR

class MyDenseLayer(Layer):
  def __init__(self,
               units,               
               mask,
               fusion_layer=None,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,               
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(MyDenseLayer, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    self.units = int(units)
    self.mask = mask
    self.fusion_layer = fusion_layer
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: input_shape[-1].value})
    self.kernel = self.add_weight(
        'kernel',
        shape=[input_shape[-1].value, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def __call__(self, inputs):
    msk = self.mask[0,:]
    kernel = tf.transpose(self.fusion_layer(tf.multiply(tf.transpose(self.kernel),tf.reshape(msk, [1, -1]))))
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    rank = common_shapes.rank(inputs)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.get_shape().as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      outputs = gen_math_ops.mat_mul(inputs, kernel)    
    
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

def build_output_layer(input_units, output_units, dtype=None, use_bias=True, activation=None, kernel_initializer=tf.glorot_uniform_initializer(), bias_initializer= tf.glorot_uniform_initializer()):
  if output_units is None:
    raise ValueError("output_units must be set to build the output layer")      
  layer = tf.layers.Dense(output_units, activation=activation, use_bias=use_bias, kernel_initializer= kernel_initializer, bias_initializer= bias_initializer)
  layer.build([None, input_units])
  return layer

def GAN_layer(src_length, kernel_size, hidden_size, inputs, domain_numb, mode, collections_ = ["tgt_Adversarial_classifier","Adversarial_classifier"], scope="tgt_Adversarial_classifier"):
  with tf.variable_scope(scope, custom_getter =  lambda getter, name, shape, collections, *args, **kwargs:
                                    getter( name = name, shape = shape, collections= collections_, *args, **kwargs )):
      with tf.variable_scope("classifier"):          
          v = tf.get_variable("v_a", shape=[kernel_size])
          W = tf.get_variable("W_a", shape=[inputs.get_shape()[-1], kernel_size])
          v_a = tf.expand_dims(tf.expand_dims(v, 0),2)
          v_a = tf.tile(v_a, [tf.shape(inputs)[0], 1, 1])
          W_a = tf.expand_dims(W, 0)
          W_a = tf.tile(W_a, [tf.shape(inputs)[0],1,1])
          attention_weight = tf.matmul(tf.tanh(tf.matmul(inputs, W_a)), v_a)
          adv_mask = tf.sequence_mask(src_length, maxlen=tf.shape(attention_weight)[1], dtype=tf.float32)
          adv_mask = tf.expand_dims(adv_mask, -1)
          attention_weight = tf.cast(tf.cast(attention_weight, tf.float32) * adv_mask + ((1.0 - adv_mask) * tf.float32.min), attention_weight.dtype)
          attention_weight = tf.cast(tf.nn.softmax(tf.cast(attention_weight, tf.float32)), attention_weight.dtype)
          attention_weight = tf.squeeze(attention_weight,-1)
          attention_weight = tf.expand_dims(attention_weight, 1)
          Adv_logits = tf.matmul(attention_weight, inputs)
          Adv_logits = tf.squeeze(Adv_logits,1)
          Adv_ff_layer_1 = build_output_layer(hidden_size, 2048, activation=tf.nn.leaky_relu)
          Adv_ff_layer_2 = build_output_layer(2048, 2048, activation=tf.nn.leaky_relu)
          Adv_ff_layer_end = build_output_layer(2048, domain_numb)
          Adv_logits = tf.layers.dropout(Adv_logits, rate=0.3, training= mode == "Training")
          Adv_outputs = Adv_ff_layer_1(Adv_logits)          
          Adv_outputs = tf.layers.dropout(Adv_outputs, rate=0.3, training= mode == "Training")
          Adv_outputs = Adv_ff_layer_2(Adv_outputs)
          Adv_outputs = tf.layers.dropout(Adv_outputs, rate=0.3, training= mode == "Training")
          Adv_outputs = Adv_ff_layer_end(Adv_outputs)
          Adv_predictions = tf.nn.softmax(Adv_outputs)
          Adv_var = [v,W] + Adv_ff_layer_1.variables + Adv_ff_layer_2.variables 
  return Adv_outputs, Adv_var

def create_embeddings(vocab_size, depth=512):
      """Creates an embedding variable."""
      return tf.get_variable("embedding", shape = [vocab_size, depth])

def create_mask(domain_numb, domain_region_size):
    mask_ = []
    #depth = sum(domain_region_size) + shared_region_size
    for i in range(domain_numb):
        mask_.append(tf.pad(tf.pad(tf.zeros(shape=(1, sum(domain_region_size[:i]))), [[0,0], [domain_region_size[i],0]], constant_values=1), [[0,0], [sum(domain_region_size[i+1:]), 0]]))
    mask_.append(tf.zeros(shape=(1,sum(domain_region_size))))
    mask_.append(tf.ones(shape=(1,sum(domain_region_size))))
    return tf.squeeze(tf.concat(mask_,0))

def make_batch(emb_domain, emb_generic, mask_, inputs_domain, inputs_ids):
    mask = tf.nn.embedding_lookup(mask_, inputs_domain)    
    mask = tf.expand_dims(mask,1)
    emb_generic_batch = tf.nn.embedding_lookup(emb_generic, inputs_ids)
    emb_domain_batch = tf.nn.embedding_lookup(emb_domain, inputs_ids)
    #emb_domain_batch = tf.Print(emb_domain_batch,[emb_domain_batch[:5,:]], message="emb_domain_batch: ", first_n=3, summarize=100)
    emb_domain_batch = tf.multiply(emb_domain_batch, mask)
    emb_domain_batch = tf.Print(emb_domain_batch,[emb_domain_batch[:5,:]], message="emb_domain_batch: ", first_n=3, summarize=100)
    return tf.concat([emb_generic_batch, emb_domain_batch],-1)

def extend_embeddings(vocab_size, dom_numb, old_emb):
    depth = tf.shape(old_emb)[-1]/dom_numb
    new_emb = tf.get_variable("embedding_dom_%d"%(dom_numb+1), shape = [vocab_size, depth])
    extended_emb = tf.concat([old_emb,new_emb],-1)
    return extended_emb 

class Model:

    def _compute_loss(self, outputs, tgt_ids_batch, tgt_length, params, mode):
        
        if mode == "Training":
            mode = tf.estimator.ModeKeys.TRAIN            
        else:
            mode = tf.estimator.ModeKeys.EVAL            
          
        logits_generic, logits_domain = outputs["logits"]
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            loss_generic, loss_normalizer, loss_token_normalizer = cross_entropy_sequence_loss(logits_generic,
                                                                                   tgt_ids_batch, 
                                                                                   tgt_length + 1,
                                                                                   label_smoothing=params.get("label_smoothing", 0.0),
                                                                                   average_in_time=params.get("average_loss_in_time", True),
                                                                                   mode=mode)
            print("loss_generic", loss_generic)
            print("tgt_ids_batch", tgt_ids_batch)
            loss_domain, _, _                                    = cross_entropy_sequence_loss(logits_domain,
                                                                                   tgt_ids_batch, 
                                                                                   tgt_length + 1,
                                                                                   label_smoothing=params.get("label_smoothing", 0.0),
                                                                                   average_in_time=params.get("average_loss_in_time", True),
                                                                                   mode=mode)
        return loss_generic, loss_domain, loss_normalizer, loss_token_normalizer

    def _initializer(self, params):        
        if params["Architecture"] == "Transformer":
            print("tf.variance_scaling_initializer")
            return tf.variance_scaling_initializer(
        mode="fan_avg", distribution="uniform", dtype=self.dtype)
        else:            
            param_init = params.get("param_init")
            if param_init is not None:
                print("tf.random_uniform_initializer")
                return tf.random_uniform_initializer(
              minval=-param_init, maxval=param_init, dtype=self.dtype)
        return None
        
    def __init__(self, config_file, mode, test_feature_file=None, test_tag_file=None):

        def _normalize_loss(num, den=None):
            """Normalizes the loss."""
            if isinstance(num, list):  # Sharded mode.
                if den is not None:
                    assert isinstance(den, list)
                    return tf.add_n(num) / tf.add_n(den) #tf.reduce_mean([num_/den_ for num_,den_ in zip(num, den)]) #tf.add_n(num) / tf.add_n(den)
                else:
                    return tf.reduce_mean(num)
            elif den is not None:
                return num / den
            else:
                return num

        def _extract_loss(loss, Loss_type="Cross_Entropy"):
            """Extracts and summarizes the loss."""
            losses = None
            print("loss numb:", len(loss))
            if Loss_type=="Cross_Entropy":
                if not isinstance(loss, tuple):                    
                    print(1)
                    actual_loss = _normalize_loss(loss)
                    tboard_loss = actual_loss
                    tf.summary.scalar("loss", tboard_loss)
                    losses = actual_loss                    
                else:                    
                    print(5)      
                    generic_loss = _normalize_loss(loss[0], den=loss[3])
                    domain_loss = _normalize_loss(loss[1], den=loss[3])
                    tf.summary.scalar("loss_generic", generic_loss)            
                    tf.summary.scalar("loss_domain", domain_loss)                    
                    losses = (generic_loss, domain_loss)
            return losses                         

        def _loss_op(inputs, params, mode):
            """Single callable to compute the loss."""
            if mode=="Training":
                logits, _, tgt_ids_out, tgt_length  = self._build(inputs, params, mode)
                losses = self._compute_loss(logits, tgt_ids_out, tgt_length, params, mode)
                return losses

            elif mode=="Inference":
                _, predictions, _, _ = self._build(inputs, config, mode)
                return predictions
        
        with open(config_file, "r") as stream:
            config = yaml.load(stream)

        Loss_type = config.get("Loss_Function","Cross_Entropy")
        self.Loss_type = Loss_type
        self.config = config 
        self.using_tf_idf = config.get("using_tf_idf", False)
        train_batch_size = config["training_batch_size"]   
        eval_batch_size = config["eval_batch_size"]
        max_len = config["max_len"]
        example_sampling_distribution = config.get("example_sampling_distribution",None)
        self.dtype = tf.float32
        # Input pipeline:
        src_vocab, _ = load_vocab(config["src_vocab_path"], config["src_vocab_size"])
        tgt_vocab, _ = load_vocab(config["tgt_vocab_path"], config["tgt_vocab_size"])
        load_data_version = config.get("dataprocess_version",None)
        if mode == "Training":    
            print("num_devices", config.get("num_devices",1))
            dispatcher = GraphDispatcher(config.get("num_devices",1), daisy_chain_variables=config.get("daisy_chain_variables",False), devices= config.get("devices",None))             
            batch_multiplier = config.get("num_devices", 1)
            num_threads = config.get("num_threads", 4)            
            iterator = load_data(config["training_label_file"], src_vocab, config["training_tag_file"], batch_size = train_batch_size, batch_type=config["training_batch_type"], batch_multiplier = batch_multiplier, tgt_path=config["training_feature_file"], tgt_vocab=tgt_vocab, max_len = max_len, mode=mode, shuffle_buffer_size = config["sample_buffer_size"], num_threads = num_threads, version = load_data_version, distribution = example_sampling_distribution)
            inputs = iterator.get_next()            
            data_shards = dispatcher.shard(inputs)

            with tf.variable_scope(config["Architecture"], initializer=self._initializer(config)):
                losses_shards = dispatcher(_loss_op, data_shards, config, mode)

            self.loss = _extract_loss(losses_shards, Loss_type=Loss_type) 

        elif mode == "Inference": 
            assert test_feature_file != None
            if config.get("multi_gpu_inference",False):
                dispatcher = GraphDispatcher(config.get("num_devices_eval",1), daisy_chain_variables=config.get("daisy_chain_variables",False), devices= config.get("devices",None))            
                batch_multiplier = config.get("num_devices_eval", 1)
                num_threads = config.get("num_threads", 4)
                iterator = load_data(test_feature_file, src_vocab, test_tag_file, batch_size = eval_batch_size, batch_type = "examples", batch_multiplier = batch_multiplier, max_len = max_len, mode = mode, version = load_data_version)
                inputs = iterator.get_next()
                data_shards = dispatcher.shard(inputs)
                with tf.variable_scope(config["Architecture"]):
                    prediction_shards = dispatcher(_loss_op, data_shards, config, mode)           
                self.predictions = prediction_shards
            else:
                iterator = load_data(test_feature_file, src_vocab, test_tag_file, batch_size = eval_batch_size, batch_type = "examples", batch_multiplier = 1, max_len = max_len, mode = mode, version = load_data_version)
                inputs = iterator.get_next()
                with tf.variable_scope(config["Architecture"]):
                    _ , self.predictions, _, _ = self._build(inputs, config, mode)

        elif mode == "new_domain":    
            print("num_devices", config.get("num_devices",1))
            dispatcher = GraphDispatcher(config.get("num_devices",1), daisy_chain_variables=config.get("daisy_chain_variables",False), devices= config.get("devices",None))
            batch_multiplier = config.get("num_devices", 1)
            num_threads = config.get("num_threads", 4)
            iterator = load_data(config["training_label_file"], src_vocab, config["training_tag_file"], batch_size = train_batch_size, batch_type=config["training_batch_type"], batch_multiplier = batch_multiplier, tgt_path=config["training_feature_file"], tgt_vocab=tgt_vocab, max_len = max_len, mode=mode, shuffle_buffer_size = config["sample_buffer_size"], num_threads = num_threads, version = load_data_version, distribution = example_sampling_distribution)
            inputs = iterator.get_next()
            data_shards = dispatcher.shard(inputs)
            with tf.variable_scope(config["Architecture"], initializer=self._initializer(config)):
                losses_shards = dispatcher(_loss_op, data_shards, config, mode)

            self.loss = _extract_loss(losses_shards, Loss_type=Loss_type)
            
        self.iterator = iterator
        self.inputs = inputs

    def loss_(self):
        return self.loss
    
    def prediction_(self):
        return self.predictions
        
    def inputs_(self):
        return self.inputs
    
    def Wasserstein_batch_info(self):
        if self.Loss_type == "Wasserstein":
            return self.cost_sample_batch, self.cost_batch, self.tf_idf_batch, self.tf_idf_sample_batch, self.sample_log_prob
        else:
            return None, None, None, None, None

    def iterator_initializers(self):
        if isinstance(self.iterator,list):
            return [iterator.initializer for iterator in self.iterator]
        else:
            return [self.iterator.initializer]        
           
    def _build(self, inputs, config, mode):        

        debugging = config.get("debugging", False)
        projector_masking = config.get("projector_masking", False)
        src_masking = config.get("src_masking", True)        
        tgt_masking = config.get("tgt_masking", False)
        adv_training = config.get("Generic_region_adversarial_training", False)
        self.Domain_forcing = config.get("Domain_forcing",False)
        src_adv_training = config.get("src_adv_training", True)
        tgt_adv_training = config.get("tgt_adv_training", True)
        Loss_type = self.Loss_type
        self.adv_training = adv_training
        self.src_adv_training = src_adv_training
        self.tgt_adv_training = tgt_adv_training
        self.Adv_var = []
        print("projector_masking", projector_masking)
        print("src_masking", src_masking)
        print("tgt_masking", tgt_masking)
        print("Adv_training", adv_training)        
        print("Loss_type: ", Loss_type)
        src_size_in_common = config["src_sharing_embedding_region_size"]
        src_size_in_domain = config["src_domain_embedding_region_size"]
        tgt_size_in_common = config.get("tgt_sharing_embedding_region_size", config["src_sharing_embedding_region_size"])
        tgt_size_in_domain = config.get("tgt_domain_embedding_region_size", config["src_domain_embedding_region_size"])
                        
        if tgt_masking:
            size_tgt = tgt_size_in_common #+ sum(tgt_size_in_domain)
            print("tgt_masking")
            print("region sizes: ", [tgt_size_in_common] + tgt_size_in_domain)
        else:
            size_tgt = config.get("tgt_embedding_size",512)
            print("tgt_size: ", size_tgt)

        if src_masking:
            size_src = src_size_in_common #+ sum(src_size_in_domain)
            print("src_masking")
            print("region sizes: ", [src_size_in_common] + src_size_in_domain)
        else:
            size_src = config.get("src_embedding_size",512)
            print("src_size: ", size_src)              

        hidden_size = config["hidden_size"]       
        print("hidden size: ", hidden_size)
                
        tgt_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(config["tgt_vocab_path"], vocab_size= int(config["tgt_vocab_size"]) - 1, default_value=constants.UNKNOWN_TOKEN)
        end_token = constants.END_OF_SENTENCE_ID

        # Embedding        
        assert src_masking
        with tf.variable_scope("src_generic_embedding"):
            src_emb_generic = create_embeddings(config["src_vocab_size"], depth=src_size_in_common)
        
        with tf.variable_scope("src_domain_embedding", initializer = tf.zeros_initializer): 
            src_emb_domain = create_embeddings(config["src_vocab_size"], depth=sum(src_size_in_domain))

        self.src_emb_domain = src_emb_domain
        self.src_emb_generic = src_emb_generic
        
        with tf.variable_scope("tgt_generic_embedding"):
            tgt_emb_generic = create_embeddings(config["tgt_vocab_size"], depth=tgt_size_in_common)
        
        with tf.variable_scope("tgt_domain_embedding", initializer = tf.zeros_initializer): 
            tgt_emb_domain = create_embeddings(config["tgt_vocab_size"], depth=sum(tgt_size_in_domain))

        """
        with tf.variable_scope("tgt_embedding"):
            tgt_emb = create_embeddings(config["tgt_vocab_size"], depth=size_tgt)
        """

        with tf.variable_scope("src_mask"):
            src_mask_ = create_mask(config["domain_numb"], src_size_in_domain)
            #src_mask_ = tf.Print(src_mask_, [src_mask_[:,:32]],message="src_mask: ", first_n=3, summarize=100)

        with tf.variable_scope("tgt_mask"):
            tgt_mask_ = create_mask(config["domain_numb"], tgt_size_in_domain)
        
        # Build encoder, decoder
        if config["Architecture"] == "GRU":
            nlayers = config.get("nlayers",4)
            encoder = onmt.encoders.BidirectionalRNNEncoder(nlayers, hidden_size, reducer=onmt.layers.ConcatReducer(), cell_class = tf.contrib.rnn.GRUCell, dropout=0.1, residual_connections=False)
            decoder = onmt.decoders.AttentionalRNNDecoder(nlayers, hidden_size, bridge=onmt.layers.CopyBridge(), cell_class=tf.contrib.rnn.GRUCell, dropout=0.1, residual_connections=False)

        elif config["Architecture"] == "LSTM":
            nlayers = config.get("nlayers",4)
            encoder = onmt.encoders.BidirectionalRNNEncoder(nlayers, num_units=hidden_size, reducer=onmt.layers.ConcatReducer(), cell_class=tf.nn.rnn_cell.LSTMCell,
                                                          dropout=0.1, residual_connections=False)
            #decoder = onmt.decoders.AttentionalRNNDecoder(nlayers, num_units=hidden_size, bridge=onmt.layers.CopyBridge(), attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
            #                                             cell_class=tf.nn.rnn_cell.LSTMCell, dropout=0.1, residual_connections=False)

            decoder = rnn_decoder.AttentionalRNNDecoder_v2(nlayers, num_units=int(hidden_size/2), bridge=onmt.layers.DenseBridge(activation=tf.math.tanh),
                                                    attention_mechanism_class=tf.contrib.seq2seq.BahdanauAttention,
                                                    cell_class=tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell, dropout=0.1, residual_connections=False)

        elif config["Architecture"] == "Transformer":
            nlayers = config.get("nlayers",6)
            decoder = self_attention_decoder_LDR.SelfAttentionDecoder(nlayers, num_units=hidden_size, num_heads=8, ffn_inner_dim=2048, dropout=0.1, attention_dropout=0.1, relu_dropout=0.1)
            encoder = self_attention_encoder_LDR.SelfAttentionEncoder(nlayers, num_units=hidden_size, num_heads=8, ffn_inner_dim=2048, dropout=0.1, attention_dropout=0.1, relu_dropout=0.1)

        print("Model type: ", config["Architecture"])

        if mode =="Training":            
            print("Building model in Training mode")
        elif mode == "Inference":
            print("Build model in Inference mode")

        start_tokens = tf.fill([tf.shape(inputs["src_ids"])[0]], constants.START_OF_SENTENCE_ID)   
        generic_domain = tf.fill(tf.shape(inputs["domain"]), config["domain_numb"])
        emb_src_batch_domain = make_batch(src_emb_domain, src_emb_generic, src_mask_, inputs["domain"], inputs["src_ids"])
        emb_src_batch_generic = make_batch(src_emb_domain, src_emb_generic, src_mask_, generic_domain, inputs["src_ids"])                     

        if mode=="Training":
            emb_tgt_batch_domain = make_batch(tgt_emb_domain, tgt_emb_generic, tgt_mask_, inputs["domain"], inputs["tgt_ids"])
            emb_tgt_batch_generic = make_batch(tgt_emb_domain, tgt_emb_generic, tgt_mask_, generic_domain, inputs["tgt_ids"])
            #emb_tgt_batch = tf.nn.embedding_lookup(tgt_emb, inputs["tgt_ids_in"])    
     
        #output_layer = build_output_layer(hidden_size, config["tgt_vocab_size"])
        output_layer = None
        if config["Architecture"] != "Transformer":
            output_layer = build_output_layer(int(hidden_size)+int(size_tgt), config["tgt_vocab_size"])
        else:
            output_layer = build_output_layer(hidden_size, config["tgt_vocab_size"])

        src_length = inputs["src_length"]
        if mode =="Training":
            tgt_ids_batch = inputs["tgt_ids_out"]
            tgt_length = inputs["tgt_length"]
            
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            if config.get("Standard",True):
                if mode=="Training":
                    encoder_output_generic = encoder.encode(emb_src_batch_generic, sequence_length = src_length, mode=tf.estimator.ModeKeys.TRAIN)
                    encoder_output_domain = encoder.encode(emb_src_batch_domain, sequence_length = src_length, mode=tf.estimator.ModeKeys.TRAIN)
                else:
                    encoder_output = encoder.encode(emb_src_batch_domain, sequence_length = src_length, mode=tf.estimator.ModeKeys.PREDICT)           
                    
        if mode == "Training":    
            if Loss_type == "Cross_Entropy":
                with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):                           
                    if config.get("Standard",True):                        
                        logits_generic, _, _ = decoder.decode(
                                              emb_tgt_batch_generic,
                                              tgt_length + 1,
                                              vocab_size = int(config["tgt_vocab_size"]),
                                              initial_state = encoder_output_generic[1],
                                              output_layer = output_layer,
                                              mode = tf.estimator.ModeKeys.TRAIN,
                                              memory = encoder_output_generic[0],
                                              memory_sequence_length = encoder_output_generic[2],
                                              return_alignment_history = False)
                        logits_domain, _, _ = decoder.decode(
                                              emb_tgt_batch_domain,
                                              tgt_length + 1,
                                              vocab_size = int(config["tgt_vocab_size"]),
                                              initial_state = encoder_output_domain[1],
                                              output_layer = output_layer,
                                              mode = tf.estimator.ModeKeys.TRAIN,
                                              memory = encoder_output_domain[0],
                                              memory_sequence_length = encoder_output_domain[2],
                                              return_alignment_history = False)
                    outputs = {
                           "logits": (logits_generic, logits_domain)
                           }
       
        else:
            outputs = None            

        if mode != "Training":  
                
            with tf.variable_scope("decoder"):        
                beam_width = config.get("beam_width", 5)
                print("Inference with beam width %d"%(beam_width))
                maximum_iterations = config.get("maximum_iterations", 250)
                if beam_width <= 1:
                    if config.get("Standard",True):
                        sampled_ids, _, sampled_length, log_probs, alignment = decoder.dynamic_decode(
                                                                                    lambda id: tf.concat([tf.nn.embedding_lookup(tgt_emb_generic, id), tf.multiply(tf.nn.embedding_lookup(tgt_emb_domain, id), tf.nn.embedding_lookup(tgt_mask_, id))],-1),
                                                                                    start_tokens,
                                                                                    end_token,
                                                                                    vocab_size=int(config["tgt_vocab_size"]),
                                                                                    initial_state=encoder_output[1],
                                                                                    maximum_iterations=maximum_iterations,
                                                                                    output_layer = output_layer,
                                                                                    mode=tf.estimator.ModeKeys.PREDICT,
                                                                                    memory=encoder_output[0],
                                                                                    memory_sequence_length=encoder_output[2],
                                                                                    dtype=tf.float32,
                                                                                    return_alignment_history=True)
                else:
                    length_penalty = config.get("length_penalty", 0)
                    if config.get("Standard",True):
                        sampled_ids, _, sampled_length, log_probs, alignment = decoder.dynamic_decode_and_search(
                                                          lambda id: tf.concat([tf.nn.embedding_lookup(tgt_emb_generic, id), tf.multiply(tf.nn.embedding_lookup(tgt_emb_domain, id), tf.nn.embedding_lookup(tgt_mask_, id))],-1),
                                                          start_tokens,
                                                          end_token,
                                                          vocab_size = int(config["tgt_vocab_size"]),
                                                          initial_state = encoder_output[1],
                                                          beam_width = beam_width,
                                                          length_penalty = length_penalty,
                                                          maximum_iterations = maximum_iterations,
                                                          output_layer = output_layer,
                                                          mode = tf.estimator.ModeKeys.PREDICT,
                                                          memory = encoder_output[0],
                                                          memory_sequence_length = encoder_output[2],           
                                                          dtype=tf.float32,                                               
                                                          return_alignment_history = True)
        
            target_tokens = tgt_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))
            
            predictions = {
              "tokens": target_tokens,
              "length": sampled_length,
              "log_probs": log_probs,
              "alignment": alignment,
            }
            tgt_ids_batch = None
            tgt_length = None
        else:
            predictions = None

        self.outputs = outputs
        
        return outputs, predictions, tgt_ids_batch, tgt_length               
        
        
        
        
    
