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
from model import *
import os
import yaml
import io
from tensorflow.python.framework import ops
import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", required=True , help="configuration file")
args = parser.parse_args()

config_file = args.config_file
with open(config_file, "r") as stream:
    config = yaml.load(stream)
# Eval directory stores prediction files
if not os.path.exists(os.path.join(config["model_dir"],"eval")):
    os.makedirs(os.path.join(config["model_dir"],"eval"))

training_model = Model(config_file, "Training")
global_step = tf.train.create_global_step()

if config.get("Loss_Function","Cross_Entropy")=="Cross_Entropy":
    loss_generic, loss_domain = training_model.loss_()

if config["mode"] == "Training":
    if config.get("Loss_Function","Cross_Entropy")=="Cross_Entropy":
        if config["generic_domain_training"]:
            print("generic_domain_training")
            with tf.variable_scope("domain_optim"):
                domain_op, domain_accum_vars_ = optimize_loss(loss_domain, config["optimizer_parameters"], var_list=[v for v in tf.trainable_variables() if "ldr" in v.name], update_global_step=False)
            with tf.variable_scope("generic_optim"):
                gen_op, gen_accum_vars_ = optimize_loss(loss_generic, config["optimizer_parameters"], var_list=[v for v in tf.trainable_variables() if "ldr" not in v.name])
            train_op = [domain_op, gen_op]
            accum_vars_ = domain_accum_vars_ + gen_accum_vars_

        elif config["generic+domain_training"]:
            print("generic+domain_training")
            inputs = training_model.inputs_()
            domain = inputs["domain"][0]
            data_sizes = [get_dataset_size(path) for path in config["training_feature_file"]]
            total_size = sum(data_sizes)
            domain_weight = tf.constant([float(s)/(total_size) for s in data_sizes])
            if config.get("domain_weighted_loss", False):
                train_op, accum_vars_ = optimize_loss(loss_domain + loss_generic * domain_weight[domain], config["optimizer_parameters"])
            else:
                train_op, accum_vars_ = optimize_loss(loss_domain * 0.5 + loss_generic * 0.5, config["optimizer_parameters"])

        elif config["generic_training"]:
            print("generic_training")
            train_op, accum_vars_ = optimize_loss(loss_generic, config["optimizer_parameters"])

        elif config["domain_training"]:
            print("domain_training")
            train_op, accum_vars_ = optimize_loss(loss_domain, config["optimizer_parameters"])
        
elif config["mode"] == "continual":
    train_op, accum_vars_ = optimize_loss(loss_domain, config["optimizer_parameters"], var_list=[v for v in tf.trainable_variables() if "ldr" in v.name]) #optimize_loss((loss_domain + loss_generic)/2, config["optimizer_parameters"]) #optimize_loss(loss_domain, config["optimizer_parameters"], var_list=[v for v in tf.trainable_variables() if "domain" in v.name])        

Eval_dataset_numb = len(config["eval_label_file"])
print("Number of validation set: ", Eval_dataset_numb)
external_evaluator = [None] * Eval_dataset_numb 
writer_bleu = [None] * Eval_dataset_numb 

for i in range(Eval_dataset_numb):
    external_evaluator[i] = BLEUEvaluator(config["eval_label_file"][i], config["model_dir"])
    writer_bleu[i] = tf.summary.FileWriter(os.path.join(config["model_dir"],"BLEU","domain_%d"%i))

with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    
    writer = tf.summary.FileWriter(config["model_dir"])        
    var_list_ = tf.global_variables()
    for v in tf.trainable_variables():
        if v not in tf.global_variables():
            var_list_.append(v)
    for v in var_list_:
        print(v)

    saver = tf.train.Saver(var_list_, max_to_keep=config["max_to_keep"])
    checkpoint_path = tf.train.latest_checkpoint(config["model_dir"])
        
    sess.run([v.initializer for v in var_list_])    
    sess.run([v.initializer for v in accum_vars_])
    training_summary = tf.summary.merge_all()
    global_step_ = sess.run(global_step) 

    if checkpoint_path:
        print("Continue training:...")
        print("Load parameters from %s"%checkpoint_path)
        saver.restore(sess, checkpoint_path)        
        global_step_ = sess.run(global_step)
        print("global_step: ", global_step_)
                    
        for i in range(Eval_dataset_numb):
            prediction_file = inference(config_file, checkpoint_path, config["eval_feature_file"][i], config["eval_tag_file"][i])
            score = external_evaluator[i].score(config["eval_label_file"][i], prediction_file)
            print("BLEU at checkpoint %s for testset %s: %f"%(checkpoint_path, config["eval_feature_file"][i], score))            
        
    elif config["mode"] == "continual":
        print("New domain continual training:...")
        checkpoint_path_ = config.get("checkpoint_path")
        print("Load parameters from %s"%checkpoint_path_)
        if checkpoint_path_:
            saver.restore(sess, checkpoint_path_)
            global_step_ = sess.run(global_step)
            print("global_step: ", global_step_)      
            if config.get("Generic_region_adversarial_training", False):
                print("lambda_e: ", sess.run(lambda_E))

            for i in range(Eval_dataset_numb):
                prediction_file = inference(config_file, checkpoint_path_, config["eval_feature_file"][i], config["eval_tag_file"][i])
                score = external_evaluator[i].score(config["eval_label_file"][i], prediction_file)
                print("BLEU at checkpoint %s for testset %s: %f"%(checkpoint_path_, config["eval_feature_file"][i], score))   
        else:
            raise Exception('x should not exceed 5. The value of x was: {}'.format(x))    
    else:
        print("Training from scratch")

    tf.tables_initializer().run()    
    sess.run(training_model.iterator_initializers())
    total_loss_generic = []            
    total_loss_domain = []
    src_lengths = []
    tgt_lengths = []
    #loop_count = 0
    
    while global_step_ <= config["iteration_number"]:                       
        #loop_count +=1
        if isinstance(train_op,list):
            loss_domain_, loss_generic_, global_step_, _, _ = sess.run([loss_domain, loss_generic, global_step]+train_op)        
        else:
            loss_domain_, loss_generic_, global_step_, _ = sess.run([loss_domain, loss_generic, global_step, train_op])

        total_loss_generic.append(loss_generic_)
        total_loss_domain.append(loss_domain_)
        
        if (np.mod(global_step_, config["printing_freq"])) == 0:            
            print((datetime.datetime.now()))
            print(("Loss generic at step %d"%(global_step_), np.mean(total_loss_generic)))                
            print(("Loss domain at step %d"%(global_step_), np.mean(total_loss_domain)))
            total_loss_generic = []
            total_loss_domain = []
  
        if (np.mod(global_step_, config["summary_freq"])) == 0:
            training_summary_ = sess.run(training_summary)
            writer.add_summary(training_summary_, global_step=global_step_)
            writer.flush()
                     
        if (np.mod(global_step_, config["save_freq"])) == 0 and global_step_ > 0:    
            print((datetime.datetime.now()))
            checkpoint_path = os.path.join(config["model_dir"], 'model.ckpt')
            print(("save to %s"%(checkpoint_path)))
            saver.save(sess, checkpoint_path, global_step = global_step_)
                                                                                                                 
        if (np.mod(global_step_, config["eval_freq"])) == 0 and global_step_ >0: 
            checkpoint_path = tf.train.latest_checkpoint(config["model_dir"])
            for i in range(Eval_dataset_numb):
                prediction_file = inference(config_file, checkpoint_path, config["eval_feature_file"][i], config["eval_tag_file"][i])
                score = external_evaluator[i].score(config["eval_label_file"][i], prediction_file)
                print("BLEU at checkpoint %s for testset %s: %f"%(checkpoint_path, config["eval_feature_file"][i], score))
                score_summary = tf.Summary(value=[tf.Summary.Value(tag="eval_score_%d"%i, simple_value=score)])
                writer_bleu[i].add_summary(score_summary, global_step_)
                writer_bleu[i].flush()
        


