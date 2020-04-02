import tensorflow as tf
import opennmt as onmt
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
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", required=True , help="configuration file")
parser.add_argument("--eval_step", required=True , help="evaluation step")
args = parser.parse_args()

config_file = args.config_file
with open(config_file, "r") as stream:
    config = yaml.load(stream)

test_feature_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/emea/EMEA.en-fr.en.bpe.tok.dev","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/epps/europarl-v7.fr-en.en.bpe.tok.dev","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/ecb/ECB.en-fr.en.dev.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/ted/TED2013.en-fr.en.bpe.tok.dev","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/IT/IT.en-fr.en.bpe.tok.dev","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/quoran/Tanzil.en-fr.en.bpe.tok.dev"]
test_label_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/emea/EMEA.en-fr.fr.bpe.tok.dev","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/epps/europarl-v7.fr-en.fr.bpe.tok.dev","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/ecb/ECB.en-fr.fr.dev.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/ted/TED2013.en-fr.fr.bpe.tok.dev","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/IT/IT.en-fr.fr.bpe.tok.dev","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/quoran/Tanzil.en-fr.fr.bpe.tok.dev"]
test_tag_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/emea/EMEA.en-fr.en.bpe.tok.dev.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/epps/europarl-v7.fr-en.en.bpe.tok.dev.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/ecb/ECB.en-fr.en.dev.bpe.tok.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/ted/TED2013.en-fr.en.bpe.tok.dev.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/IT/IT.en-fr.en.bpe.tok.dev.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/quoran/Tanzil.en-fr.en.bpe.tok.dev.tag.1"]

#test_tag_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/emea/EMEA.en-fr.en.bpe.tok.dev.tag.0","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/epps/europarl-v7.fr-en.en.bpe.tok.dev.tag.0","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/ecb/ECB.en-fr.en.dev.bpe.tok.tag.0","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/ted/TED2013.en-fr.en.bpe.tok.dev.tag.0","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/IT/IT.en-fr.en.bpe.tok.dev.tag.0","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/valid_corpora/quoran/Tanzil.en-fr.en.bpe.tok.dev.tag.0"]

"""
test_feature_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/"]
test_label_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/"]
test_tag_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/"]
"""

"""
test_feature_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/EMEA.en-fr.en.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/europarl-v7.fr-en.en.bpe.tok.tst.o","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ted/IWSLT16.TED.tst2010.en-fr.en.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/IT/IT.en-fr.en.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/quoran/Tanzil.en-fr.en.bpe.tok.tst"]

test_label_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/EMEA.en-fr.fr.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/europarl-v7.fr-en.fr.bpe.tok.tst.o","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.fr.test.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ted/IWSLT16.TED.tst2010.en-fr.fr.bpe.tok","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/IT/IT.en-fr.fr.bpe.tok.tst","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/quoran/Tanzil.en-fr.fr.bpe.tok.tst"]

test_tag_files = ["/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/emea/EMEA.en-fr.en.bpe.tok.tst.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/epps/europarl-v7.fr-en.en.bpe.tok.tst.o.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ecb/ECB.en-fr.en.test.bpe.tok.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/ted/IWSLT16.TED.tst2010.en-fr.en.bpe.tok.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/IT/IT.en-fr.en.bpe.tok.tst.tag.1","/mnt/beegfs/home/pham/multi-domain-nmt/sparse/data/test_corpora/quoran/Tanzil.en-fr.en.bpe.tok.tst.tag.1"]

"""
print("number of testsets: ", len(test_label_files))
external_evaluator = [None] * len(test_feature_files)

for i in range(len(test_label_files)):
    external_evaluator[i] = BLEUEvaluator(test_label_files[i], config["model_dir"])

with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    for i in range(len(test_feature_files)):
        checkpoint_path = os.path.join(config["model_dir"], "model.ckpt-%d"%int(args.eval_step))
        prediction_file = inference(config_file, checkpoint_path, test_feature_files[i], test_tag_files[i])
        score = external_evaluator[i].score(test_label_files[i], prediction_file)
        print("BLEU at checkpoint %s for testset %s: %f"%(checkpoint_path, test_label_files[i], score),flush=True)
