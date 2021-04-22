import argparse
import logging
import multiprocessing
import os
import pprint
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from Functions.telegrambot import Bot
from Packages.NoveltyDetection.setup.noveltyDetectionConfig import CONFIG
from Packages.NoveltyDetection.VariationalAutoencoders.VAENoveltyDetectionAnalysis import \
    VAENoveltyDetectionAnalysis

# Argument Parser config
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--gpu_id", default=1, type=int, help="GPU device ID")

args = parser.parse_args()
gpu_id = args.gpu_id


tf.get_logger().setLevel(logging.ERROR)

tf.debugging.set_log_device_placement(False)

gpus = tf.config.experimental.list_physical_devices('GPU')

print("Num GPUs Available: ", len(gpus))

if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_id], f'GPU')

num_processes = multiprocessing.cpu_count()

my_bot = Bot("lisa_thebot")

# Enviroment variables
data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']


def main(training_params: dict):
    analysis = VAENoveltyDetectionAnalysis(parameters=training_params,
                                           load_hash=False, load_data=True, verbose=True)
    all_data, all_trgt, trgt_sparse = analysis.getData()

    analysis.emulate_novelties()
    analysis.build_vae_models()

    trn_data = analysis.trn_data
    trn_trgt = analysis.trn_trgt
    trn_trgt_sparse = analysis.trn_trgt_sparse

    pp = pprint.PrettyPrinter(indent=1)
    print(analysis.model_hash)
    print(analysis.getBaseResultsPath())
    pp.pprint(analysis.parameters)

    analysis.train_all()

    return analysis.model_hash


if __name__ == '__main__':

    experiments = [
        {"intermediate_dim": [256, 128], "latent_dim": 64},
        {"intermediate_dim": [256, 128, 64], "latent_dim": 32},
        {"intermediate_dim": [384, 256, 128], "latent_dim": 64},
        {"intermediate_dim": [384, 256, 128], "latent_dim": 64},
        {"intermediate_dim": [384, 256, 128, 64], "latent_dim": 32}
    ]

    hashes = []

    for experiment in experiments:
        intermediate_dim = experiment.get("intermediate_dim")
        latent_dim = experiment.get("latent_dim")

        training_params = {
            "Technique": "VariationalAutoencoder",
            "TechniqueParameters": {
                "IntermediateDimension": intermediate_dim,
                "LatentDimension": latent_dim
            },
            "DevelopmentMode": False,
            "DevelopmentEvents": 1600,
            "NoveltyDetection": True,
            "InputDataConfig": {
                "database": "4classes",
                "n_pts_fft": 1024,
                "decimation_rate": 3,
                "spectrum_bins_left": 400,
                "n_windows": 1,
                "balance_data": False
            },
            "OptmizerAlgorithm": {
                "name": "Adam",
                "parameters": {
                    "learning_rate": 0.001,
                    "beta_1": 0.90,
                    "beta_2": 0.999,
                    "epsilon": 1e-08,
                    "learning_decay": 1e-6,
                    "momentum": 0.3,
                    "nesterov": True
                }
            },
            "HyperParameters": {
                "n_folds": 4,
                "pretraining_n_epochs": 500,
                "finetuning_n_epochs": 300,
                "n_inits": 2,
                "batch_size": 64,
                "kernel_initializer": "uniform",
                "bias_initializer": "ones",
                "encoder_activation_function": "relu",
                "decoder_activation_function": "relu",
                "norm": "mapstd",
                "metrics": ["accuracy"],
                "loss": "mean_squared_error",
                "classifier_loss": "mean_squared_error"
            },
            "callbacks": {
                "EarlyStopping": {
                    "patience": 50,
                    "monitor": "val_total_loss"
                }
            }
        }

        model_hash = main(training_params)
        hashes.append(model_hash)

    print("*" * 50)

    print(hashes)

