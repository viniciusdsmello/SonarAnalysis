#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This file contains the Novelty Detection Analysis with Stacked AutoEncoders
    Author: Vinicius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
"""
import multiprocessing
import os
import time
from datetime import timedelta

import tensorflow as tf
from keras.utils import np_utils
from sklearn import preprocessing

from Functions.telegrambot import Bot
from Packages.NoveltyDetection.NoveltyDetectionAnalysis import NoveltyDetectionAnalysis
from .models.vae import VariationalAutoencoder

my_bot = Bot("lisa_thebot")

num_processes = multiprocessing.cpu_count()


class VAENoveltyDetectionAnalysis(NoveltyDetectionAnalysis):

    def __init__(self, parameters=None, model_hash=None, load_hash=False, load_data=True, verbose=False):
        super().__init__(parameters=parameters, model_hash=model_hash, load_hash=load_hash, load_data=load_data,
                         verbose=verbose)

        self.trn_data = {}
        self.trn_trgt = {}
        self.trn_trgt_sparse = {}
        self.vae_models = {}

    def emulate_novelties(self):
        # Divide the dataset to do a novelty class emulation
        for inovelty in range(self.all_trgt_sparse.shape[1]):
            self.trn_data[inovelty] = self.all_data[self.all_trgt != inovelty]
            self.trn_trgt[inovelty] = self.all_trgt[self.all_trgt != inovelty]
            self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] = self.trn_trgt[inovelty][
                                                                              self.trn_trgt[inovelty] > inovelty] - 1
            self.trn_trgt_sparse[inovelty] = np_utils.to_categorical(self.trn_trgt[inovelty].astype(int))

    def build_vae_models(self):
        for inovelty in range(self.all_trgt_sparse.shape[1]):
            # Initialize VAE objects for all novelties
            original_dim = self.all_data.shape[1]
            intermediate_dim = int(self.parameters['HyperParameters']['IntermediateDimension'])
            latent_dim = 2
            self.vae_models[inovelty] = VariationalAutoencoder(original_dim, intermediate_dim, latent_dim)

        return self.vae_models

    def get_vae_model(self, inovelty):
        return self.vae_models[inovelty]

    def get_normalized_data(self, data, ifold, inovelty):
        train_id, test_id = self.CVO[inovelty][ifold]

        # Fit the scaler based on the training data
        if self.parameters["HyperParameters"]["norm"] == 'mapstd':
            scaler = preprocessing.StandardScaler().fit(self.trn_data[inovelty][train_id, :])
        elif self.parameters["HyperParameters"]["norm"] == 'mapstd_rob':
            scaler = preprocessing.RobustScaler().fit(self.trn_data[train_id, :])
        elif self.parameters["HyperParameters"]["norm"] == 'mapminmax':
            scaler = preprocessing.MinMaxScaler().fit(self.trn_data[train_id, :])
        else:
            return self.trn_data[inovelty]
        norm_data = scaler.transform(data)

        return norm_data

    def train(self, inovelty, ifold):
        model = self.get_vae_model(inovelty)

        train_id, test_id = self.CVO[inovelty][ifold]
        norm_train_data = self.get_normalized_data(self.trn_data[inovelty][train_id], ifold)
        norm_test_data = self.get_normalized_data(self.trn_data[inovelty][test_id], ifold)

        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.parameters["callbacks"]["EarlyStopping"]["monitor"],
            patience=self.parameters["callbacks"]["EarlyStopping"]["patience"],
            verbose=self.verbose,
            mode='auto')

        csv_log_filename = f'pretraining.log'
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(self.getBaseResultsPath(), 'Logs', csv_log_filename))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.parameters["HyperParameters"]["OptimizerAlgorithm"])
        model.compile(loss=tf.keras.losses.mse,
                      optimizer=optimizer,
                      metrics=[tf.keras.losses.mse, tf.keras.losses.mae])
        train_start_time = time.time()
        init_trn_desc = model.fit(norm_train_data,
                                  norm_train_data,
                                  epochs=int(self.parameters["HyperParameters"]["pretraining_n_epochs"]),
                                  # PRE-TRAINING
                                  batch_size=self.parameters["HyperParameters"]["batch_size"],
                                  callbacks=[early_stopping, csv_logger],
                                  verbose=self.verbose,
                                  validation_data=(norm_test_data,
                                                   norm_test_data),
                                  shuffle=True
                                  )
        duration = str(timedelta(seconds=float(time.time() - train_start_time)))
        self.__notify_training_conclusion(duration, 'Pre-Training', inovelty)

    def __notify_training_conclusion(self, duration, training_type, inovelty):
        message = "Technique: {}\n".format(self.parameters['Technique'])
        message += "Development Mode: {}\n".format(self.parameters['DevelopmentMode'])
        message += "Training Type: {}\n".format(training_type)
        message += "Novelty Class: {}\n".format(self.class_labels[inovelty])
        message += "Hash: {}\n".format(self.model_hash)
        message += "Duration: {}\n".format(duration)
        try:
            my_bot.sendMessage(message)
        except Exception as e:
            print("Erro ao enviar mensagem. Erro: " + str(e))
