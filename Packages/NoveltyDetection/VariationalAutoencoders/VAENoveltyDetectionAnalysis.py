#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This file contains the Novelty Detection Analysis with Variational Autoencoders
    Author: Vinicius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
"""
import multiprocessing
import os
import time
from datetime import timedelta

import tensorflow as tf
from sklearn import preprocessing

from Functions.telegrambot import Bot
from Packages.NoveltyDetection.NoveltyDetectionAnalysis import NoveltyDetectionAnalysis
from .models.vae import VariationalAutoencoder

my_bot = Bot("lisa_thebot")

num_processes = multiprocessing.cpu_count()

class VAENoveltyDetectionAnalysis(NoveltyDetectionAnalysis):

    def __init__(self, parameters=None, model_hash=None, load_hash=False, load_data=True, verbose=False):
        super().__init__(parameters=parameters, model_hash=model_hash, load_hash=load_hash, load_data=load_data,
                         verbose=verbose, logger_level='DEBUG')

        self.trn_data = {}
        self.trn_trgt = {}
        self.trn_trgt_sparse = {}
        self.vae_models = {}

    def emulate_novelties(self):
        # Divide the dataset to do a novelty class emulation
        self.logger.info('Emulating novelty classes...')
        for inovelty in range(self.all_trgt_sparse.shape[1]):
            self.logger.debug(f'Spliting data for novelty class #{self.getClassLabels()[inovelty]}...')
            self.trn_data[inovelty] = self.all_data[self.all_trgt != inovelty]
            self.trn_trgt[inovelty] = self.all_trgt[self.all_trgt != inovelty]
            self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] = self.trn_trgt[inovelty][
                                                                              self.trn_trgt[inovelty] > inovelty] - 1
            self.trn_trgt_sparse[inovelty] = tf.keras.np_utils.to_categorical(self.trn_trgt[inovelty].astype(int))

    def build_vae_models(self):
        for inovelty, novelty_class in enumerate(self.getClassLabels()):
            self.logger.debug(f'Building VAE model for novelty class #{novelty_class}...')
            # Initialize VAE objects for all novelties
            original_dim = self.all_data.shape[1]
            intermediate_dim = int(self.parameters['HyperParameters']['IntermediateDimension'])
            latent_dim = 2
            model = VariationalAutoencoder(original_dim, intermediate_dim, latent_dim)
            self.vae_models[inovelty] = []
            for ifold in range(self.n_folds):
                self.vae_models[inovelty][ifold] = model

        return self.vae_models

    def get_vae_model(self, inovelty, ifold):
        return self.vae_models[inovelty][ifold]

    def update_vae_model(self, inovelty, ifold, model):
        self.vae_models[inovelty][ifold] = model

    def get_normalized_data(self, data, ifold, inovelty):
        train_id, _ = self.CVO[inovelty][ifold]

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

    def train_all(self, method='sequential'):
        if method == 'sequential':
            for inovelty, _ in enumerate(self.getClassLabels()):
                for ifold, _ in enumerate(self.CVO[inovelty]):
                    self.train(inovelty, ifold)

    def train(self, inovelty, ifold):
        self.logger.info(f'Called train() for novelty class {self.getClassLabels()[inovelty]} and Fold {ifold}')
        train_id, test_id = self.CVO[inovelty][ifold]
        norm_train_data = self.get_normalized_data(self.trn_data[inovelty][train_id], ifold, inovelty)
        norm_test_data = self.get_normalized_data(self.trn_data[inovelty][test_id], ifold, inovelty)

        best_init = 0
        best_loss = 999

        best_model = []
        trn_desc = {}

        for i_init in range(self.parameters["HyperParameters"]["n_inits"]):
            model = self.get_vae_model(inovelty, ifold)
            self.logger.info(f'Running {i_init+1} initialization...')
            # Callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=self.parameters["callbacks"]["EarlyStopping"]["monitor"],
                patience=self.parameters["callbacks"]["EarlyStopping"]["patience"],
                verbose=self.verbose,
                mode='auto')

            csv_log_filename = f'pretraining_init_{init}.log'
            csv_log_filepath = os.path.join(self.getBaseResultsPath(), 'Logs', csv_log_filename)
            csv_logger = tf.keras.callbacks.CSVLogger(csv_log_filepath)

            self.logger.debug(f'Logging training to {csv_log_filepath}')
        
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
                                      validation_data=(norm_test_data, norm_test_data),
                                      shuffle=True
                                      )
            duration = str(timedelta(seconds=float(time.time() - train_start_time)))
            self.logger.info(f'The training of the #{i_init+1} initilization took {duration} seconds')
            if np.min(init_trn_desc.history['val_loss']) < best_loss:
                best_init = i_init
                trn_desc['best_init'] = best_init
                best_loss = np.min(init_trn_desc.history['val_loss'])
                best_model = model
                trn_desc['epochs'] = init_trn_desc.epoch

                for imetric in range(len(self.parameters["HyperParameters"]["metrics"])):
                    if self.parameters["HyperParameters"]["metrics"][imetric] == 'accuracy':
                        metric = 'acc'
                    else:
                        metric = self.parameters["HyperParameters"]["metrics"][imetric]
                    trn_desc[metric] = init_trn_desc.history[metric]
                    trn_desc['val_' + metric] = init_trn_desc.history['val_' + metric]

                trn_desc['loss'] = init_trn_desc.history['loss']
                trn_desc['val_loss'] = init_trn_desc.history['val_loss']
        # save model
        file_name = f'{model_str}_fold_{ifold}_model.h5'
        best_model.save(file_name)
        file_name = f'{model_str}_fold_{ifold}_trn_desc.jbl'
        joblib.dump([trn_desc], file_name, compress=9)

        self.update_vae_model(inovelty, ifold, best_model)

        self.__notify_training_conclusion(duration, 'Pre-Training', inovelty)

        return ifold, best_model, trn_desc

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
