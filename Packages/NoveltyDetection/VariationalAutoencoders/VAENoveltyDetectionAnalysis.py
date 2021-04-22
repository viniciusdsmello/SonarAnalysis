#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This file contains the Novelty Detection Analysis with Variational Autoencoders
    Author: Vinicius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
"""
import logging
import multiprocessing
import os
import time
from copy import deepcopy
from datetime import timedelta

import joblib
import numpy as np
import tensorflow as tf
from Functions.telegrambot import Bot
from Packages.NoveltyDetection.NoveltyDetectionAnalysis import NoveltyDetectionAnalysis
from sklearn import preprocessing

from models.vae import (Decoder, Encoder, VariationalAutoencoder, kl_div,
                        negative_expected_log_likelihood)

my_bot = Bot("lisa_thebot")

num_processes = multiprocessing.cpu_count()


def euclidean_distance(x, y):
    x = tf.cast(x, dtype='float32')
    y = tf.cast(y, dtype='float32')
    dist = tf.math.reduce_euclidean_norm(x - y, axis=1)
    return dist


class VAENoveltyDetectionAnalysis(NoveltyDetectionAnalysis):

    def __init__(self, parameters=None, model_hash=None, load_hash=False, load_data=True, verbose=False):
        super().__init__(parameters=parameters, model_hash=model_hash, load_hash=load_hash, load_data=load_data,
                         verbose=verbose, logger_level='DEBUG')

        self.trn_data = {}
        self.trn_trgt = {}
        self.trn_trgt_sparse = {}
        self.vae_models = {}

        self.base_model_str = 'vae_{inovelty}_inovelty_{ifold}_ifold'

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

    def emulate_novelties(self):
        # Divide the dataset to do a novelty class emulation
        self.logger.info('Emulating novelty classes...')
        for inovelty in range(self.all_trgt_sparse.shape[1]):
            self.logger.debug(f'Spliting data for novelty class #{self.getClassLabels()[inovelty]}...')
            self.trn_data[inovelty] = self.all_data[self.all_trgt != inovelty]
            self.trn_trgt[inovelty] = self.all_trgt[self.all_trgt != inovelty]
            self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] = self.trn_trgt[inovelty][
                self.trn_trgt[inovelty] > inovelty] - 1
            self.trn_trgt_sparse[inovelty] = tf.keras.utils.to_categorical(self.trn_trgt[inovelty].astype(int))

    def build_vae_models(self, load_from_storage: bool = True):
        # Initialize VAE objects for all novelties
        for inovelty, novelty_class in enumerate(self.getClassLabels()):
            self.logger.debug(f'Building VAE model for novelty class #{novelty_class}...')
            self.vae_models[inovelty] = {}
            for ifold in range(self.n_folds):
                base_model = self._create_vae_model()
                novelty_model = base_model

                self.vae_models[inovelty][ifold] = novelty_model

                if load_from_storage:
                    self.load_vae_model_from_storage(inovelty, ifold)

        return self.vae_models

    def _create_vae_model(self):
        original_dim = self.all_data.shape[1]
        intermediate_dim = self.parameters['TechniqueParameters']['IntermediateDimension']
        latent_dim = self.parameters['TechniqueParameters']['LatentDimension']

        encoder = Encoder(intermediate_dim=intermediate_dim, latent_dim=latent_dim)
        encoder.build((None, original_dim))
        decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
        decoder.build((None, latent_dim))
        model = VariationalAutoencoder(encoder, decoder, latent_dim=latent_dim)

        return model

    def get_vae_model(self, inovelty, ifold):
        return self.vae_models[inovelty][ifold]

    def update_vae_model(self, inovelty, ifold, model):
        self.vae_models[inovelty][ifold] = model

    def load_vae_model_from_storage(self, inovelty, ifold):
        model = self.get_vae_model(inovelty, ifold)

        model_str = self.base_model_str.format(inovelty=inovelty, ifold=ifold)

        file_name = f'{model_str}_fold_{ifold}'

        weights_filepath = os.path.join(self.getBaseResultsPath(), 'Models', file_name)
        try:
            self.logger.debug(f"Loading model weights from %s", weights_filepath)
            model.load_weights(weights_filepath)
            self.update_vae_model(inovelty, ifold, model)
        except Exception as e:
            self.logger.warn("No models files were found - %s", e)

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
        best_loss = 9999999999999

        best_model = []
        trn_desc = {}
        model_str = self.base_model_str.format(inovelty=inovelty, ifold=ifold)

        for i_init in range(self.parameters["HyperParameters"]["n_inits"]):
            model = self.get_vae_model(inovelty, ifold)
            self.logger.info(f'Running {i_init+1} initialization...')

            # Callbacks
            callbacks = []
            if self.parameters["callbacks"].get("EarlyStopping"):
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor=self.parameters["callbacks"]["EarlyStopping"]["monitor"],
                    patience=self.parameters["callbacks"]["EarlyStopping"]["patience"],
                    verbose=self.verbose,
                    mode='auto'
                )
                callbacks.append(early_stopping)

            csv_log_filename = f'training_history_inovelty_{inovelty}_ifold_{ifold}_init_{i_init}.log'
            csv_log_filepath = os.path.join(self.getBaseResultsPath(), 'Logs', csv_log_filename)
            csv_logger = tf.keras.callbacks.CSVLogger(csv_log_filepath)

            callbacks.append(csv_logger)

            self.logger.debug(f'Logging training to {csv_log_filepath}')

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.parameters["OptmizerAlgorithm"]["parameters"]["learning_rate"]
            )
            model.compile(optimizer=optimizer)

            train_start_time = time.time()
            init_trn_desc = model.fit(norm_train_data, norm_train_data,
                                      validation_data=(norm_test_data, norm_test_data),
                                      epochs=int(self.parameters["HyperParameters"]["pretraining_n_epochs"]),
                                      # PRE-TRAINING
                                      batch_size=self.parameters["HyperParameters"]["batch_size"],
                                      callbacks=callbacks,
                                      verbose=self.verbose,
                                      shuffle=True
                                      )
            duration = str(timedelta(seconds=float(time.time() - train_start_time)))
            self.logger.info(f'The training of the #{i_init+1} initilization took {duration} seconds')
            if np.min(init_trn_desc.history['val_total_loss']) < best_loss:
                best_init = i_init
                trn_desc['best_init'] = best_init
                best_loss = np.min(init_trn_desc.history['val_total_loss'])
                best_model = init_trn_desc.model
                trn_desc['epochs'] = init_trn_desc.epoch

                trn_desc['reconstruction_loss'] = init_trn_desc.history['reconstruction_loss']
                trn_desc['val_reconstruction_loss'] = init_trn_desc.history['val_reconstruction_loss']

                trn_desc['kl_loss'] = init_trn_desc.history['kl_loss']
                trn_desc['val_kl_loss'] = init_trn_desc.history['val_kl_loss']

                trn_desc['total_loss'] = init_trn_desc.history['total_loss']
                trn_desc['val_total_loss'] = init_trn_desc.history['val_total_loss']

        self.logger.info(f'Best initialization: {best_init+1} - Achieved loss: {best_loss}')
        # save model
        file_name = f'{model_str}_fold_{ifold}'
        weights_filepath = os.path.join(self.getBaseResultsPath(), 'Models', file_name)
        best_model.save_weights(weights_filepath)

        file_name = f'{model_str}_fold_{ifold}_trn_desc.jbl'
        trn_desc_filepath = os.path.join(self.getBaseResultsPath(), 'Models', file_name)
        joblib.dump([trn_desc], trn_desc_filepath, compress=9)

        self.update_vae_model(inovelty, ifold, best_model)

        self.__notify_training_conclusion(duration, 'Training', inovelty)

        return ifold, best_model, trn_desc

    def reconstruct(self, data: np.ndarray, inovelty: int, ifold: int, deterministic: bool = True):
        vae = self.get_vae_model(inovelty, ifold)

        z_mean, z_log_var, z = vae.encoder.predict(data)
        output_mean, output_log_var, output = vae.decoder.predict(z)

        if deterministic:
            return output_mean, output_log_var, output_mean
        else:
            return output_mean, output_log_var, output

    def compute_novelty_scores(self, inovelty: int, ifold: int, test_data: np.ndarray, n_samples: int = 10):
        """
        Args:
            test_data (np.ndarray): test data to compute novelty scores on
            n_samples (int): number of samples from normal distribution used for stochastic reconstructions
        Returns
            (dict): Dictionary containing novelty scores for VAE-regularizer metric and for all
                    reconstruction based metrics. Novelty score as a full VAE loss can be computed by
                    adding VAE-regularizer novelty score to one of the reconstruction likelihood scores.
        """

        novelty_scores = {}

        vae = self.get_vae_model(inovelty, ifold)

        # VAE regularizer
        z_mean, z_log_var, z = vae.encoder(test_data)
        novelty_scores["vae-reg"] = kl_div(z_mean, z_log_var, reduce_on_batch=False)

        # Deterministic reconstruction error
        output_mean, output_log_var, output = self.reconstruct(test_data, inovelty, ifold, deterministic=True)
        novelty_scores["dre"] = euclidean_distance(output, test_data)

        # Deterministic reconstruction likelihood
        novelty_scores["drl"] = negative_expected_log_likelihood(
            tf.cast(test_data, dtype='float32'),
            tf.cast(output_mean, dtype='float32'),
            tf.cast(output_log_var, dtype='float32'),
            sum_on_batch=False
        )

        # Decoder stochastic reconstruction error
        ns = np.zeros((len(test_data), n_samples))
        for i in range(n_samples):
            output_sampled = vae.decoder.sample((output_mean, output_log_var))
            ns[:, i] = euclidean_distance(test_data, output_sampled)

        novelty_scores["dsre"] = np.mean(ns, axis=-1)

        # Stochastic encoder
        ns_det_dec = np.zeros((len(test_data), n_samples))
        ns_ll = np.zeros((len(test_data), n_samples))
        ns_st_dec = np.zeros((len(test_data), n_samples))

        for i in range(n_samples):
            output_mean, output_log_var, output = self.reconstruct(test_data, inovelty, ifold, deterministic=False)
            ns_det_dec[:, i] = euclidean_distance(test_data, output_mean)
            ns_ll[:, i] = negative_expected_log_likelihood(
                tf.cast(test_data, dtype='float32'),
                tf.cast(output_mean, dtype='float32'),
                tf.cast(output_log_var, dtype='float32'),
                sum_on_batch=False
            )
            output_sampled = vae.decoder.sample((output_mean, output_log_var))
            ns_st_dec[:, i] = euclidean_distance(test_data, output_sampled)

        # Encoder stochastic reconstruction error
        novelty_scores["esre"] = np.mean(ns_det_dec, axis=-1)

        # Encoder stochastic reconstruction likelihood
        novelty_scores["esrl"] = np.mean(ns_ll, axis=-1)

        # Fully stochastic reconstruction error
        novelty_scores["fsre"] = np.min(ns_ll, axis=-1)

        return novelty_scores
