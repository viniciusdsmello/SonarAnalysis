#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This file contains the Novelty Detection Analysis with Variational Autoencoders
    Author: Vinicius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
"""
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, '..')

import pickle
import numpy as np
import time
import string
import json
import multiprocessing

from sklearn import metrics
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.externals import joblib

from Packages.NoveltyDetection.NoveltyDetectionAnalysis import NoveltyDetectionAnalysis
from Functions.StackedAutoEncoders import StackedAutoEncoders

from Functions.telegrambot import Bot

my_bot = Bot("lisa_thebot")

num_processes = multiprocessing.cpu_count()


class VAENoveltyDetectionAnalysis(NoveltyDetectionAnalysis):

    def __init__(self, parameters=None, model_hash=None, load_hash=False, load_data=True, verbose=False):
        super().__init__(parameters=parameters, model_hash=model_hash, load_hash=load_hash, load_data=load_data, verbose=verbose)
        
        self.trn_data = {}
        self.trn_trgt = {}
        self.trn_trgt_sparse = {}
        self.VAE = {}
        
        for inovelty in range(self.all_trgt_sparse.shape[1]):
            self.trn_data[inovelty] = self.all_data[self.all_trgt != inovelty]
            self.trn_trgt[inovelty] = self.all_trgt[self.all_trgt != inovelty]
            self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] = self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] - 1
            self.trn_trgt_sparse[inovelty] = np_utils.to_categorical(self.trn_trgt[inovelty].astype(int))
            
            if self.parameters["HyperParameters"]["classifier_output_activation_function"] in ["tanh"]:
                self.trn_trgt_sparse[inovelty] = 2 * self.trn_trgt_sparse[inovelty] - np.ones(self.trn_trgt_sparse[inovelty].shape)
            
            
    def create_vae_models(self):        
        for inovelty in range(self.all_trgt_sparse.shape[1]):
            # Initialize VAE objects for all novelties
            self.VAE[inovelty] = StackedAutoEncoders(parameters=self.parameters,
                                                     save_path=self.getBaseResultsPath(),
                                                     CVO=self.CVO,
                                                     inovelty=inovelty, 
                                                     verbose=self.verbose
                                                     )

        return self.VAE

    def train(self, model_hash="", inovelty=0, fineTuning=False, trainingType="normal", ifold=0, hidden_neurons=[1],
              neurons_variation_step=50, layer=1, numThreads=num_processes):
        startTime = time.time()
        if fineTuning == False:
            fineTuning = 0
        else:
            fineTuning = 1

        hiddenNeuronsStr = str(hidden_neurons[0])
        if len(hidden_neurons) > 1:
            for ineuron in hidden_neurons[1:]:
                hiddenNeuronsStr = hiddenNeuronsStr + 'x' + str(ineuron)
        packagePath = os.path.join(self.PACKAGE_PATH, "StackedAutoEncoders")
        file_to_run = os.path.join(packagePath, "VAE_train.py")
        sysCall = "python " + file_to_run + " --layer {0} --novelty {1} --finetunning {2} --threads {3} --type {4} --hiddenNeurons {5} --neuronsVariationStep {6} --modelhash {7}".format(
            layer, inovelty, fineTuning, numThreads, trainingType, hiddenNeuronsStr, neurons_variation_step, model_hash)
        print(sysCall)
        os.system(sysCall)
        duration = str(timedelta(seconds=float(time.time() - startTime)))
        message = "Technique: {}\n".format(self.parameters['Technique'])
        message += "Development Mode: {}\n".format(self.parameters['DevelopmentMode'])
        message += "Training Type: {}\n".format(trainingType)
        message += "Novelty Class: {}\n".format(self.class_labels[inovelty])
        message += "Hash: {}\n".format(self.model_hash)
        message += "Duration: {}\n".format(duration)
        try:
            my_bot.sendMessage(message)
        except Exception as e:
            print("Erro ao enviar mensagem. Erro: " + str(e))

    def get_data_scaler(self, inovelty=0, ifold=0):
        train_id, test_id = self.CVO[inovelty][ifold]

        # normalize known classes
        if self.parameters["HyperParameters"]["norm"] == "mapstd":
            scaler = preprocessing.StandardScaler().fit(self.all_data[self.all_trgt!=inovelty][train_id,:])
        elif self.parameters["HyperParameters"]["norm"] == "mapstd_rob":
            scaler = preprocessing.RobustScaler().fit(self.all_data[self.all_trgt!=inovelty][train_id,:])
        elif self.parameters["HyperParameters"]["norm"] == "mapminmax":
            scaler = preprocessing.MinMaxScaler().fit(self.all_data[self.all_trgt!=inovelty][train_id,:])
        
        return scaler