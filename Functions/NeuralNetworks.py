# -*- coding: utf-8 -*-
"""
   Author: Vinícius dos Santos Mello viniciusdsmello at poli.ufrj.br
   Class created to implement a Neural Network for Classification and Novelty Detection.
"""
import os
import numpy as np

from sklearn.externals import joblib
from sklearn import preprocessing

from keras.models import Sequential
from keras import regularizers
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD
from keras import callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras import losses

from Functions.MetricsLosses import kullback_leibler_divergence
from Functions.lossWeights import getGradientWeights

import multiprocessing

num_processes = multiprocessing.cpu_count()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)))

# noinspection PyUnusedLocal
class NeuralNetworks:
    def __init__(self, parameters=None, save_path='', CVO=None, inovelty=0, verbose=False):
        self.save_path = save_path
        self.model_save_path = os.path.join(save_path, "Models")
        self.logs_save_path = os.path.join(save_path, "Logs")
        
        self.inovelty = inovelty
        self.parameters = parameters
        self.verbose = verbose

        # Distinguish between a SAE for Novelty Detection and SAE for just Classification
        if bool(self.parameters["NoveltyDetection"]):
            self.CVO = CVO[self.inovelty]
            self.prefix_str = "neuralnetwork_model_%i_novelty" % self.inovelty
        else:
            self.CVO = CVO
            self.prefix_str = "neuralnetwork_model"

        # Choose optmizer algorithm
        if self.parameters["OptmizerAlgorithm"]["name"] == 'SGD':
            self.optmizer = SGD(lr=self.parameters["OptmizerAlgorithm"]["parameters"]["learning_rate"],
                                nesterov=self.parameters["OptmizerAlgorithm"]["parameters"]['nesterov'])

        elif self.parameters["OptmizerAlgorithm"]["name"] == 'Adam':
            self.optmizer = Adam(lr=self.parameters["OptmizerAlgorithm"]["parameters"]["learning_rate"],
                                 beta_1=self.parameters["OptmizerAlgorithm"]["parameters"]["beta_1"],
                                 beta_2=self.parameters["OptmizerAlgorithm"]["parameters"]["beta_2"],
                                 epsilon=self.parameters["OptmizerAlgorithm"]["parameters"]["epsilon"])
        else:
            self.optmizer = self.parameters["OptmizerAlgorithm"]["name"]

        # Choose loss functions
        if self.parameters["HyperParameters"]["loss"] == "kullback_leibler_divergence":
            self.lossFunction = kullback_leibler_divergence
        else:
            self.lossFunction = self.parameters["HyperParameters"]["loss"]
        losses.custom_loss = self.lossFunction

    # Method that creates a string in the format: (InputDimension)x(1º Layer Dimension)x...x(Nº Layer Dimension)
    @staticmethod
    def get_neurons_str(data, hidden_neurons=None):
        if hidden_neurons is None:
            hidden_neurons = [1]
        neurons_str = str(data.shape[1])
        for ineuron in hidden_neurons:
            neurons_str = neurons_str + 'x' + str(ineuron)
        return neurons_str

    # Method that preprocess data normalizing it according to "norm" parameter.
    def normalize_data(self, data, ifold):
        # normalize data based in train set
        train_id, test_id = self.CVO[ifold]
        if self.parameters["HyperParameters"]["norm"] == 'mapstd':
            scaler = preprocessing.StandardScaler().fit(data[train_id, :])
        elif self.parameters["HyperParameters"]["norm"] == 'mapstd_rob':
            scaler = preprocessing.RobustScaler().fit(data[train_id, :])
        elif self.parameters["HyperParameters"]["norm"] == 'mapminmax':
            scaler = preprocessing.MinMaxScaler().fit(data[train_id, :])
        else:
            return data
        norm_data = scaler.transform(data)

        return norm_data

    # Method that return the Neural Network Model
    def get_model(self, data, trgt, hidden_neurons=None, layer=1, ifold=0):
        if hidden_neurons is None:
            hidden_neurons = [1]
        model = None
        if layer > len(hidden_neurons):
            print("[-] Error: The parameter layer must be less or equal to the size of list hidden_neurons")
            return 1
        
        neurons_str = self.get_neurons_str(data, hidden_neurons[:layer])

        previous_model_str = os.path.join(self.model_save_path, self.prefix_str + "_{}_neurons".format(neurons_str))

        file_name = '%s_fold_%i_model.h5' % (previous_model_str, ifold)

        # Check if previous layer model was trained
        if not os.path.exists(file_name):
            self.train(data=data,
                       trgt=trgt,
                       ifold=ifold,
                       hidden_neurons=hidden_neurons[:layer],
                       layer=layer
                       )

        model = load_model(file_name,
                           custom_objects={'%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})
        
        return model

    # Method used to perform the training of the neural network
    def train(self, data=None, trgt=None, ifold=0, hidden_neurons=None, layer=1):
        # Change elements equal to zero to one
        if hidden_neurons is None:
            hidden_neurons = [1]
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1
        if (layer <= 0) or (layer > len(hidden_neurons)):
            print("[-] Error: The parameter layer must be greater than zero and less " \
                  "or equal to the length of list hidden_neurons")
            return -1

        neurons_str = self.get_neurons_str(data, hidden_neurons[:layer])

        model_str = os.path.join(self.model_save_path, self.prefix_str + "_{}_neurons".format(neurons_str))

        trgt_sparse = np_utils.to_categorical(trgt.astype(int))
    
        file_name = '%s_fold_%i_model.h5' % (model_str, ifold)
        if os.path.exists(file_name):
            if self.verbose:
                print('File %s exists' % file_name)
            # load model
            file_name = '%s_fold_%i_model.h5' % (model_str, ifold)
            classifier = load_model(file_name, custom_objects={
                '%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})
            model_str = os.path.join(self.logs_save_path, self.prefix_str + "_{}_neurons".format(neurons_str))

            file_name = '%s_fold_%i_trn_desc.jbl' % (model_str, ifold)
            trn_desc = joblib.load(file_name)
            return ifold, classifier, trn_desc

        train_id, test_id = self.CVO[ifold]

        best_init = 0
        best_loss = 999

        classifier = []
        trn_desc = {}

        norm_data = self.normalize_data(data, ifold)
        
        for i_init in range(self.parameters["HyperParameters"]["n_inits"]):
            print('Neural Network - Layer: %i - Topology: %s - Fold %i of %i Folds -  Init %i of %i Inits' % (layer,
                                                                                                              neurons_str,
                                                                                                              ifold + 1,
                                                                                                              self.parameters["HyperParameters"]["n_folds"],
                                                                                                              i_init + 1,
                                                                                                              self.parameters["HyperParameters"]["n_inits"]
                                                                                                              ))
            model = Sequential()
            
            for ilayer in range(layer):
                if ilayer == 1:
                    #if bool(self.parameters["HyperParameters"]["dropout"]):
                    #    model.add(Dropout(int(self.parameters["HyperParameters"]["dropout_parameter"])))                
                    
                    if self.parameters["HyperParameters"]["regularization"] == "l1":
                        model.add(Dense(units=hidden_neurons[ilayer], input_dim=data.shape[1],
                                    kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"],
                                    kernel_regularizer=regularizers.l1(self.parameters["HyperParameters"]["regularization_parameter"])))

                    elif self.parameters["HyperParameters"]["regularization"] == "l2":
                        model.add(Dense(hidden_neurons[ilayer], input_dim=data.shape[1],
                                        kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"],
                                        kernel_regularizer=regularizers.l2(self.parameters["HyperParameters"]["regularization_parameter"])))
                    else:
                        model.add(Dense(hidden_neurons[ilayer], input_dim=data.shape[1],
                                        kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"]))    
                else:
                    #if bool(self.parameters["HyperParameters"]["dropout"]):
                    #    model.add(Dropout(int(self.parameters["HyperParameters"]["dropout_parameter"])))                

                    if self.parameters["HyperParameters"]["regularization"] == "l1":
                        model.add(Dense(units=hidden_neurons[ilayer],
                                    kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"],
                                    kernel_regularizer=regularizers.l1(self.parameters["HyperParameters"]["regularization_parameter"])))

                    elif self.parameters["HyperParameters"]["regularization"] == "l2":
                        model.add(Dense(hidden_neurons[ilayer],
                                        kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"],
                                        kernel_regularizer=regularizers.l2(self.parameters["HyperParameters"]["regularization_parameter"])))
                    else:
                        model.add(Dense(hidden_neurons[ilayer],
                                        kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"]))    
                
                model.add(Activation(self.parameters["HyperParameters"]["hidden_activation_function"]))
                
            # Add Output Layer
            model.add(Dense(units=trgt_sparse.shape[1],
                            kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"]))
            model.add(Activation(self.parameters["HyperParameters"]["classifier_output_activation_function"]))

            model.compile(loss=self.lossFunction,
                          optimizer=self.optmizer,
                          metrics=self.parameters["HyperParameters"]["metrics"])
            
            # Train model
            earlyStopping = callbacks.EarlyStopping(monitor=self.parameters["callbacks"]["EarlyStopping"]["monitor"],
                                                    patience=self.parameters["callbacks"]["EarlyStopping"]["patience"],
                                                    verbose=self.verbose,
                                                    mode='auto')
            class_weights = getGradientWeights(trgt[train_id])
            init_trn_desc = model.fit(norm_data[train_id], trgt_sparse[train_id],
                                      epochs=self.parameters["HyperParameters"]["n_epochs"],
                                      batch_size=self.parameters["HyperParameters"]["batch_size"],
                                      callbacks=[earlyStopping],
                                      verbose=self.verbose,
                                      validation_data=(norm_data[test_id], trgt_sparse[test_id]),
                                      shuffle=True,
                                      class_weight=class_weights
                                      )
            if np.min(init_trn_desc.history['val_loss']) < best_loss:
                best_init = i_init
                trn_desc['best_init'] = best_init
                best_loss = np.min(init_trn_desc.history['val_loss'])
                classifier = model
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
        file_name = '%s_fold_%i_model.h5' % (model_str, ifold)
        classifier.save(file_name)
        
        # save train history
        model_str = os.path.join(self.logs_save_path, self.prefix_str + "_{}_neurons".format(neurons_str))
        file_name = '%s_fold_%i_trn_desc.jbl' % (model_str, ifold)
        joblib.dump([trn_desc], file_name, compress=9)

        return ifold, classifier, trn_desc