import sys
import os
import time
import multiprocessing
import pprint 

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from Packages.NoveltyDetection.setup.noveltyDetectionConfig import CONFIG
from Packages.NoveltyDetection.StackedAutoEncoders.SAENoveltyDetectionAnalysis import SAENoveltyDetectionAnalysis
from Functions.telegrambot import Bot

num_processes = multiprocessing.cpu_count()

my_bot = Bot("lisa_thebot")

# Enviroment variables
data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']

training_params = {
    "Technique": "StackedAutoEncoder"
}
analysis = SAENoveltyDetectionAnalysis(parameters=training_params, 
#                                        model_hash='3aa7b6b2a784922c348292561edca3d5201d6d6567f727e6ce7e403d7f175b10',
#                                        model_hash="048afa2017b8b40203db1ef3f94e806a4d9772fe14b032a479f96f9551853574", 
#                                        model_hash='5fee10c0e061666bbd9b5ad4544503a562deb56f25bd5f80b9f2c8ca3bf76b81', # PF linear no decoder
                                       model_hash='f84a0f006d8a19e5c3dbe25d4907659013495f79bb8b377e88673a899d9e2a2d', # PF tanh no decoder
                                       load_hash=True, load_data=True, verbose=False)
all_data, all_trgt, trgt_sparse = analysis.getData()

SAE = analysis.createSAEModels()

trn_data = analysis.trn_data
trn_trgt = analysis.trn_trgt
trn_trgt_sparse = analysis.trn_trgt_sparse

print("Model Hash: {}".format(analysis.model_hash))

# Neurons variation x Figures of Merit
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
from keras.utils import to_categorical
from Functions.StatisticalAnalysis import KLDiv, EstPDF

analysis_name = "neurons_variation_results_dataframe"
analysis_file = os.path.join(analysis.getBaseResultsPath(), "AnalysisFiles", analysis_name + ".csv")  

if not os.path.exists(analysis_file): 
    columns = [
                'layers',
                'topology',
                'neurons_value',
                'novelty_class',
                'fold',
                'train_mse_known',
                'test_mse_known',
                'train_mse_A',
                'test_mse_A',
                'train_mse_B',
                'test_mse_B',
                'train_mse_C',
                'test_mse_C',
                'train_mse_D',
                'test_mse_D',
                'mse_novelty',
                'train_kl_div_known',
                'test_kl_div_known', 
                'kl_div_novelty', 
                'train_sp_index',
                'test_sp_index',
                'train_accuracy',
                'test_accuracy',
                'train_recall',
                'test_recall',
                'train_precision',
                'test_precision',
                'train_f1_score',
                'test_f1_score',
                'train_efficiency_A',
                'test_efficiency_A',
                'train_efficiency_B',
                'test_efficiency_B',
                'train_efficiency_C',
                'test_efficiency_C',
                'train_efficiency_D',
                'test_efficiency_D'
              ]
    results = pd.DataFrame(columns=columns)


    layers = [1]
    layer = 1

    n_folds = analysis.n_folds

    neurons_mat = [1] + list(range(50,450,50))

    ineuron_inovelty_ifold = [(x,y,z) for x in neurons_mat for y in range(len(analysis.class_labels)) for z in range(analysis.n_folds)]

    n_bins = 100

    def getKL(y_true, y_pred):
        kl = np.zeros([y_true.shape[1]], dtype=object)
        for ifrequency in range(0,y_true.shape[1]):
            # Calculate KL Div for known data reconstruction
            y_true_freq = y_true[:,ifrequency]
            reconstructed_y_true = y_pred[:,ifrequency]

            m_bins = np.linspace(y_true_freq.min(), y_true_freq.max(), n_bins)

            kl[ifrequency] = KLDiv(y_true_freq.reshape(-1,1), reconstructed_y_true.reshape(-1,1),
                                   bins=m_bins, mode='kernel', kernel='epanechnikov',
                                   kernel_bw=0.1, verbose=False)

            kl[ifrequency] = kl[ifrequency][0]
        return kl

    def get_results(param_ineuron_inovelty_ifold):
        ineuron, inovelty, ifold = param_ineuron_inovelty_ifold

        hidden_neurons = [ineuron]
        neurons_str = SAE[inovelty].get_neurons_str(trn_data[inovelty],hidden_neurons=hidden_neurons)  

        verbose = True

        class_eff_train_mat = np.zeros([len(np.unique(all_trgt))])
        class_eff_test_mat = np.zeros([len(np.unique(all_trgt))])

        train_id, test_id = analysis.CVO[inovelty][ifold]

        print('Novelty class: {} - Topology: {} - Fold {}'.format(analysis.class_labels[inovelty],
                                                                  SAE[inovelty].get_neurons_str(data=trn_data[inovelty], hidden_neurons=hidden_neurons)+'x'+str(trn_trgt_sparse[inovelty].shape[1]),
                                                                  ifold
                                                                 )
             )
        # Load Models
        autoencoder = SAE[inovelty].get_model(data=trn_data[inovelty],
                                              trgt=trn_trgt[inovelty],
                                              hidden_neurons=hidden_neurons[:layer-1]+[ineuron],
                                              layer=layer,
                                              ifold=ifold)

        classifier = SAE[inovelty].load_classifier(data=analysis.trn_data[inovelty],
                                                   trgt=analysis.trn_trgt[inovelty],
                                                   hidden_neurons = hidden_neurons[:layer-1]+[ineuron],
                                                   layer = layer,
                                                   ifold = ifold)


        scaler = analysis.get_data_scaler(inovelty=inovelty, ifold=ifold)

        known_train_data = scaler.transform(analysis.trn_data[inovelty][train_id,:])
        known_train_target = analysis.trn_trgt[inovelty][train_id]

        known_test_data = scaler.transform(analysis.trn_data[inovelty][test_id,:])
        known_test_target = analysis.trn_trgt[inovelty][test_id]

        novelty_data = scaler.transform(all_data[all_trgt==inovelty])

        # Get reconstruction outputs
        autoencoder_known_train_output = autoencoder.predict(known_train_data)
        autoencoder_known_test_output = autoencoder.predict(known_test_data)

        autoencoder_novelty_output = autoencoder.predict(novelty_data)

        # Get classification outputs
        classifier_known_train_output = classifier.predict(known_train_data)
        classifier_known_test_output = classifier.predict(known_test_data)

        classifier_novelty_output = classifier.predict(novelty_data)

        buff_train = np.zeros([len(np.unique(analysis.all_trgt))-1])
        buff_test = np.zeros([len(np.unique(analysis.all_trgt))-1])

        thr_value = 0.1
        for iclass, class_id in enumerate(np.unique(analysis.all_trgt)):
            if iclass == inovelty:
                continue
            ### Train 
            output_of_class_events = classifier_known_train_output[known_train_target==iclass-(iclass>inovelty),:]
            correct_class_output = np.argmax(output_of_class_events,axis=1)==iclass-(iclass>inovelty)

            output_above_thr = output_of_class_events[correct_class_output,iclass-(iclass>inovelty)]>=thr_value
            class_eff_train_mat[iclass] = float(sum(output_above_thr))/float(len(output_of_class_events))
            buff_train[iclass-(iclass>inovelty)] = class_eff_train_mat[iclass]
            ########################################3
            ### Test 
            output_of_class_events = classifier_known_test_output[known_test_target==iclass-(iclass>inovelty),:]
            correct_class_output = np.argmax(output_of_class_events,axis=1)==iclass-(iclass>inovelty)

            output_above_thr = output_of_class_events[correct_class_output,iclass-(iclass>inovelty)]>=thr_value
            class_eff_test_mat[iclass] = float(sum(output_above_thr))/float(len(output_of_class_events))
            buff_test[iclass-(iclass>inovelty)] = class_eff_test_mat[iclass]

        # Get results
        train_sp_index = (np.sqrt(np.mean(buff_train,axis=0)*np.power(np.prod(buff_train),1./float(len(buff_train)))))
        test_sp_index = (np.sqrt(np.mean(buff_test,axis=0)*np.power(np.prod(buff_test),1./float(len(buff_test)))))

        train_acc = np.mean(buff_train,axis=0)
        test_acc = np.mean(buff_test,axis=0)

        y_train_pred_above_thr = np.argmax(classifier_known_train_output[classifier_known_train_output.max(axis=1)>=thr_value], axis=1)
        y_train_true_above_thr = known_train_target[classifier_known_train_output.max(axis=1)>=thr_value]

        y_test_pred_above_thr = np.argmax(classifier_known_test_output[classifier_known_test_output.max(axis=1)>=thr_value], axis=1)
        y_test_true_above_thr = known_test_target[classifier_known_test_output.max(axis=1)>=thr_value]

        if len(y_train_pred_above_thr)!=0 and len(y_train_true_above_thr)!=0:
            train_recall = metrics.recall_score(y_train_true_above_thr, y_train_pred_above_thr, average='macro')
            train_precision = metrics.precision_score(y_train_true_above_thr, y_train_pred_above_thr, average='macro')
            train_f1_score =metrics.f1_score(y_train_true_above_thr, y_train_pred_above_thr, average='macro')
        else: 
            train_recall = 0
            train_precision = 0
            train_f1_score = 0

        if len(y_test_pred_above_thr)!=0 and len(y_test_true_above_thr)!=0:
            test_recall = metrics.recall_score(y_test_true_above_thr, y_test_pred_above_thr, average='macro')
            test_precision = metrics.precision_score(y_test_true_above_thr, y_test_pred_above_thr, average='macro')
            test_f1_score =metrics.f1_score(y_test_true_above_thr, y_test_pred_above_thr, average='macro')
        else: 
            test_recall = 0
            test_precision = 0
            test_f1_score = 0

        # Pre-training
        mse_train_known = metrics.mean_squared_error(known_train_data, autoencoder_known_train_output)
        mse_test_known = metrics.mean_squared_error(known_test_data, autoencoder_known_test_output)
        mse_novelty = metrics.mean_squared_error(novelty_data, autoencoder_novelty_output)

        train_mse_per_class = {}
        test_mse_per_class = {}

        for iclass, class_id in enumerate(np.unique(analysis.all_trgt)):
            if iclass == inovelty:
                train_mse_per_class[iclass] = 0
                test_mse_per_class[iclass] = 0
                continue

            train_mse_per_class[iclass] = metrics.mean_squared_error(known_train_data[known_train_target==iclass-(iclass>inovelty),:],
                                                                    autoencoder_known_train_output[known_train_target==iclass-(iclass>inovelty),:])

            test_mse_per_class[iclass] = metrics.mean_squared_error(known_test_data[known_test_target==iclass-(iclass>inovelty),:],
                                                                    autoencoder_known_test_output[known_test_target==iclass-(iclass>inovelty),:])

        kl_div_train_known = '' #getKL(known_train_data, autoencoder_known_train_output)
        kl_div_test_known = getKL(known_test_data, autoencoder_known_test_output)
        kl_div_novelty = getKL(novelty_data, autoencoder_novelty_output)

        # Append to dataframe
        result = {
                    'novelty_class': analysis.getClassLabels()[inovelty],
                    'layers': layer, 
                    'topology': neurons_str,
                    'neurons_value': ineuron,
                    'fold': ifold,

                    'train_mse_known': mse_train_known,
                    'test_mse_known': mse_test_known,

                    'train_mse_A': train_mse_per_class[0], 
                    'test_mse_A': test_mse_per_class[0], 

                    'train_mse_B': train_mse_per_class[1], 
                    'test_mse_B': test_mse_per_class[1], 

                    'train_mse_C': train_mse_per_class[2],
                    'test_mse_C': test_mse_per_class[2],

                    'train_mse_D': train_mse_per_class[3],
                    'test_mse_D': test_mse_per_class[3],

                    'mse_novelty': mse_novelty,

                    'train_kl_div_known': kl_div_train_known,
                    'test_kl_div_known': kl_div_test_known,

                    'kl_div_novelty': kl_div_novelty,

                    'train_sp_index': test_sp_index,
                    'test_sp_index': train_sp_index,

                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,

                    'train_recall': train_recall,
                    'test_recall': test_recall,

                    'train_precision': train_precision,
                    'test_precision': test_precision,

                    'train_f1_score': train_f1_score,
                    'test_f1_score': test_f1_score,

                    'train_efficiency_A': class_eff_train_mat[ 0], 
                    'test_efficiency_A': class_eff_test_mat[0], 

                    'train_efficiency_B': class_eff_train_mat[1], 
                    'test_efficiency_B': class_eff_test_mat[1], 

                    'train_efficiency_C': class_eff_train_mat[2],
                    'test_efficiency_C': class_eff_test_mat[2],

                    'train_efficiency_D': class_eff_train_mat[3],
                    'test_efficiency_D': class_eff_test_mat[3]
        }
        return result

    # Start Parallel processing
    with multiprocessing.Pool(processes=num_processes) as p:
        results_map = p.map(get_results, ineuron_inovelty_ifold)

    for result in results_map:
        results = results.append(result, ignore_index=True)
    # Save results
    results.drop_duplicates(inplace=True, keep='last')
    results.sort_values(by=['neurons_value', 'fold']).to_csv(analysis_file,index=False)
    try: 
        my_bot.sendMessage(message="Neurons Variation Results obtained!", filePath=analysis_file)
    except Exception as e:
        print("Error when sending message to the bot. Error: {}".format(str(e)))
else:
    print("[+] File {} exists".format(analysis_file))
    
    try: 
        my_bot.sendMessage(message="Neurons Variation Results obtained!", filePath=analysis_file)
    except Exception as e:
        print("Error when sending message to the bot. Error: {}".format(str(e)))
    