import pandas as pd
from tabulate import tabulate
import numpy as np
from datetime import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# seaborn: added to env.
# Keras: added to env.
# hyperopt: added to env.
# ADD ? imbalance-learn

import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from os import path
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE  ####
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.utils import shuffle
import random
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.optimizers import RMSprop, SGD
# import tensorflow_datasets as tensorflow_datasets
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
# from __future__ import absolute_import, division, print_function, unicode_literals ###
import matplotlib.image as mpimg
from scipy import misc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
print(K.image_data_format())
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
# from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping  # keras.callbacks
from tensorflow.keras.callbacks import ModelCheckpoint # keras.callbacks


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval  # rand, GridSearch

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from datetime import datetime

import lifelines
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

# import sksurv # scikit-survival
# from sksurv.ensemble import RandomSurvivalForest


from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

import pycox
from pycox.datasets import metabric
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

import bokeh as bk
import shap

import platform


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
# warnings.filterwarnings('ignore')
np.random.seed(7)
tf.random.set_seed(12)
_ = torch.manual_seed(123)



print("tf", tf.__version__) # 2.5.0 # Default GPU Device: /device:GPU:0
print("np",np.__version__) # 1.19.5
print("pd",pd.__version__) # 1.2.4
print("sk",sk.__version__) # 0.24.2
print("bk",bk.__version__) # 2.3.3
print("sns",sns.__version__) # 0.11.1
print("torch",torch.__version__) # 1.9.0+cu111
print("pycox",pycox.__version__) # 0.2.2
print("lifelines",lifelines.__version__) # 0.26.0
# print("sksurv",sksurv.__version__) # 0.15.0.post0  # scikit-survival
print("hyperopt",hyperopt.__version__) # 0.2.5
print("keras",keras.__version__) # 2.5.0

if tf.test.gpu_device_name():

    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF\n")

if not torch.cuda.is_available():
    print("Please install GPU version of Torch\n")

# tf 2.5.0
# np 1.19.5
# pd 1.2.4
# sk 0.24.2
# bk 2.3.3
# sns 0.11.1
# torch 1.9.0+cpu  # torch 1.9.0+cu111 # got virt env and ./pip3
# pycox 0.2.2
# lifelines 0.26.0
# sksurv 0.15.0.post0
# hyperopt 0.2.5
# keras 2.5.0
# Please install GPU version of TF

# nvcc --version
# Cuda compilation tools, release 11.1, V11.1.105
# Build cuda_11.1.TC455_06.29190527_0

# https://pytorch.org/get-started/locally/
########################################################################################################################
glob_x_train, glob_x_val, glob_x_test = [],[],[]
glob_y_train, glob_y_val, glob_y_test = [],[],[]
final_features = []
labtrans= []
final_model_ae = False
########################################################################################################################
logLocation = ""
if platform.system() != 'Windows':
    logLocation = "/home/hmoradi/Downloads/PycharmProject/JacksonHeart/"
logFile = open(logLocation+'Result_Logfile.txt', 'w')

def print_log(*args): # file = logFile # *args,**kwargs  # kwargs = { 'file' : logFile }
    print(*args)
    print(*args,file = logFile)
########################################################################################################################
def getXfromBestModelfromTrials(trials,x):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result'][x]
########################################################################################################################
def eval_model(param_grid):

    global glob_x_train, glob_x_val, glob_x_test
    global glob_y_train, glob_y_val, glob_y_test
    global labtrans

    in_features = glob_x_train.shape[1]
    num_layers = param_grid['num_layers']
    num_neurons = param_grid['num_neurons']
    out_features = labtrans.out_features # glob_y_train.shape[1]
    batch_norm = param_grid['batch_norm']
    dropout = param_grid['dropout']  # 0.1
    learning_rate = param_grid['learning_rate'] # 0.01
    batch_size = param_grid['batch_size']
    epochs = param_grid['epochs'] # 100

    # momentu = param_grid['']
    # actiovation_func = param_grid['']
    # loss =  param_grid['']
    # optimizer = param_grid['']

    ###### <<<<<<
    layers = []
    #######
    layers.append(torch.nn.Linear(in_features, num_neurons))
    layers.append(torch.nn.ReLU())
    if batch_norm:
        layers.append(torch.nn.BatchNorm1d(num_neurons))
    layers.append(torch.nn.Dropout(dropout))
    ######
    for l in range(num_layers):
        layers.append(torch.nn.Linear(num_neurons, num_neurons))
        layers.append(torch.nn.ReLU())
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(num_neurons))
        layers.append(torch.nn.Dropout(dropout))

    layers.append(torch.nn.Linear(num_neurons, out_features))
    ######
    net = torch.nn.Sequential( *layers  )
    ###### <<<<<

    model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)

    model.optimizer.set_lr(learning_rate)

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(glob_x_train, glob_y_train, batch_size, epochs, callbacks, val_data=(glob_x_val,glob_y_val))
    # _ = log.plot()

    surv = model.predict_surv_df(glob_x_test)
    # surv = model.interpolate(10).predict_surv_df(glob_x_test)

    durations_test, events_test = glob_y_test

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    acc =  ev.concordance_td('antolini')
    print('v.concordance_td(\'antolini\')',acc )
    return({'loss': -acc, 'status': STATUS_OK, 'test':['test'], 'log':log, 'acc':acc, 'model':model})
########################################################################################################################
def eval_ae(param_grid):
    print('TEST__ param_grid:', param_grid)
    global glob_x_train, glob_x_val, glob_x_test
    global glob_y_train, glob_y_val, glob_y_test
    global labtrans
    global final_model_ae

    in_features = glob_x_train.shape[1]
    num_layers = param_grid['num_layers']
    num_neurons = param_grid['num_neurons']
    mid_dim = param_grid['encode_dim']
    batch_norm = param_grid['batch_norm']
    dropout = param_grid['dropout']  # 0.1
    learning_rate = param_grid['learning_rate'] # 0.01
    batch_size = param_grid['batch_size']
    epochs = param_grid['epochs'] # 100

    ###### <<<<<<

    # model__ae
    print('glob_x_train.shape ',glob_x_train.shape)
    inputA = Input(shape=(in_features,))

    l = 'En'
    fully = Dense(num_neurons, activation="relu", name="Fully-con-" + str(l))(inputA)
    fully = Dropout(dropout, name="Drop-" + str(l))(fully)

    for lay in range(2,num_layers+1,1):
        l = 'En-'+str(lay)
        fully = Dense(num_neurons//lay, activation="relu", name="Fully-con-" + str(l))(fully)
        fully = Dropout(dropout, name="Drop-" + str(l))(fully)

    l = 'mid'
    encode = Dense(mid_dim, activation="relu", name="Fully-con-" + str(l))(fully)
    # fully = Dropout(drop_rate_ae,name="Drop-"+str(l))(fully)
    # for l in range(2,layers_comb+1,1):

    for lay in range(num_layers,1, -1):
        l = 'De-' + str(lay)
        fully = Dense(num_neurons // lay, activation="relu", name="Fully-con-" + str(l))(fully)
        fully = Dropout(dropout, name="Drop-" + str(l))(fully)

    l = 'De'
    decode = Dense(num_neurons, activation="relu", name="Fully-con-" + str(l))(fully)
    decode = Dropout(dropout, name="Drop-" + str(l))(fully)

    l = 'out'
    decode = Dense( in_features, activation="sigmoid", name="Fully-con-" + str(l))(decode)
    # decode = Dropout(drop_rate_ae,name="Drop-"+str(l))(fully)

    ae_model = Model(inputs=inputA, outputs=decode)
    encoder = Model(inputs=inputA, outputs=encode)
    # if final_model:
    # model.summary()
    verbose = 0
    patience= 15
    # print("Compile")
    loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mse_loss") #  MeanAbsolutePercentageError # MeanSquaredLogarithmicError #
    metric = tf.keras.metrics.MeanSquaredError(name="mse_metric", dtype=None)
    ae_model.compile(loss=loss,
                     # 'binary_crossentropy' # hinge # squared_hinge (-1 1 binary)  # categorical_crossentropy  # sparse_categorical_crossentropy # kullback_leibler_divergence (distribution based - auto encoder)
                     optimizer=Adam(learning_rate=learning_rate),
                     # RMSprop(lr=0.001) # 'adam' # SGD(lr=learning_ratio, momentum=momentu, lr=0.01, momentum=0.9) # 'rmsprop'
                     metrics=metric)  # 'crossentropy' #  'mse' # loss # accuracy

    # es
    es = EarlyStopping(monitor='val_mse_metric', mode='min', verbose=verbose, patience=patience, min_delta=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_mse_metric', mode='min', verbose=verbose, save_best_only=True)

    #  # .transpose(0,2, 1)  # expand_dims(data, axis=2) ### moveaxis(data, 2, 0) # .reshape(X_learn.shape[0], 7, 14,1)
    #  #  pts_col  #  os_col  #  drug_col
    print('glob_x_train.shape 2 ', glob_x_train.shape)
    history = ae_model.fit(x=glob_x_train,
                           y=glob_x_train,
                           validation_data=(glob_x_val
                                            , glob_x_val),  # .transpose(0,2, 1)
                           # validation_split = 0.2, # validation_data = validation_generator
                           # validation_steps=50,
                           # steps_per_epoch=100,
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=verbose,
                           callbacks=[es, mc]  # ,
                           # class_weight=class_weights
                           )
    # print("Evaluate") eval__
    ae_model = load_model('best_model.h5')

    pred_ae = ae_model.predict(glob_x_test)
    acc = sk.metrics.mean_squared_error(glob_x_test, pred_ae)
    print("final mse",acc)
    if final_model_ae:
        encoded_train = encoder.predict(glob_x_train)  # .transpose(0,2, 1)
        encoded_val = encoder.predict(glob_x_val)
        encoded_test = encoder.predict(glob_x_test)


    ##### <<<<<<
    durations_test, events_test = glob_y_test



    return({'loss': acc, 'status': STATUS_OK, 'test':['test'], 'acc':acc, 'model':encoder, 'encoded_train':encoded_train, 'encoded_val':encoded_val, 'encoded_test':encoded_test}) # 'log':log,
########################################################################################################################
def set_global_data(df_train,df_test):

    global glob_x_train, glob_x_val, glob_x_test
    global glob_y_train, glob_y_val, glob_y_test
    global labtrans

    # categorical
    # pa3cat : [1 0 2] # poor intermeiate ideal
    # nutrition3cat: [1 0 2] # poor intermeiate ideal
    # fmlyinc: [2 1 4 3]
    # TODO: occupation: [ 1.3.2.6.5.11.9.7.2.58509567 4.12.8.] # 12 jobs
    # TODO: edu3cat: [2 0 1] # high school, less than high school ...    #
    # TODO: privatepublicins: [2 3 1 0] # none, private, pulic, both

    # not found: s0pai, s0opai
    print("\npts onehot")  # .apply(lambda x:1 if i<100 else 2 if i>100 else 0)
    df_full = pd.concat([df_train, df_test], axis=0)
    for col in ['occupation', 'edu3cat', 'privatepublicins']:  # use column name as prefix
        cats= sorted(df_full[col].unique())
        # print(cats)
        onehot = pd.get_dummies(df_train[col].astype(pd.CategoricalDtype(categories=cats)), drop_first=False, prefix=str(col))
        # print(onehot.columns)
        df_train = pd.concat([df_train, onehot], axis=1)
        df_train = df_train.drop([col], axis=1)
        onehot = pd.get_dummies(df_test[col].astype(pd.CategoricalDtype(categories=cats)), drop_first=False,prefix=str(col))
        # print(onehot.columns)
        df_test = pd.concat([df_test, onehot], axis=1)
        df_test = df_test.drop([col], axis=1)
    # exit(0)

    df_val = df_train.sample(frac=0.2) # change from 0.3
    df_train = df_train.drop(df_val.index)


    cols_standardize = df_train.columns.values.tolist()
    # print(cols_standardize)
    cols_standardize.remove('duration')
    cols_standardize.remove('event')  # covariates_all
    global final_features
    final_features = cols_standardize
    # print(cols_standardize)
    standardize = [([col], MinMaxScaler()) for col in cols_standardize] # StandardScaler
    cols_leave = [x for x in df_train.columns.values if x not in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize)  # + leave
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')

    num_durations = 10
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))
    y_test = labtrans.transform(*get_target(df_test))

    glob_x_train, glob_x_val, glob_x_test = x_train, x_val, x_test
    glob_y_train, glob_y_val, glob_y_test = y_train, y_val, y_test
########################################################################################################################
def modeling(covariates_all,outcomes):

    global glob_x_train, glob_x_val, glob_x_test
    global glob_y_train, glob_y_val, glob_y_test
    global labtrans
    global final_model_ae


    print_log("==================")
    dataLocation = "C:\\Users\\hrmor\\OneDrive - University of Mississippi Medical Center\\03_UMMC\\Projects__Jackson heart\\JHS data (Moradi-Morris)\\"
    resultsLocation = ""
    if platform.system() != 'Windows':
        dataLocation = "/home/hmoradi/Downloads/Data/JHS/"
        resultsLocation = "/home/hmoradi/Downloads/PycharmProject/JacksonHeart/"
    jhs_analysis = pd.read_csv(
        dataLocation + "Data_set_hf" + ".csv")  # Data_set_hf_rfImpute-True # _rfImpute-True # Data_set_hf_10y-False # Data_set_hf
    # for d in ["chd_date","hf_date","st_date"]:  # "visitdate",
    #     jhs_analysis[d] = pd.to_datetime(jhs_analysis[d], infer_datetime_format=True)
    covariates_all = [x.strip().lower() for x in covariates_all]

    # categorical
    for var in covariates_all:
        items = jhs_analysis[var].unique()
        if len(items) < 20:
            print(var, ":", str(items))
            # pa3cat : [1 0 2] # poor intermeiate ideal
            # nutrition3cat: [1 0 2] # poor intermeiate ideal
            # TODO: fmlyinc: [2 1 4 3]
            # TODO: occupation: [ 1.3.2.6.5.11.9.7.2.58509567 4.12.8.] # 12 jobs
            # TODO: edu3cat: [2 0 1] # high school, less than high school ...    #
            # TODO: privatepublicins: [2 3 1 0] # none, private, pulic, both

            # not found: s0pai, s0opai
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FROM ENCLAVE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # https: // nbviewer.jupyter.org / github / havakv / pycox / blob / master / examples / deephit.ipynb
    # exit()

    start = datetime.now()
    print_log("datetime.now()", datetime.now())

    jhs_analysis = jhs_analysis.rename(columns={'outcome': 'duration', 'any_inc': 'event'})

    #### >>>>>
    df_train = jhs_analysis.copy()
    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)

    # global glob_x_train, glob_x_val, glob_x_test
    # global glob_y_train, glob_y_val, glob_y_test
    # global labtrans
    set_global_data(df_train, df_test)

    start = datetime.now()
    print_log("datetime.now()", datetime.now())



    #### >>>>>

    max_eval=100
    optimize = True

    # optimize = True



    # __param
    num_layers = hp.choice('num_layers', [1, 2, 3]) #, 4, 5])
    num_neurons = hp.choice('num_neurons', [2,4,6,8,10,12,14,16,18,20])

    batch_norm = hp.choice('batch_norm', [True, False])
    dropout = hp.choice('dropout', [0.1, 0.2, 0.3, 0.4, 0.5]) # 0.1
    learning_rate = hp.choice('learning_rate', [0.1, 0.01, 0.001, 0.0001])  # 0.01
    batch_size = hp.choice('batch_size', [4, 16, 32, 64])
    epochs = hp.choice('epochs', [4, 16, 32, 64, 128, 256, 512, 1024]) # 100

    # momentu = param_grid['']
    # actiovation_func = param_grid['']
    # loss =  param_grid['']
    # optimizer = param_grid['']

    param_grid = dict(num_layers=num_layers,
                      num_neurons=num_neurons,
                      batch_norm=batch_norm, dropout=dropout,
                      learning_rate=learning_rate, batch_size=batch_size,
                      epochs=epochs)
    # print_log('param_grid: ', param_grid)

    tpe_trials = Trials()
    tpe_best = []
    model = []



    # np.savez("Reduce-Dim//dim_reduced.npz", glob_x_train=glob_x_train,glob_x_val=glob_x_val,glob_x_test=glob_x_test,
    #             glob_y_train = glob_y_train, glob_y_val = glob_y_val, glob_y_test = glob_y_test)

    # files = dict(glob_x_train=glob_x_train, glob_x_val=glob_x_val, glob_x_test=glob_x_test,
    #              glob_y_train=glob_y_train, glob_y_val=glob_y_val, glob_y_test=glob_y_test)
    npzfile = np.load("Reduce-Dim//dim_reduced.npz")

    glob_x_train = npzfile['glob_x_train']
    glob_x_val = npzfile['glob_x_val']
    glob_x_test = npzfile['glob_x_test']

    for file in npzfile.files: # for k, v in files.items():
        print(file)
    #     # files[file]= 0
    # for file in npzfile.files: # for k, v in files.items():
    #     files[file]= npzfile[file]
    #     print(npzfile[file].shape)

    print('\nglob_x_train.shape final ', glob_x_train.shape)
    print('glob_x_val.shape final ', glob_x_val.shape)
    print('glob_x_test.shape final ', glob_x_test.shape)
    print('\nglob_y_train.shape final ', glob_y_train[:3])
    print('glob_y_train.shape final ', glob_y_train[0].shape)
    print('glob_y_val.shape final ', glob_y_val[1].shape)
    print('glob_y_test.shape final ', glob_y_test[0].shape)

    if optimize:
        tpe_best = fmin(fn=eval_model, space=param_grid, algo=tpe.suggest, max_evals=max_eval, trials=tpe_trials,
                    rstate=np.random.RandomState(14))
        param_grid= space_eval(param_grid, tpe_best)
        print_log("Best: ", tpe_best, getXfromBestModelfromTrials(tpe_trials, 'acc'), param_grid)
        model = getXfromBestModelfromTrials(tpe_trials, 'model')

        print_log("datetime.now(): ", datetime.now())
        seconds_in_day = 24 * 60 * 60
        difference = datetime.now() - start
        print_log("Pased (hours): ", divmod(difference.days * seconds_in_day + difference.seconds, 60)  )
    else:
        param_grid = {'batch_norm': True, 'batch_size': 64, 'dropout': 0.4, 'epochs': 128, 'learning_rate': 0.001, 'num_layers': 4, 'num_neurons': 50}
        model = eval_model(param_grid)['model']
        # {'batch_norm': True, 'batch_size': 16, 'dropout': 0.5, 'epochs': 128, 'learning_rate': 0.01, 'num_layers': 1, 'num_neurons': 4} #  70


    # # global glob_x_train, glob_x_val, glob_x_test
    # # global glob_y_train, glob_y_val, glob_y_test
    # # global labtrans
    # set_global_data(df_train, df_test)





    exit()


    return()
####################################################################################################################################
if __name__ == '__main__':
    ############################################################################
    covariates_main = ["subjid","visitdate",
                       "age","waist","BMI","sex", # 4
                      "sbp","dbp","abi","ldl","hdl","trigs","FPG","HbA1c", # 8
                      "BPmeds","totchol","alc","alcw","currentSmoker","everSmoker","PA3cat","nutrition3cat","activeIndex","depression","weeklyStress","perceivedStress", # 12
                      "fmlyinc","occupation","edu3cat","dailyDiscr","lifetimeDiscrm","discrmBurden", # 6
                      "Insured","PrivatePublicIns", # 2
                      "MIHx", "CHDHx","strokeHx", # , # 4
                      "CVDHx", "CardiacProcHx" ]
                      # , # 5-3 # "CHDHx","StrokeHx","MIHx",
                      # "MHXB","MHXC", #  "HFHx" <<<<<< outcome
                      # "HFHx",  # event  <<<<<< outcome
                      #"cesfdate","hfevdate","cevtdat3", # time to event  9
                      #"ddate","ddate0","dthdate","hfaa31a","hfaa32a","year"
    print_log("Find Main Covariates:")
    # find_covariates(covariates_main)
    ###########################################################################
    dataLocation = "C:\\Users\\hrmor\\OneDrive - University of Mississippi Medical Center\\03_UMMC\\Projects__Jackson heart\\JHS data (Moradi-Morris)\\"
    if platform.system() != 'Windows':
        dataLocation = "/home/hmoradi/Downloads/Data/JHS/"
    env_var = pd.read_csv(dataLocation + "env_var" + ".csv",header=None)
    covariates_env =  env_var.values.flatten()
    # fake_stcotrk --->>> replaced with --->>> FakeCensusTractID
    print_log("\nFind Env. Covariates:")
    # find_covariates(covariates_env)
    ############################################################################
    covariates_all=[]
    for x in covariates_main:
        covariates_all.append(x)
    for x in covariates_env:
        covariates_all.append(x)
    print_log("\nFeature Collector:")
    # feature_collector(covariates_all)
    ############################################################################
    covariates_all = [x.strip().lower() for x in covariates_all]
    for x in ['mihx','chdhx','strokehx','cvdhx','cardiacprochx']:
        covariates_all.remove(x)
    print_log("\nOutcome Collector:")
    # outcome_collector(covariates_all)
    ############################################################################
    print_log("\nModeling:")
    for x in ['visitdate', 'exam', 'subjid', 'subjid']:  # 'depression', 'weeklystress',  # , 'subjid.2'
        covariates_all.remove(x)
    outcomes=["chd_inc","chd_years", "chd_days","chd_date","chd_year"
                ,"hf_inc", "hf_date", "hf_year", "hf_years", "hf_days"
                ,"st_inc", "st_date", "st_year", "st_years", "st_days"]
    modeling(covariates_all,outcomes)
####################################################################################################################################