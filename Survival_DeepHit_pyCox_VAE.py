import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval  # rand, GridSearch
from datetime import datetime
import lifelines
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
import pycox
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import bokeh as bk
import platform
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(K.image_data_format())
np.random.seed(7)
tf.random.set_seed(12)
_ = torch.manual_seed(123)
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

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

########################################################################################################################
glob_x_train, glob_x_val, glob_x_test = [],[],[]
glob_y_train, glob_y_val, glob_y_test = [],[],[]
final_features = []
labtrans= []
final_model = False
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
def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.shape(distribution_variance)[0]
    random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(0.5 * distribution_variance) * random
########################################################################################################################
def get_loss(distribution_mean, distribution_variance):
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch #* 28 * 28 ??????

    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - tf.square(distribution_mean) - tf.exp(
            distribution_variance)
        kl_loss_batch = tf.reduce_mean(kl_loss)
        return kl_loss_batch * (-0.5)

    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        return reconstruction_loss_batch + kl_loss_batch

    return total_loss
########################################################################################################################
def eval_ae(param_grid):
    global glob_x_train, glob_x_val, glob_x_test
    global glob_y_train, glob_y_val, glob_y_test
    global labtrans
    global final_model_ae

    print('TEST__ param_grid:', param_grid)
    in_features = glob_x_train.shape[1]
    num_layers = param_grid['num_layers']
    num_neurons = param_grid['num_neurons']
    encode_dim = param_grid['encode_dim']
    batch_norm = param_grid['batch_norm']
    drop_out = param_grid['drop_out']
    dropout = param_grid['dropout']  # 0.1
    learning_rate = param_grid['learning_rate'] # 0.01
    batch_size = param_grid['batch_size']
    epochs = param_grid['epochs'] # 100
    verbose = 0

    # __vae
    input_en = Input(shape=(in_features,))
    l = 'En-1'
    encoder = Dense(num_neurons, activation="relu", name="Fully-con-" + str(l))(input_en)
    if drop_out:
        encoder = Dropout(dropout, name="Drop-" + str(l))(encoder)
    if batch_norm:
        encoder = BatchNormalization(axis=1, name=("BatchNorm-" + str(l)))(encoder)
    stack = []
    stack.append(num_neurons)
    for lay in range(2,num_layers+1,1):
        if num_neurons//lay < encode_dim: break
        stack.append(num_neurons//lay)
        l = 'En-'+str(lay)
        encoder = Dense(num_neurons//lay, activation="relu", name="Fully-con-" + str(l))(encoder)
        if drop_out:
            encoder = Dropout(dropout, name="Drop-" + str(l))(encoder)
        if batch_norm:
            encoder = BatchNormalization(axis=1, name=("BatchNorm-" + str(l)))(encoder)

    distribution_mean = tf.keras.layers.Dense(encode_dim, name='mean')(encoder)
    distribution_variance = tf.keras.layers.Dense(encode_dim, name='log_variance')(encoder)
    latent_encoding = tf.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])

    encoder_model = Model(inputs=input_en, outputs=latent_encoding)
    # encoder_model.summary()


    input_de = Input(shape=(encode_dim,))
    l = 'De-' + str(len(stack))
    decoder = Dense(stack.pop(), activation="relu", name="Fully-con-" + str(l))(input_de)
    if drop_out:
        decoder = Dropout(dropout, name="Drop-" + str(l))(decoder)
    if batch_norm:
        decoder = BatchNormalization(axis=1, name=("BatchNorm-" + str(l)))(decoder)
    en_layers = len(stack)
    for lay in range(en_layers,0,-1):
        l = 'De-' + str(lay)
        decoder = Dense(stack.pop(), activation="relu", name="Fully-con-" + str(l))(decoder)
        if drop_out:
            decoder = Dropout(dropout, name="Drop-" + str(l))(decoder)
        if batch_norm:
            decoder = BatchNormalization(axis=1, name=("Batch-Norm-" + str(l)))(decoder)
    l = 'Decoder-out'
    decoder = Dense( in_features, activation="sigmoid", name="Fully-con-" + str(l))(decoder)

    decoder_model = Model(inputs=input_de, outputs=decoder)
    # decoder_model.summary()

    encoded = encoder_model(input_en)
    decoded = decoder_model(encoded)
    autoencoder = tf.keras.models.Model(input_en, decoded)
    # autoencoder.summary()

    metric = tf.keras.metrics.MeanSquaredError(name="mse_metric", dtype=None) # metrics=metric
    autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance),
                        optimizer=Adam(learning_rate=learning_rate),
                        experimental_run_tf_function=False)

    autoencoder.fit(x=glob_x_train, y=glob_x_train,
                    validation_data=(glob_x_val, glob_x_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose)

    pred_ae = autoencoder.predict(glob_x_test)
    if np.isnan(pred_ae).any():
        print("ERROR np.nan_to_num(pred_ae)")
        pred_ae = np.nan_to_num(pred_ae)
    loss =  sk.metrics.mean_squared_error(glob_x_test, pred_ae)
    print("Final VAE mse: ",loss)

    encoded_train = encoder_model.predict(glob_x_train)  # .transpose(0,2, 1)
    encoded_val = encoder_model.predict(glob_x_val)
    encoded_test = encoder_model.predict(glob_x_test)

    return({'loss': loss, 'status': STATUS_OK, 'test':['test'], 'acc':loss, 'model':encoder, 'encoded_train':encoded_train, 'encoded_val':encoded_val, 'encoded_test':encoded_test}) # 'log':log,
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
    jhs_analysis = pd.read_csv(dataLocation + "Data_set_hf" + ".csv") # Data_set_hf_rfImpute-True # _rfImpute-True # Data_set_hf_10y-False # Data_set_hf
    jhs_analysis = jhs_analysis.rename(columns={'outcome': 'duration', 'any_inc': 'event'})
    covariates_all = [x.strip().lower() for x in covariates_all]

    start = datetime.now()
    print_log("datetime.now()", datetime.now())

    #### >>>>>
    df_train = jhs_analysis.copy()
    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)
    set_global_data(df_train, df_test)

    max_eval=200
    optimize_ae = False

    # __param
    num_layers = hp.choice('num_layers', [1, 2, 3,4,5]) #, 4, 5])
    num_neurons = hp.choice('num_neurons', [25, 50, 75,90, 100,120,150])
    encode_dim = hp.choice('encode_dim', [2,3,5,7,10,20,25,30,40,50,70])
    batch_norm = hp.choice('batch_norm', [True, False])
    drop_out = hp.choice('drop_out', [True, False])
    dropout = hp.choice('dropout', [0.1, 0.2, 0.3, 0.4, 0.5]) # 0.1
    learning_rate = hp.choice('learning_rate', [0.1, 0.01, 0.001, 0.0001])  # 0.01
    batch_size = hp.choice('batch_size', [4, 16, 32, 64,128])
    epochs = hp.choice('epochs', [4, 16, 32, 64, 128, 256, 512, 1024]) # 100

    param_grid_ae = dict(num_layers=num_layers, drop_out=drop_out,
                      num_neurons=num_neurons, encode_dim=encode_dim,
                      batch_norm=batch_norm, dropout=dropout,
                      learning_rate=learning_rate, batch_size=batch_size,
                      epochs=epochs)

    tpe_trials = Trials()
    tpe_best = []
    model = []
    print('glob_x_train.shape: ', glob_x_train.shape)
    if optimize_ae:
        tpe_best = fmin(fn=eval_ae, space=param_grid_ae, algo=tpe.suggest, max_evals=max_eval, trials=tpe_trials,
                    rstate=np.random.RandomState(14))
        param_grid_ae= space_eval(param_grid_ae, tpe_best)
        print_log("Best: ", getXfromBestModelfromTrials(tpe_trials, 'acc'), param_grid_ae)
        model = getXfromBestModelfromTrials(tpe_trials, 'model')

        print_log("datetime.now(): ", datetime.now())
        seconds_in_day = 24 * 60 * 60
        difference = datetime.now() - start
        print_log("Pased (hours): ", divmod(difference.days * seconds_in_day + difference.seconds, 60)  )
    else:
        param_grid_ae = {'batch_norm': True, 'batch_size': 32, 'drop_out': True, 'dropout': 0.2, 'encode_dim': 25, 'epochs': 128, 'learning_rate': 0.01, 'num_layers': 2, 'num_neurons': 50} # best loss: 0.04293834

    final_model_ae =  True
    encoded = eval_ae(param_grid_ae)
    print('glob_x_train.shape final ', glob_x_train.shape)
    print('glob_x_val.shape final ', glob_x_val.shape)
    print('glob_x_test.shape final ', glob_x_test.shape)

    glob_x_train = encoded['encoded_train']
    glob_x_val = encoded['encoded_val']
    glob_x_test = encoded['encoded_test']

    print("\nnp.savez")
    np.savez("Reduce-Dim//dim_reduced.npz", glob_x_train=glob_x_train, glob_x_val=glob_x_val, glob_x_test=glob_x_test)

    print('\nglob_x_train.shape final ', glob_x_train.shape)
    print('glob_x_val.shape final ', glob_x_val.shape)
    print('glob_x_test.shape final ', glob_x_test.shape)

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
    dataLocation = "C:\\Users\\hrmor\\OneDrive - University of Mississippi Medical Center\\03_UMMC\\Projects__Jackson heart\\JHS data (Moradi-Morris)\\"
    if platform.system() != 'Windows':
        dataLocation = "/home/hmoradi/Downloads/Data/JHS/"
    env_var = pd.read_csv(dataLocation + "env_var" + ".csv",header=None)
    covariates_env =  env_var.values.flatten()
    covariates_all=[]
    for x in covariates_main:
        covariates_all.append(x)
    for x in covariates_env:
        covariates_all.append(x)
    covariates_all = [x.strip().lower() for x in covariates_all]
    for x in ['mihx','chdhx','strokehx','cvdhx','cardiacprochx']:
        covariates_all.remove(x)
    ############################################################################
    print_log("\nModeling:")
    for x in ['visitdate', 'exam', 'subjid', 'subjid']:  # 'depression', 'weeklystress',  # , 'subjid.2'
        covariates_all.remove(x)
    outcomes=["chd_inc","chd_years", "chd_days","chd_date","chd_year"
                ,"hf_inc", "hf_date", "hf_year", "hf_years", "hf_days"
                ,"st_inc", "st_date", "st_year", "st_years", "st_days"]
    modeling(covariates_all,outcomes)
####################################################################################################################################