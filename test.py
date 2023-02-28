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

import sksurv # scikit-survival
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
print("sksurv",sksurv.__version__) # 0.15.0.post0  # scikit-survival
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