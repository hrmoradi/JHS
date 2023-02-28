import pandas as pd
# from tabulate import tabulate
import numpy as np
from datetime import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# seaborn: added to env.
# Keras: added to env.
# hyperopt: added to env.
# ADD ? imbalance-learn
import platform

import torch
import time
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

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval  # rand, GridSearch

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn_pandas import DataFrameMapper
from lifelines.utils import k_fold_cross_validation

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
# warnings.filterwarnings('ignore')
from datetime import datetime

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
from tensorflow import keras
import platform
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
np.random.seed(7)
tf.random.set_seed(12)
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import cross_val_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
print("tf", tf.__version__) # 2.5.0
print("np",np.__version__) # 1.19.5
print("pd",pd.__version__) # 1.2.4
print("sk",sk.__version__) # 0.24.2
print("keras",keras.__version__) # 2.5.0
print("torch",torch.__version__) # 2.5.0
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

import pandas as pd
from tabulate import tabulate
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from os import path
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval  # rand, GridSearch
from datetime import datetime
import lifelines
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import torch
import torchtuples as tt
import pycox
from pycox.datasets import metabric
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
# import bokeh as bk
import shap
import platform
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
np.random.seed(7)
tf.random.set_seed(12)
_ = torch.manual_seed(123)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF\n")
if not torch.cuda.is_available():
    print("Please install GPU version of Torch\n")