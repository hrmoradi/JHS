from Param_JHS import *
from Libraries_JHS import *

import Survival_Reader_me as reader_raw
import Survival_Lifelines as lifeline_rf
import Survival_Sksurv as sksurv_cox
import Survival_DeepHit_pyCox as deep_hit
import get_stat as get_stat

if recreate_input:
    reader_raw.main()
if get_info:
    get_stat.main()
    print("\nget_stat.main()")
    exit()
# if algorithm == 'lifeline_rf':
#     lifeline_rf.main()
# exit()
if algorithm == 'sksurv_cox_cox' or algorithm == 'sksurv_cox_rf':
    sksurv_cox.main()
if algorithm == 'deep_hit':
    deep_hit.main()


