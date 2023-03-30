import platform

# PC:
# UMC0221026 # .ntummc.umsmed.edu
# Unix:
# hmoradi@umc0195009 # 10.57.43.7  # umc0195009 # umc0195009.umsmed.edu
# gnome-session-quit --force
# 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
# screen -xS hamid # screen -S hamid
# Ctrl+a  "then" Esc
# watch nvidia-smi
# source venv/bin/activate
# alt + f2 : xkill  or gnome-system-monitor (set as alt+del) # sudo service lightdm restart

# ToDo: Get some stat
get_info = False
eval_correlation = False

# ToDO: Survival_DeepHit_pyCox as deep_hit
calc_shap = False # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
graph_shap = True # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
kmeans_k =100
rows_devideby_to_use = 1

optimize = False # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
max_eval = 50

k_fold_croos_validation = 10

# ToDo Survival_Sksurv as

# feature_set = "main" # "main" # "psycho" # "all" # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# feature_set = "psycho"
feature_set = "all"

# ToDO: Survival_Reader_me as reader_modeler_raw # ToDo: mainly input reader
recreate_input = False               # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
### algorithm = 'lifeline_rf'
# algorithm = 'sksurv_cox_rf'            # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< with all deep 76
# algorithm = 'sksurv_cox_cox'
algorithm = 'deep_hit'
#
ptr_debug = True
ptr_debug_outcome_func = False
#
impute_rf = False
ptr_imputer = False
with_hf = 'with' # with # no            features
max_incident_years = 10                 # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#.isna().sum(axis=1).values<individual_miss_threshold
individual_miss_threshold = 166 # 166    # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# if len(sub_set_column_to_drop_rows)>=1:
sub_set_column_to_drop_rows = [] # ['weeklystress','depression'] # if len() >=1  # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

incident_long_after = False
name = "DataSet_"+with_hf+"HF-" +str(max_incident_years)+'y-impRf'+str(impute_rf)+'-RowThreshold'+str(individual_miss_threshold)+"-ColSpec"+str(len(sub_set_column_to_drop_rows))

# ToDo: File locations ##############################################################################################
dataLocation = "C:\\Users\\hrmor\\OneDrive - University of Mississippi Medical Center\\03_Projects\\Proj__Matthew\\JHS data (Moradi-Morris)\\"
resultsLocation =""
if platform.system() != 'Windows':
    dataLocation = "/home/hmoradi/Downloads/Data/JHS/"
    resultsLocation = "/home/hmoradi/Downloads/PycharmProject/JacksonHeart/Results/"
logFile = open(dataLocation+'Result_Logfile.txt', 'a')
# ToDo: features ####################################################################################################3#
main_features =         ["age","waist","BMI","sex",
                      "sbp","dbp","abi","ldl","hdl","trigs","FPG","HbA1c",
                      "BPmeds","totchol","alc","alcw","currentSmoker","everSmoker","PA3cat","nutrition3cat","activeIndex" ]
main_features = [x.strip().lower() for x in main_features]
psychosocial_features = ["depression","weeklyStress","perceivedStress",
                        "fmlyinc","occupation","edu3cat","dailyDiscr","lifetimeDiscrm","discrmBurden",
                        "Insured","PrivatePublicIns" ]
psychosocial_features = [x.strip().lower() for x in psychosocial_features]
outcome_features =      ['outcome', 'any_inc']
outcome_features = [x.strip().lower() for x in outcome_features]
# env_features = pd.read_csv(dataLocation + "env_var" + ".csv",header=None)
# env_features =  env_features.values.flatten()

# ToDo: Comments ####################################################################################################3#
# with_hf # years # threshold # column # incident # rows
#   T     #   10  #     166   #   0    #  382     #  3980
#   T     #   15  #     166   #   0    #  488     #  3980
#   T     #   15  #     16    #   0    #  483     #  3950
#   T     #   15  #     10    #   0    #  450     #  3572
#   T     #   15  #     5     #   0    #  374     #  3073 ___
#   F     #   10  #     166   #   0    #  290     #  4396
#   F     #   15  #     166   #   0    #  389     #  4396
#   F     #   15  #     16    #   0    #  385     #  4360
#   F     #   15  #     10    #   0    #  354     #  3928
#   F     #   15  #     5     #   0    #  291     #  3360
#   F     #   15  #     5     #   2    #  142     #  1928
# ToDo: feature_collector(covariates_all) # "Features" + ".csv"
# find all covariates in the files then -> # feature_collector(covariates_all)
# male_female one column # limited jhs_analysis file to visit one 5306 # removed all with history 4566 #
# left joined all the other tables with limiting to exam1 # remove histories(Hx) from covariates_all #
#     abi  ldl  hdl  trigs  fpg  hba1c  totchol  nutrition3cat  depression  weeklystress  fmlyinc  lifetimediscrm  discrmburden  pcom0  pcom1  pcom14  pres0  pres1  pres14  pret0  pret1  pret14
# 0  525  402  366    365  353    172      365            444        1556          1960      684             166           826    673    656     675    673    656     675    673    656     675
# ToDo: outcome_collector(covariates_all)  #
# v2 4 years later  # v3 8 years later
# chd -> stroke -> hf # chd_inc 1-0-nan #
#columns={"stroke": "st_inc", "date": "st_date", "year": "st_year", "years": "st_years", "days": "st_days", 'status': "st_status"}
# Setting Incident to Min year, No-incident to Max year