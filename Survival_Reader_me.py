from Param_JHS import *
from Libraries_JHS import *

y_val = []
X_val = []
y_learn = []
X_learn = []
####################################################################################################################################
def find_covariates(covariates):
    print("==================")

    jhs_analysis = pd.read_csv(dataLocation+"jhs_analysis"+".csv")
    jhs_analysis_new = pd.read_csv(dataLocation+"jhs_analysis_new"+".csv")
    jhs_be = pd.read_csv(dataLocation + "jhs_be" + ".csv")
    jhs_fc = pd.read_csv(dataLocation + "jhs_fc" + ".csv")
    jhs_imps = pd.read_csv(dataLocation + "jhs_imps" + ".csv")
    jhs_nb_long = pd.read_csv(dataLocation + "jhs_nb_long" + ".csv")
    jhs_nb_cen = pd.read_csv(dataLocation + "jhs_nb_cen" + ".csv")
    jhs_nets = pd.read_csv(dataLocation + "jhs_nets" + ".csv")
    jhs_park = pd.read_csv(dataLocation + "jhs_park" + ".csv")
    incevtchd = pd.read_csv(dataLocation + "incevtchd" + ".csv")
    incevthfder = pd.read_csv(dataLocation + "incevthfder" + ".csv")
    incevtstroke = pd.read_csv(dataLocation + "incevtstroke" + ".csv")
    file_dict = {"jhs_analysis":jhs_analysis, "jhs_analysis_new":jhs_analysis_new, "jhs_be":jhs_be, "jhs_fc":jhs_fc, \
                 "jhs_imps":jhs_imps, "jhs_nb_long":jhs_nb_long,"jhs_nb_cen":jhs_nb_cen, "jhs_nets":jhs_nets, "jhs_park":jhs_park, \
                 "incevtchd":incevtchd, "incevthfder":incevthfder, "incevtstroke":incevtstroke}

    covariates = [x.strip().lower() for x in covariates] ### Lower Case
    Found = ""
    overall_found=""
    Not_Found = ""
    for file_name, file in file_dict.items():
        file.columns = file.columns.str.strip().str.lower() ### Lower Case
        for var in covariates:
            try:
                get = file[var]
                Found += ("\""+var+"\",")+" "
            except:
                #print("\""+var+"\""+", ", end="")
                if var not in Not_Found:
                    Not_Found += ("\""+var+"\",")+" "

        if Found != "":
            print(file_name,": \n\t\t\t",Found)
            overall_found+=Found
            Found = ""

    for var in covariates:
        if ("\""+var+"\",") in overall_found:
            Not_Found=Not_Found.replace( ("\""+var+"\",") ,"")
    if ptr_debug: print("Not Found: \n\t\t\t",Not_Found.strip().split(","))

    if ptr_debug: print("Files checked:  \n\t\t\t", end="")
    for file_name, file in file_dict.items():
        if ptr_debug: print(file_name," ",end="")
    if ptr_debug: print("\nlen(covariates_all): ", len(covariates))
    if ptr_debug: print("len(overall_found): ", len(overall_found.strip().split("\", ")))
    if ptr_debug: print("len(Not_Found): ", len(Not_Found.strip().split(",")))

    if ptr_debug: print("Any Issue: ")
    for var in covariates:
        if ( ("\""+var+"\",") in overall_found) and ( ("\""+var+"\",") in Not_Found):
            print(var," ",end="")
####################################################################################################################################
def feature_collector(covariates_all):
    if ptr_debug: print("==================")

    jhs_analysis = pd.read_csv(dataLocation + "jhs_analysis" + ".csv")
    jhs_analysis_new = pd.read_csv(dataLocation + "jhs_analysis_new" + ".csv")
    jhs_be = pd.read_csv(dataLocation + "jhs_be" + ".csv")
    jhs_fc = pd.read_csv(dataLocation + "jhs_fc" + ".csv")
    jhs_imps = pd.read_csv(dataLocation + "jhs_imps" + ".csv")
    jhs_nb_long = pd.read_csv(dataLocation + "jhs_nb_long" + ".csv")
    jhs_nb_cen = pd.read_csv(dataLocation + "jhs_nb_cen" + ".csv")
    jhs_nets = pd.read_csv(dataLocation + "jhs_nets" + ".csv")
    jhs_park = pd.read_csv(dataLocation + "jhs_park" + ".csv")
    incevtchd = pd.read_csv(dataLocation + "incevtchd" + ".csv")
    incevthfder = pd.read_csv(dataLocation + "incevthfder" + ".csv")
    incevtstroke = pd.read_csv(dataLocation + "incevtstroke" + ".csv")
    file_dict = {"jhs_analysis": jhs_analysis, "jhs_analysis_new": jhs_analysis_new, "jhs_be": jhs_be, "jhs_fc": jhs_fc, \
                 "jhs_imps": jhs_imps, "jhs_nb_long": jhs_nb_long,"jhs_nb_cen": jhs_nb_cen,  "jhs_nets": jhs_nets, "jhs_park": jhs_park, \
                 "incevtchd": incevtchd, "incevthfder": incevthfder, "incevtstroke": incevtstroke}
    covariates_all = [x.strip().lower() for x in covariates_all]  ### Lower Case
    for file_name, file in file_dict.items():
        file.columns = file.columns.str.strip().str.lower() ### Lower Case

    # print("\nFill dates 1/1/1900")
    # file_dict["jhs_analysis"]['visitdate'] = file_dict["jhs_analysis"]['visitdate'].fillna("1/1/1900")
    # print(file_dict["jhs_analysis"].head())

    # print("\nFill na -1 ")
    # file_dict["jhs_analysis"] = file_dict["jhs_analysis"].fillna(-1)
    # print(file_dict["jhs_analysis"].head())

    if ptr_debug: print("\nConvert date")
    # df_platform[['timeStamp']] = df_platform[['timeStamp']].astype('datetime64[ns]')
    file_dict["jhs_analysis"]["visitdate"]= pd.to_datetime(file_dict["jhs_analysis"]["visitdate"],  infer_datetime_format=True) # ,errors='ignore' # ,errors='coerce' # format='%m/%d/%y'
    if ptr_debug: print(file_dict["jhs_analysis"].head())
    # print(file_dict["jhs_analysis"].select_dtypes(include=[np.datetime64]))

    if ptr_debug: print("\nConvert Sex ")
    file_dict["jhs_analysis"]['sex']=file_dict["jhs_analysis"]['sex'].apply(lambda x: 1 if x == 'Male' else 0 if x == 'Female' else np.nan )
    if ptr_debug: print(file_dict["jhs_analysis"].head())
    if ptr_debug: print("df.shape[0]: ", file_dict["jhs_analysis"].shape[0])
    file_dict["jhs_analysis"]["fmlyinc"]= file_dict["jhs_analysis"]["income"]
    # ToDO: Income Status derived from family income and family size and adjusted by interview year of data collection to account for inflation

    if ptr_debug: print("\nGroup subjects get max")
    # print(file_dict["jhs_analysis"].groupby(by='subjid').max().head()) # skipna=True # >>> NaN
    # print(file_dict["jhs_analysis"].groupby(by='subjid').apply(lambda x: x.max(skipna=True)).reset_index().head()) # >>> converts to multiple rows
    # print(file_dict["jhs_analysis"].groupby(by='subjid').agg("max").reset_index().head()) # skipna=True # >>> unexpected keyword argument >>> NaN
    # file_dict["jhs_analysis"] = file_dict["jhs_analysis"].groupby(by='subjid').max() ### this one worked
    file_dict["jhs_analysis"] = file_dict["jhs_analysis"][file_dict["jhs_analysis"]['visit']==1]
    if ptr_debug: print(file_dict["jhs_analysis"].head())
    if ptr_debug: print("df.shape[0]: ", file_dict["jhs_analysis"].shape[0])



    if ptr_debug: print("\nFilter by CVD history: CHDHx,MIHx,CVDHx,CardiacProcHx,StrokeHx")
    HxHistory = [x.lower() for x in ["CHDHx","MIHx","CVDHx","CardiacProcHx","StrokeHx"] ]
    for hist in HxHistory:
        file_dict["jhs_analysis"]= file_dict["jhs_analysis"][ file_dict["jhs_analysis"][ hist ] != 1   ]
    if ptr_debug: print(file_dict["jhs_analysis"].head())
    if ptr_debug: print("df.shape[0]: ", file_dict["jhs_analysis"].shape[0])
    # print(jhs_analysis[['chd_inc', 'hf_inc', 'st_inc']].sum())
    # exit()


    if ptr_debug: print("\nLeft Join: jhs_analysis,jhs_analysis_new:")
    jhs_analysis = file_dict["jhs_analysis"].drop(columns=['fakecensustractid']) # drop to take from new (name overlap
    jhs_analysis = jhs_analysis.merge(file_dict["jhs_analysis_new"], suffixes=('', '_duplicate_jhs_analysis_new'),on='subjid', how='left')
    if ptr_debug: print(jhs_analysis.head())
    if ptr_debug: print("df.shape[0]: ", jhs_analysis.shape[0])
    # TODO: weekly stress and depression have a lot of missing values

    if ptr_debug: print("\njhs_be: subjects get max")
    # file_dict["jhs_be"] = file_dict["jhs_be"].groupby(by='subjid').max()
    file_dict["jhs_be"] = file_dict["jhs_be"][file_dict["jhs_be"]['exam'] == 'exam1' ]
    if ptr_debug: print(file_dict["jhs_be"].head())
    if ptr_debug: print("jhs_be.shape[0]: ", file_dict["jhs_be"].shape[0])

    if ptr_debug: print("\nLeft Join: jhs_be")
    jhs_analysis = jhs_analysis.merge(file_dict["jhs_be"].drop(columns=['exam']),suffixes=('', '_duplicate_jhs_be'), on='subjid', how='left')
    if ptr_debug: print(jhs_analysis.head())

    if ptr_debug: print("\njhs_nb_long: subjects get max")
    # file_dict["jhs_nb_long"] = file_dict["jhs_nb_long"].groupby(by='subjid').max()
    file_dict["jhs_nb_long"] = file_dict["jhs_nb_long"][file_dict["jhs_nb_long"]['exam'] == 'exam1']
    if ptr_debug: print(file_dict["jhs_nb_long"].head())
    if ptr_debug: print("jhs_nb_long.shape[0]: ", file_dict["jhs_nb_long"].shape[0])

    if ptr_debug: print("\nLeft Join: jhs_nb_long")
    jhs_analysis = jhs_analysis.merge(file_dict["jhs_nb_long"].drop(columns=['exam']), suffixes=('', '_duplicate_jhs_nb_long'),on='subjid', how='left')
    if ptr_debug: print(jhs_analysis.head())

    file_dict["jhs_nb_cen"] = file_dict["jhs_nb_cen"][file_dict["jhs_nb_cen"]['exam'] == 'exam1']
    file_dict["jhs_nb_cen"] = file_dict["jhs_nb_cen"][['subjid',"scpca_uebe","vopca_uebe","nppca_uebe"]]
    jhs_analysis = jhs_analysis.merge(file_dict["jhs_nb_cen"], suffixes=('', '_duplicate_jhs_nb_cen'),on='subjid', how='left')

    if ptr_debug: print("\njhs_nets: subjects get max")
    # file_dict["jhs_nets"] = file_dict["jhs_nets"].groupby(by='subjid').max()
    file_dict["jhs_nets"] = file_dict["jhs_nets"][file_dict["jhs_nets"]['exam'] == 'exam1']
    if ptr_debug: print(file_dict["jhs_nets"].head())
    if ptr_debug: print("jhs_nets.shape[0]: ", file_dict["jhs_nets"].shape[0])

    if ptr_debug: print("\nLeft Join: jhs_nets")
    jhs_analysis = jhs_analysis.merge(file_dict["jhs_nets"].drop(columns=['exam']),suffixes=('', '_duplicate_jhs_nets'), on='subjid', how='left')
    if ptr_debug: print(jhs_analysis.head())

    if ptr_debug: print("\nFilter Required Variables")
    covariates_all.remove('exam')
    for x in ['mihx','chdhx','strokehx','cvdhx','cardiacprochx']:
        covariates_all.remove(x)
    print(str(list(jhs_analysis.columns)))
    # exit()
    jhs_analysis = jhs_analysis[covariates_all]
    if ptr_debug: print(jhs_analysis.head())


    if ptr_debug: print("\njhs_analysis: \n", jhs_analysis.describe())
    if ptr_debug: print("jhs_analysis missing: \n", pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values))
    missing = pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values)
    missing128 = missing.loc[: , missing.apply(lambda s: s > 128).to_numpy().flatten()] # apply(lambda s: s > 128).all() if s.nunique() < 3 else s.nunique() > 1)
    print("\njhs_analysis 128: \n", missing128)
    print("jhs_analysis: ", jhs_analysis.shape)
    # exit()
    print("jhs_analysis: ", jhs_analysis.dropna().shape)

    jhs_analysis.columns = jhs_analysis.columns.str.strip().str.lower()
    jhs_analysis.to_csv(dataLocation + "Features" + ".csv",index=False)


    return()
####################################################################################################################################
def outcome_collector(covariates_all):
    print("==================")

    jhs_analysis = pd.read_csv(dataLocation + "Features" + ".csv")
    jhs_analysis["visitdate"] = pd.to_datetime(jhs_analysis["visitdate"],infer_datetime_format=True)
    covariates_all = [x.strip().lower() for x in covariates_all]

    # max values observerd are collected
    # TODO: Filter by CVD history: CHDHx,MIHx,CVDHx,CardiacProcHx,StrokeHx"
    # TODO: weekly stress and depression have a lot of missing values - jhs_analysis,jhs_analysis_new
    # TODO: jhs_be, jhs_nb_long, jhs_nets: subjects get max

    # TODO: jhs_:: 5307  _nets::9505/2=4752 _nb_cen::9505/2=4752  ev-chd:: 5307
    # TODO: Finally: jhs_analysis:  (4228, 171) >>> filtered by history >>> jhs_analysis:  (1085, 171) >>>

    if ptr_debug_outcome_func: print("jhs_analysis: \n", jhs_analysis.head())
    if ptr_debug_outcome_func: print("\njhs_analysis: \n", jhs_analysis.describe())
    if ptr_debug_outcome_func: print("jhs_analysis: \n", pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values))
    missing = pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values)
    missing128 = missing.loc[:, missing.apply(lambda s: s > 128).to_numpy().flatten()]  # apply(lambda s: s > 128).all() if s.nunique() < 3 else s.nunique() > 1)
    if ptr_debug_outcome_func: print("jhs_analysis 128: \n", missing128)
    if ptr_debug_outcome_func: print("jhs_analysis: ", jhs_analysis.shape)
    if ptr_debug_outcome_func: print("jhs_analysis: ", jhs_analysis.dropna().shape)
    if ptr_debug_outcome_func: print("\n")

    incevtchd = pd.read_csv(dataLocation + "incevtchd" + ".csv")
    incevthfder = pd.read_csv(dataLocation + "incevthfder" + ".csv")
    incevtstroke = pd.read_csv(dataLocation + "incevtstroke" + ".csv")

    # TODO: hard CHD ?
    print("\nincevtchd: ", incevtchd.shape)
    incevtchd.columns = incevtchd.columns.str.strip().str.lower()
    incevtchd = incevtchd.rename(columns={"chd":"chd_inc","date": "chd_date","year":"chd_year","years":"chd_years","days":"chd_days",'status':"chd_status"})
    incevtchd["chd_date"] = pd.to_datetime(incevtchd["chd_date"], infer_datetime_format=True)
    # incevtchd = incevtchd[incevtchd['status'] != 'CHD Hx']
    incevtchd = incevtchd[incevtchd['chd_status'] != 'V1 Medical Record Refusal'] # v2 4 years later  # v3 8 years later  # no info on them
    incevtchd['chd_inc'] = incevtchd['chd_inc'].apply(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else np.nan)
    print("incevtchd drop 'V1 Medical Record Refusal'  : ", incevtchd.shape) # 5307
    # exit()
    # print(incevtchd[incevtchd['chd_days'] != incevtchd['harddays']].head() )
    # print(jhs_analysis.head())
    print("incevtchd NA: \n", pd.DataFrame([incevtchd.isna().sum().values], columns=incevtchd.columns.values))
    # incevtchd = incevtchd[incevtchd['chd_status'] != 'CHD Hx']
    # print("incevtchd 'chd_status != CHD Hx'  : ", incevtchd.shape)
    print("Left Join: incevtchd")
    print("jhs_analysis  : ", jhs_analysis.shape)
    jhs_analysis = jhs_analysis.merge(incevtchd[["subjid","chd_inc","chd_years","chd_status"]], suffixes=('', '_duplicate_incevtchd'),on='subjid', how='left')  # "chd_date", "chd_year", "chd_days",
    jhs_analysis = jhs_analysis[jhs_analysis['chd_status'] != 'CHD Hx']
    jhs_analysis = jhs_analysis[jhs_analysis["chd_inc"] != np.nan]
    print("jhs_analysis drop status 'CHD Hx'  : ", jhs_analysis.shape)
    # jhs_analysis = jhs_analysis.drop(columns=['chd_status'])

    # TODO: stroke
    print("\nincevtstroke: ", incevtstroke.shape)
    incevtstroke.columns = incevtstroke.columns.str.strip().str.lower()
    incevtstroke = incevtstroke.rename(
        columns={"stroke": "st_inc", "date": "st_date", "year": "st_year", "years": "st_years", "days": "st_days",
                 'status': "st_status"})
    incevtstroke["st_date"] = pd.to_datetime(incevtstroke["st_date"], infer_datetime_format=True)
    incevtstroke = incevtstroke[incevtstroke['st_status'] != 'V1 Medical Record Refusal']  # v2 4 years later  # v3 8 years later
    incevtstroke['st_inc'] = incevtstroke['st_inc'].apply(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else np.nan)
    print("incevtstroke drop V1 Medical Record Refusal: ", incevtstroke.shape)  # 5307
    print("incevtstroke NA: \n", pd.DataFrame([incevtstroke.isna().sum().values], columns=incevtstroke.columns.values))
    print("Left Join: incevtstroke")
    print("jhs_analysis  : ", jhs_analysis.shape)
    jhs_analysis = jhs_analysis.merge(
        incevtstroke[["subjid", "st_inc",  "st_years","st_status"]], suffixes=('', '_duplicate_incevtstroke'),on='subjid', # "st_date", "st_year",  "st_days",
        how='left')
    jhs_analysis = jhs_analysis[jhs_analysis['st_status'] != 'Stroke Hx']
    jhs_analysis = jhs_analysis[jhs_analysis["st_inc"] != np.nan]
    print("jhs_analysis drop status 'Stroke Hx'  : ", jhs_analysis.shape)
    jhs_analysis = jhs_analysis.drop(columns=['st_status','chd_status'])


    # TODO: Previous or Uncertain HF ? VA Censored ? Confirmed Deceased ?

    if with_hf=='with':
        print("\nincevthfder: ", incevthfder.shape)
        incevthfder.columns = incevthfder.columns.str.strip().str.lower()
        incevthfder = incevthfder.rename(columns={"hf": "hf_inc", "date": "hf_date", "year": "hf_year", "years": "hf_years", "days": "hf_days",'status':"hf_status"})
        incevthfder["hf_date"] = pd.to_datetime(incevthfder["hf_date"], infer_datetime_format=True)
        incevthfder = incevthfder[incevthfder['hf_status'] != 'V1 Medical Record Refusal']  # v2 4 years later  # v3 8 years later
        print("incevthfder drop V1 Medical Record Refusal: ", incevthfder.shape)  # 5307
        print("incevthfder NA: \n", pd.DataFrame([incevthfder.isna().sum().values], columns=incevthfder.columns.values))
        if ptr_debug_outcome_func: print('incevthfder[examdate] <= incevthfder[date]: ',incevthfder[    incevthfder['examdate'] >= incevthfder['hf_date']  ].shape)
        print("Left Join: incevthfder")
        print("jhs_analysis  : ", jhs_analysis.shape)
        jhs_analysis = jhs_analysis.merge(incevthfder[["subjid", "hf_inc", "hf_years", "hf_status"]],suffixes=('', '_duplicate_incevthfder'),on='subjid', how='left')  # "hf_date", "hf_year", "hf_days",
        jhs_analysis = jhs_analysis[jhs_analysis['hf_status'] != 'Previous or Uncertain HF']
        jhs_analysis = jhs_analysis[jhs_analysis['hf_inc'] != np.nan]
        print("jhs_analysis drop status 'Previous or Uncertain HF'  : ", jhs_analysis.shape)

        jhs_analysis = jhs_analysis.drop(columns=['hf_status'])
    # jhs_analysis = jhs_analysis.drop(columns=['st_status','chd_status'])

    # print("### chd_status\n", jhs_analysis['chd_status'].value_counts())
    # print("### st_status\n", jhs_analysis['st_status'].value_counts())
    # print("### hf_status\n", jhs_analysis['hf_status'].value_counts())
    # print('exit to stop, test data')
    # exit(0)

    if ptr_debug_outcome_func: print("\n***jhs_analysis: \n", jhs_analysis.head())
    if ptr_debug_outcome_func: print("\njhs_analysis: \n", jhs_analysis.describe())
    if ptr_debug_outcome_func: print("jhs_analysis: \n", pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values))
    missing = pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values)
    missing128 = missing.loc[:, missing.apply(lambda s: s > 128).to_numpy().flatten()]  # apply(lambda s: s > 128).all() if s.nunique() < 3 else s.nunique() > 1)
    print("\n\njhs_analysis na>128: \n", missing128)
    print("jhs_analysis: ", jhs_analysis.shape)
    print("jhs_analysis: if drop na", jhs_analysis.dropna().shape)
    print("\n")

    if with_hf=='with':
        missing = pd.DataFrame([jhs_analysis[['chd_inc','chd_years','hf_inc','hf_years','st_inc','st_years']].isna().sum().values], columns=['chd_inc','chd_years','hf_inc','hf_years','st_inc','st_years'])
        subset = jhs_analysis[['subjid','chd_inc','chd_years','hf_inc','hf_years','st_inc','st_years']]
        print("jhs_analysis 'chd * st * hf': \n", missing)#, '\n', subset[subset.isna().any(axis=1)])
    else:
        missing = pd.DataFrame(
            [jhs_analysis[['chd_inc', 'chd_years', 'st_inc', 'st_years']].isna().sum().values],
            columns=['chd_inc', 'chd_years', 'st_inc', 'st_years'])
        subset = jhs_analysis[['subjid', 'chd_inc', 'chd_years',  'st_inc', 'st_years']]
        print("jhs_analysis 'chd * st': \n", missing)#, '\n', subset[subset.isna().any(axis=1)])



    if with_hf=='with':
        print("\nINITIAL incidents count (all): ", ((jhs_analysis['chd_inc'] == 1) | (jhs_analysis['hf_inc'] == 1) | (jhs_analysis['st_inc'] == 1)).apply(lambda x: 1 if x else 0).sum())
        print(jhs_analysis[['chd_inc','hf_inc','st_inc']].sum())
        # exit()
    else:
        print("\nINITIAL incidents count (all): ",  ((jhs_analysis['chd_inc'] == 1) |  (jhs_analysis['st_inc'] == 1)).apply(lambda x: 1 if x else 0).sum())
        print(jhs_analysis[['chd_inc', 'hf_inc', 'st_inc']].sum())
        # exit()

    # todo: decide outputs LIMIT TO XX (max_incident_years) YEARS
    if incident_long_after:  jhs_analysis =  jhs_analysis[ ~((jhs_analysis['chd_inc' == 1]) & (jhs_analysis['chd_years'] > max_incident_years)) ]
    jhs_analysis.loc[   (jhs_analysis['chd_years'] > max_incident_years)    ,   'chd_inc'      ] = 0
    jhs_analysis.loc[   (jhs_analysis['chd_years'] > max_incident_years)    ,   'chd_years'      ] = max_incident_years
    if with_hf=='with':
        if incident_long_after:  jhs_analysis = jhs_analysis[~((jhs_analysis['hf_inc' == 1]) & (jhs_analysis['hf_years'] > max_incident_years))]
        jhs_analysis.loc[(jhs_analysis['hf_years'] > max_incident_years), 'hf_inc'] = 0
        jhs_analysis.loc[(jhs_analysis['hf_years'] > max_incident_years), 'hf_years'] = max_incident_years
    if incident_long_after:  jhs_analysis = jhs_analysis[~((jhs_analysis['st_inc' == 1]) & (jhs_analysis['st_years'] > max_incident_years))]
    jhs_analysis.loc[(jhs_analysis['st_years'] > max_incident_years), 'st_inc'] = 0
    jhs_analysis.loc[(jhs_analysis['st_years'] > max_incident_years), 'st_years'] = max_incident_years

    if with_hf=='with':
        jhs_analysis['any_inc'] = (     (jhs_analysis['chd_inc']==1) | (jhs_analysis['hf_inc']==1) | (jhs_analysis['st_inc']==1)        ).apply(lambda x: 1 if x else 0)
    else:
        jhs_analysis['any_inc'] = ((jhs_analysis['chd_inc'] == 1) |  (jhs_analysis['st_inc'] == 1)).apply(lambda x: 1 if x else 0)
    print("LIMITED 10years: ['any_inc']==1: ", (jhs_analysis['any_inc'] == 1).sum(),"\n")  # jhs_analysis[  ]
    print(jhs_analysis[['chd_inc', 'hf_inc', 'st_inc']].sum())
    # exit()

    if with_hf=='with':
        jhs_analysis['num_inc']=jhs_analysis['chd_inc']+jhs_analysis['st_inc']+jhs_analysis['hf_inc']
    else:
        jhs_analysis['num_inc'] = jhs_analysis['chd_inc'] + jhs_analysis['st_inc']
    print("\njhs_analysis['num_inc']:3 ", (jhs_analysis['num_inc']==3).sum() )
    print("jhs_analysis['num_inc']:2 ", (jhs_analysis['num_inc'] == 2).sum())
    print("jhs_analysis['num_inc']:1 ", (jhs_analysis['num_inc'] == 1).sum(),"\n")

    print("Setting Incident to Min year, No-incident to Max year")
    jhs_analysis['chd_mult']=jhs_analysis['chd_inc']*jhs_analysis['chd_years']
    jhs_analysis.loc[jhs_analysis['chd_mult']==0,'chd_mult']= np.nan
    jhs_analysis['st_mult'] = jhs_analysis['st_inc'] * jhs_analysis['st_years']
    jhs_analysis.loc[jhs_analysis['st_mult'] == 0,'st_mult'] = np.nan
    if with_hf=='with':
        jhs_analysis['hf_mult'] = jhs_analysis['hf_inc'] * jhs_analysis['hf_years']
        jhs_analysis.loc[jhs_analysis['hf_mult'] == 0,'hf_mult'] = np.nan

        jhs_analysis['outcome'] = (      jhs_analysis[['chd_mult','hf_mult','st_mult']]   ).min(axis=1)
        jhs_analysis.loc[jhs_analysis['outcome'].isna(), 'outcome'] = (jhs_analysis.loc[jhs_analysis['outcome'].isna(), ['chd_years', 'hf_years', 'st_years']]).max(axis=1)
        if ptr_debug_outcome_func: print("no outcome: \n",jhs_analysis.loc[jhs_analysis['outcome'].isna(),['chd_mult','chd_inc','chd_years', 'hf_mult','hf_inc','hf_years','st_mult','st_inc','st_years']])
        jhs_analysis = jhs_analysis.drop(columns=['chd_mult', 'hf_mult', 'st_mult', 'num_inc','chd_years', 'hf_years', 'st_years','chd_inc', 'hf_inc', 'st_inc'])

    else:
        jhs_analysis['outcome'] = (jhs_analysis[['chd_mult',  'st_mult']]).min(axis=1)
        jhs_analysis.loc[jhs_analysis['outcome'].isna(), 'outcome'] = (jhs_analysis.loc[jhs_analysis['outcome'].isna(), ['chd_years', 'st_years']]).max(axis=1)
        jhs_analysis = jhs_analysis.drop(columns=['chd_mult', 'st_mult', 'num_inc','chd_years',   'st_years','chd_inc',  'st_inc'])


    print("jhs_analysis: ", jhs_analysis.shape)
    jhs_analysis.dropna(subset=['any_inc','outcome'], inplace=True)
    print("jhs_analysis: drop missing 'any_inc'or'outcome'", jhs_analysis.shape)
    print("jhs_analysis['any_inc']==1 ", (jhs_analysis['any_inc'] == 1).sum(),"\n")
    # exit()

    if ptr_debug_outcome_func: print("\njhs_analysis: \n", jhs_analysis.head(10))
    if ptr_debug_outcome_func: print("\njhs_analysis: \n", jhs_analysis.describe())
    if ptr_debug_outcome_func: print("jhs_analysis: \n", pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values))
    missing = pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values)
    missing128 = missing.loc[:, missing.apply(lambda s: s > 128).to_numpy().flatten()]  # apply(lambda s: s > 128).all() if s.nunique() < 3 else s.nunique() > 1)
    print("jhs_analysis na>128: \n", missing128)
    print("jhs_analysis: ", jhs_analysis.shape)
    print("jhs_analysis if drop na: ", jhs_analysis.dropna().shape)
    print("jhs_analysis: drop NA, remaining 'any_inc'", jhs_analysis.loc[(jhs_analysis['any_inc'] == 1)].dropna().shape)
    print("\n")

    jhs_analysis = jhs_analysis.drop(columns=['visitdate','subjid','subjid','subjid.1',]) # 'depression', 'weeklystress',
    for x in ['visitdate','exam','subjid','subjid']: # #'depression','weeklystress',
        covariates_all.remove(x)
    # print(covariates_all)
    print("Drop Rows based on number of missing columns")
    print(" jhs_analysis:", jhs_analysis.shape)
    jhs_analysis= jhs_analysis[jhs_analysis.isna().sum(axis=1).values<individual_miss_threshold]
    print(" jhs_analysis: after threshold, ", jhs_analysis.shape)
    print("jhs_analysis['any_inc']==1 ", (jhs_analysis['any_inc'] == 1).sum(), "\n")

    print("Drop Rows based on columns")
    print(" jhs_analysis:", jhs_analysis.shape)
    if len(sub_set_column_to_drop_rows)>=1:
        if (len(sub_set_column_to_drop_rows))==1 and (sub_set_column_to_drop_rows[0]=="all"):
            jhs_analysis.dropna(inplace=True)
        else:
            jhs_analysis.dropna(subset=sub_set_column_to_drop_rows, inplace=True)
    print(" jhs_analysis: after threshold, ", jhs_analysis.shape)
    print("jhs_analysis['any_inc']==1 ", (jhs_analysis['any_inc'] == 1).sum(), "\n")

    print("\nImpute:: RF or mean/Mod")
    if impute_rf:
        # for col in jhs_analysis.columns:
        #     unique_values =  jhs_analysis[col].unique()
        #     if len(unique_values) <= 5:
        #         # print(col,': ',unique_values )
        #         jhs_analysis.loc[jhs_analysis[col].isna(),col] =  jhs_analysis[col].median()
        #         print(col, ': ', jhs_analysis[col].unique())
        imp = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
        imp.fit(jhs_analysis[covariates_all]) # .dropna().values
        jhs_analysis[covariates_all] = imp.transform(jhs_analysis[covariates_all])
    elif (not get_info):
        for col in jhs_analysis.columns:
            unique_values =  jhs_analysis[col].unique()
            if len(unique_values) <= 13:
                print(col,': ',unique_values )
                jhs_analysis.loc[jhs_analysis[col].isna(),col] =  jhs_analysis[col].median()
                if ptr_imputer: print(col,":: missing",jhs_analysis[[col]].isna().sum().values ,'use median(): ', jhs_analysis[col].unique())
            else:
                if ptr_imputer: print(col,":: missing",jhs_analysis[[col]].isna().sum().values ,'use mean(): ')#, unique_values)
                jhs_analysis.loc[jhs_analysis[col].isna(), col] = jhs_analysis[col].mean()
    else:
        print("************not imputed")
        # exit()

    # pts.loc[pts['bmi'].isna(), 'bmi'] = pts['bmi'].mean()  # 50%

    print(str(list(jhs_analysis.columns)))
    # exit()
    print("\njhs_analysis: ", jhs_analysis.shape)
    print("jhs_analysis['any_inc']==1 ", (jhs_analysis['any_inc'] == 1).sum(), "\n")

    # jhs_analysis.to_csv(dataLocation + "Data_set" + ".csv", index=False)

    # name = "DataSet_withHF-" +str(max_incident_years)+'y-impRf'+str(impute_rf)+'-RowThreshold'+str(individual_miss_threshold)+"-ColSpec"+str(len(sub_set_column_to_drop_rows))
    jhs_analysis.columns = jhs_analysis.columns.str.strip().str.lower()
    if get_info:
        jhs_analysis.to_csv(dataLocation + name + "_for-Get-Info.csv", index=False)
        print(name + "_for-Get-Info","\n")
    else:
        jhs_analysis.to_csv(dataLocation + name+ ".csv", index=False) # +'rfImpute-'+
        print(name,"\n")
    # else:
    #     # name = "DataSet_noHF-" +str(max_incident_years)+'y-impRf'+str(impute_rf)+'-Rowthreshold'+str(individual_miss_threshold)+"-ColSpec"+str(len(sub_set_column_to_drop_rows))
    #     jhs_analysis.columns = jhs_analysis.columns.str.strip().str.lower()
    #     jhs_analysis.to_csv(dataLocation + name + ".csv", index=False)
    #     print(name)
####################################################################################################################################
def create_model(param_grid):   #,
    global y_val
    global X_val
    global y_learn
    global X_learn

    # print("Model: ", param_grid)

    input_columns = param_grid['input_columns']
    output_classes = param_grid['output_classes']

    first_layer = param_grid['first_layer']
    second_layer = first_layer  # param_grid['second_layer']
    third_layer = first_layer  # param_grid['third_layer']

    drop_rate = param_grid['drop_rate']
    learning_ratio = param_grid['learning_ratio']
    momentu = param_grid['momentu']

    # actiovation_func = param_grid['']
    # loss =  param_grid['']
    # optimizer = param_grid['']

    epochs = param_grid['epochs']
    batch_size = param_grid['batch_size']

    print("input_columns \t output_classes \t first_layer \t second_layer \t third_layer \t drop_rate \t learning_ratio \t momentu \t epochs \t batch_size: \n"+\
          str(input_columns)+"\t"+str(output_classes)+"\t"+str(first_layer)+"\t"+str(second_layer)+"\t"+str(third_layer)+"\t"+str(drop_rate)+"\t"+str(learning_ratio)+"\t"+str(momentu)+"\t"+str(epochs)+"\t"+str(batch_size))

    model = models.Sequential()
    model.add(layers.Dense(first_layer, input_shape=(input_columns,)))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Activation(activations.relu))  # layers.Activation(activations.relu) # activations.sigmoid # activations.softmax
    model.add(layers.Dense(second_layer))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(third_layer))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(output_classes))
    model.add(layers.Activation(activations.sigmoid))
    # model.summary()

    # print("Compile")
    model.compile(loss='mean_squared_error',  # 'binary_crossentropy'
                  optimizer=SGD(learning_rate=learning_ratio, momentum=momentu),
                  # RMSprop(lr=0.001) # 'adam' # SGD(lr=0.01, momentum=0.9) # 'rmsprop'
                  metrics=['mean_squared_error'])

    # print("Fit: x,y learn,val: "+ str(len(X_learn))+"\t"+str( len(y_learn))+"\t"+str( len(X_val))+"\t"+str( len( y_val))) # ",X_learn.shape,y_learn.shape,X_val.shape,y_val.shape)
    history = model.fit(X_learn, y_learn,
                        # validation_data=(X_test,y_test),
                        validation_split=0.2,  # validation_data = validation_generator, # validation_split = 0.2
                        # validation_steps=50,
                        # steps_per_epoch=100,
                        epochs=epochs,  # bp.['epoch'],
                        batch_size=batch_size,  # bp.['batch'],
                        verbose=0)

    # print("Evaluate")
    history_test = model.evaluate(X_val, y_val)  # , verbose=0)
    pred_prob = model.predict(X_val)
    predictions = pred_prob#.argmax(axis=-1)
    y_val_binary = y_val
    y_val = y_val#.values.argmax(axis=-1)
    # predictions = model.predict_classes(X_test)

    # print("Save")
    # model.save("NN-initial.model")

    # print("History", history.history.keys())
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # print("Sklearn Eval:  class,pred binary,prob\n", y_test[:30],"\n",predictions[:30],"\n",y_test_binary[:5],"\n", pred_prob[:5])
    # print('\naccuracy_score: ', round(accuracy_score(y_test,predictions),2))
    # print('f1_score: ',f1_score(y_test, predictions, average='macro'))
    # print('precision_score: ',precision_score(y_test, predictions, average='macro'))
    # print('recall_score: ',recall_score(y_test, predictions, average='macro'))
    # print('Roc_auc score: ',roc_auc_score(y_test_binary, pred_prob))

    # results = {'accuracy_score': round(accuracy_score(y_val,predictions),2), 'f1_score': f1_score(y_val, predictions, average='macro'),
    #             'precision_score': precision_score(y_val, predictions, average='macro'), 'recall_score': recall_score(y_val, predictions, average='macro')}
    return (r2_score(y_val, predictions))
####################################################################################################################################
def modeling(covariates_all,outcomes):

    global y_val
    global X_val
    global y_learn
    global X_learn

    print("==================")
    dataLocation = "C:\\Users\\hrmor\\OneDrive - University of Mississippi Medical Center\\03_UMMC\\Projects__Jackson heart\\JHS data (Moradi-Morris)\\"
    if platform.system() != 'Windows':
        dataLocation = "/home/hmoradi/Downloads/Data/JHS/"
    jhs_analysis = pd.read_csv(dataLocation + "Data_set" + ".csv")
    for d in ["chd_date","hf_date","st_date"]:  # "visitdate",
        jhs_analysis[d] = pd.to_datetime(jhs_analysis[d], infer_datetime_format=True)
    covariates_all = [x.strip().lower() for x in covariates_all]


    #
    #
    # print("INITIAL incidents count (all): ", ((jhs_analysis['chd_inc'] == 1) | (jhs_analysis['hf_inc'] == 1) | (jhs_analysis['st_inc'] == 1)).apply(lambda x: 1 if x else 0).sum())
    #
    # jhs_analysis.loc[   (jhs_analysis['chd_years'] > 11)    ,   'chd_inc'      ] = 0
    # jhs_analysis.loc[   (jhs_analysis['chd_years'] > 11)    ,   'chd_years'      ] = 11
    # jhs_analysis.loc[(jhs_analysis['hf_years'] > 11), 'hf_inc'] = 0
    # jhs_analysis.loc[(jhs_analysis['hf_years'] > 11), 'hf_years'] = 11
    # jhs_analysis.loc[(jhs_analysis['st_years'] > 11), 'st_inc'] = 0
    # jhs_analysis.loc[(jhs_analysis['st_years'] > 11), 'st_years'] = 11
    #
    # jhs_analysis['any_inc'] = (     (jhs_analysis['chd_inc']==1) | (jhs_analysis['hf_inc']==1) | (jhs_analysis['st_inc']==1)        ).apply(lambda x: 1 if x else 0)
    # print("LIMITED 10years: ['any_inc']==1: ", (jhs_analysis['any_inc'] == 1).sum(),"\n")  # jhs_analysis[  ]
    #
    # jhs_analysis['num_inc']=jhs_analysis['chd_inc']+jhs_analysis['st_inc']+jhs_analysis['hf_inc']
    # print("\njhs_analysis['num_inc']:3 ", (jhs_analysis['num_inc']==3).sum() )
    # print("jhs_analysis['num_inc']:2 ", (jhs_analysis['num_inc'] == 2).sum())
    # print("jhs_analysis['num_inc']:1 ", (jhs_analysis['num_inc'] == 1).sum(),"\n")
    #
    # print("Setting Incident to Min year, No-incident to Max year")
    # jhs_analysis['chd_mult']=jhs_analysis['chd_inc']*jhs_analysis['chd_years']
    # jhs_analysis.loc[jhs_analysis['chd_mult']==0,'chd_mult']= np.nan
    # jhs_analysis['st_mult'] = jhs_analysis['st_inc'] * jhs_analysis['st_years']
    # jhs_analysis.loc[jhs_analysis['st_mult'] == 0,'st_mult'] = np.nan
    # jhs_analysis['hf_mult'] = jhs_analysis['hf_inc'] * jhs_analysis['hf_years']
    # jhs_analysis.loc[jhs_analysis['hf_mult'] == 0,'hf_mult'] = np.nan
    # jhs_analysis['outcome'] = (      jhs_analysis[['chd_mult','hf_mult','st_mult']]   ).min(axis=1)
    #
    # jhs_analysis.loc[jhs_analysis['outcome'].isna(),'outcome'] = (jhs_analysis.loc[jhs_analysis['outcome'].isna(),['chd_years', 'hf_years', 'st_years']]).max(axis=1)
    #
    # jhs_analysis = jhs_analysis.drop(columns=['chd_mult','hf_mult','st_mult','num_inc'])
    #
    #
    #
    # print("jhs_analysis: ", jhs_analysis.shape)
    # jhs_analysis.dropna(subset=['any_inc','outcome'], inplace=True)
    # print("jhs_analysis: drop missing 'any_inc','outcome'", jhs_analysis.shape)

    # <<<<<<<<<<<<<<<<<<<<<< imputation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # jhs_analysis = jhs_analysis.drop(columns=['depression', 'weeklystress','visitdate','subjid'])
    # for x in ['depression', 'weeklystress','visitdate','subjid']: #
    #     covariates_all.remove(x)
    # imp = IterativeImputer(max_iter=10, random_state=0)
    # imp.fit(jhs_analysis[covariates_all].dropna().values) #( x for x in covariates_all if x not in ['visitdate','subjid'] )
    # print("Write Impute back")
    # jhs_analysis[covariates_all] = imp.transform(jhs_analysis[covariates_all])



    print("\njhs_analysis: \n", jhs_analysis.head(10))
    print("\njhs_analysis: \n", jhs_analysis.describe())
    print("jhs_analysis: \n", pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values))
    missing = pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values)
    missing128 = missing.loc[:, missing.apply(lambda s: s > 360).to_numpy().flatten()]  # apply(lambda s: s > 128).all() if s.nunique() < 3 else s.nunique() > 1)
    print("jhs_analysis 360: \n", missing128)
    print("jhs_analysis: ", jhs_analysis.shape)
    print("jhs_analysis.dropna(): ", jhs_analysis.dropna().shape)
    print("jhs_analysis: 'any_inc'", jhs_analysis.loc[(jhs_analysis['any_inc'] == 1)].dropna().shape)
    print("\n")




    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FROM ENCLAVE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    start = time.time()
    print("datetime.now()", datetime.now())

    print("Numpy ", np.version.version)
    print("Sklearn ", sk.__version__)
    print("Keras ", keras.__version__)
    print("TF ", tf.__version__)
    if tf.test.gpu_device_name():
        print('    GPU: {}'.format(tf.test.gpu_device_name()))
    else:
        print('    CPU version')
    print()


    data_df = jhs_analysis[covariates_all]
    target_df = jhs_analysis[['outcome','any_inc']]


    print("data_df.shape: ", data_df.shape)
    print("target_df.shape: ", target_df.shape)
    print("before split, avg: ", target_df.sum() / float(target_df.shape[0]))
    print()

    X_train, X_test, y_train, y_test = train_test_split(data_df, target_df, test_size=0.30, random_state=7,stratify=target_df[['any_inc']])
    # sm = SMOTE(random_state=7)  # imbalance-learn # over sample
    # print("after Train-test", y_train.shape)
    # X_train, y_train = sm.fit_sample(X_train.values, y_train.values.argmax(axis=-1))
    # print("after sm.fit", y_train.shape)
    X_train = pd.DataFrame(data=X_train, columns=data_df.columns)
    y_train = pd.DataFrame(data=y_train, columns=target_df.columns)
    y_test = pd.DataFrame(data=y_test, columns=target_df.columns)
    # y_train = pd.DataFrame(data=tf.keras.utils.to_categorical(y_train, num_classes=2), columns=target_df.columns)
    X_learn, X_val, y_learn, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=7, stratify=y_train[['any_inc']])
    y_learn = pd.DataFrame(data=y_learn, columns=y_train.columns)
    y_val = pd.DataFrame(data=y_val, columns=y_train.columns)

    y_train_inc = y_train[['any_inc']]
    y_train = y_train[['outcome']]
    y_test_inc = y_test[['any_inc']]
    y_test = y_test[['outcome']]

    y_learn_inc = y_learn[['any_inc']]
    y_learn = y_learn[['outcome']]
    y_val_inc = y_val[['any_inc']]
    y_val = y_val[['outcome']]


    scale_x = preprocessing.MinMaxScaler().fit(X_train)  # StandardScaler()#MinMaxScaler()
    X_train = scale_x.transform(X_train)
    X_train = pd.DataFrame(data=X_train, columns=data_df.columns)
    X_test = scale_x.transform(X_test)
    X_test = pd.DataFrame(data=X_test, columns=data_df.columns)

    scale_learn_x = preprocessing.MinMaxScaler().fit(X_learn)
    X_learn = scale_learn_x.transform(X_learn)
    X_learn = pd.DataFrame(data=X_learn, columns=data_df.columns)
    X_val = scale_learn_x.transform(X_val)
    X_val = pd.DataFrame(data=X_val, columns=data_df.columns)
    # TODO: FIX ME !!!
    # scale_y = preprocessing.MinMaxScaler().fit(y_train)  # StandardScaler()#MinMaxScaler()
    # y_train = scale_y.transform(y_train)
    # y_train = pd.DataFrame(data=y_train, columns=['outcome'])
    # y_test = scale_y.transform(y_test)
    # y_test = pd.DataFrame(data=y_test, columns=['outcome'])

    # scale_learn_y = preprocessing.MinMaxScaler().fit(y_learn)
    # y_learn = scale_learn_y.transform(y_learn)
    # # y_learn = pd.DataFrame(data=y_learn, columns=y_train.columns)
    # y_val = scale_learn_y.transform(y_val)
    # # y_val = pd.DataFrame(data=y_val, columns=y_train.columns)

    print("X_train.shape, X_test.shape, X_learn.shape, X_val.shape, len(y_train), len(y_test), len(y_learn),len(y_val)")
    print( X_train.shape, X_test.shape, X_learn.shape, X_val.shape,  len(y_train), len(y_test), len(y_learn), len(y_val))


    # print("\nSMOTE(random_state = 2):")
    # if target_df.shape[1]==1:
    # print("\nCounts of target_df '2': "+str(sum(y_train == 2)))
    # print("Counts of target_df '1': "+str(sum(y_train == 1)))
    # print("Counts of target_df '0': "+str(sum(y_train == 0)))
    # print("Counts of target_df '-1': "+str(sum(y_train == -1)))
    # print("After split, y_train.sum(): \n", y_train.sum(), target_df.sum() / float(target_df.shape[0]))
    # print()

    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


    input_columns = hp.choice('input_columns', [X_train.shape[1]])
    output_classes = hp.choice('output_classes', [y_train.shape[1]])

    first_layer = hp.choice('first_layer', [25, 50, 75, 100])  # +X_train.shape[1]/5 # 20% more # approx. 38 columns
    # second_layer = hp.choice( 'second_layer',[2*X_train.shape[1]]) #/2
    # third_layer =  hp.choice( 'third_layer',[2*X_train.shape[1]]) #/4

    drop_rate = hp.choice('drop_rate', [0.1, 0.2, 0.3, 0.4, 0.5])  #
    learning_ratio = hp.choice('learning_ratio', [0.1, 0.01, 0.001, 0.0001])  #
    momentu = hp.choice('momentu', [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])  #

    # actiovation_func =
    # loss =
    # optimizer =

    epochs = hp.choice('epochs', [4, 16, 32, 64,128,256,512,1024,2048])  # 4,16,32,64,128,256,512,1024,2048
    batch_size = hp.choice('batch_size', [4, 16, 32, 64])  # 4,16,32,64

    max_eval = 50

    param_grid = dict(input_columns=input_columns, output_classes=output_classes,
                      first_layer=first_layer,  # second_layer=second_layer,third_layer=third_layer ,
                      drop_rate=drop_rate,
                      learning_ratio=learning_ratio, momentu=momentu,
                      batch_size=batch_size, epochs=epochs)  # not function param
    print('param_grid: ', param_grid)



    tpe_trials = Trials()
    tpe_best = fmin(fn=create_model, space=param_grid, algo=tpe.suggest, max_evals=max_eval, trials=tpe_trials,
                    rstate=np.random.RandomState(14))
    print("Pased (hours): ", (time.time() - start) / float(60 * 60), "\t", datetime.now())
    print(tpe_best)
    print(space_eval(param_grid, tpe_best))



    param_grid = space_eval(param_grid, tpe_best)

    input_columns = param_grid['input_columns']
    output_classes = param_grid['output_classes']

    first_layer = param_grid['first_layer']
    second_layer = first_layer  # param_grid['second_layer']
    third_layer = first_layer  # param_grid['third_layer']

    drop_rate = param_grid['drop_rate']
    learning_ratio = param_grid['learning_ratio']
    momentu = param_grid['momentu']

    # actiovation_func = param_grid['']
    # loss =  param_grid['']
    # optimizer = param_grid['']

    epochs = param_grid['epochs']
    batch_size = param_grid['batch_size']

    print(
        "input_columns \t output_classes \t first_layer \t second_layer \t third_layer \t drop_rate \t learning_ratio \t momentu \t epochs \t batch_size\n",
        input_columns, "\t", output_classes, "\t", first_layer, "\t", second_layer, "\t", third_layer, "\t", drop_rate,
        "\t", learning_ratio, "\t", momentu, "\t", epochs, "\t", batch_size)

    model = models.Sequential()
    model.add(layers.Dense(first_layer, input_shape=(input_columns,)))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Activation(
        activations.relu))  # layers.Activation(activations.relu) # activations.sigmoid # activations.softmax
    model.add(layers.Dense(second_layer))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(third_layer))
    model.add(layers.Dropout(drop_rate))
    model.add(layers.Activation(activations.relu))
    model.add(layers.Dense(output_classes))
    model.add(layers.Activation(activations.sigmoid))
    model.summary()

    print("Compile")
    model.compile(loss='mean_squared_error',  # 'binary_crossentropy'
                  optimizer=SGD(learning_rate=learning_ratio, momentum=momentu),
                  # RMSprop(lr=0.001) # 'adam' # SGD(lr=0.01, momentum=0.9) # 'rmsprop'
                  metrics=['mean_squared_error'])

    # print("Fit: x,y train,test ",X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    history = model.fit(X_train, y_train,
                        # validation_data=(X_test,y_test),
                        validation_split=0.2,  # validation_data = validation_generator, # validation_split = 0.2
                        # validation_steps=50,
                        # steps_per_epoch=100,
                        epochs=epochs,  # bp.['epoch'],
                        batch_size=batch_size,  # bp.['batch'],
                        verbose=1)

    print("Evaluate")
    history_test = model.evaluate(X_test, y_test)  # , verbose=0)
    pred_prob = model.predict(X_test)
    predictions = pred_prob#.argmax(axis=-1)
    y_test_binary = y_test
    y_test = y_test#.values.argmax(axis=-1)
    # predictions = model.predict_classes(X_test)

    print("Save")
    model.save("NN-initial.model")

    print("History", history.history.keys())
    acc = history.history['mean_squared_error']
    val_acc = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    # fig, ax = plt.subplots()
    # ax.plot(epochs, acc, 'r', label='Training accuracy')
    # ax.plot(epochs, val_acc, 'b', label='Validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.legend(loc=0)
    # plt.show()
    fig = plt.figure(figsize=(10, 20))
    ax0 = plt.subplot2grid((6, 3), (0, 0), rowspan=1, colspan=3)
    ax1 = plt.subplot2grid((6, 3), (1, 0), rowspan=1, colspan=3)
    ax2 = plt.subplot2grid((6, 3), (2, 0), rowspan=4, colspan=3)
    sns.lineplot(ax=ax0, x=epochs, y=acc, legend='brief', label="accuracy")
    sns.lineplot(ax=ax0, x=epochs, y=val_acc, legend='brief', label="val_accuracy")
    sns.lineplot(ax=ax0, x=epochs, y=loss, legend='brief', label="loss")
    sns.lineplot(ax=ax0, x=epochs, y=val_loss, legend='brief', label="val_loss")
    ax0.set_title('Learning Curve: Training and validation accuracy/loss')  # , y=1.05, size=15)

    # RFE
    print("Sklearn Eval:  class,pred binary,prob\n", y_test[:2], "\n", predictions[:2], "\n", y_test_binary[:2], "\n", pred_prob[:2])
    # print('\naccuracy_score: ', round(accuracy_score(y_test, predictions), 2))
    # print('f1_score: ', f1_score(y_test, predictions, average='macro'))
    # print('precision_score: ', precision_score(y_test, predictions, average='macro'))
    # print('recall_score: ', recall_score(y_test, predictions, average='macro'))
    # print('Roc_auc score: ', roc_auc_score(y_test_binary, pred_prob))
    print( "Concordance_index NN:",        concordance_index(y_test, predictions, y_test_inc.values)         ,"\n" )

    #
    # print("Nulls: ",X_train.shape,X_train.isna().sum().values, y_train.shape,y_train.isna().sum().values, y_train_inc.shape,y_train_inc.isna().sum().values)
    # y_train=y_train.reset_index()
    # y_train=y_train.reset_index()
    # y_train_inc=y_train_inc.reset_index()
    # print("columns: ", X_train.columns, y_train.columns, y_train_inc.columns)
    # print("y_train.head():\n ",y_train.head())
    # concat_for_Cox = pd.concat([X_train,y_train[['outcome']],y_train_inc[['any_inc']]],axis=1,names=([X_train.columns.values+ 'outcome'+ 'any_inc'])) # ,columns=(X_train.columns+y_train.columns+y_train_inc.columns
    # print("concat_for_Cox.columns: ",concat_for_Cox.columns)
    # print("concat_for_Cox.isna(): ", concat_for_Cox.isna().sum().values)
    # print(concat_for_Cox.describe())
    #
    # cph = CoxPHFitter().fit(concat_for_Cox, 'outcome', event_col='any_inc')
    # print( "Concordance_index CoxPHFitter:",  concordance_index(y_test, -cph.predict_partial_hazard(  X_test.values  ), y_test_inc.values) )
    #




    exit()
    #####################   Concordance_index NN: 0.5424166493418389    ##########################
    #  y normilized
    # {'batch_size': 16, 'drop_rate': 0.3, 'epochs': 4, 'first_layer': 25, 'input_columns': 162, 'learning_ratio': 0.0001, 'momentu': 0.4, 'output_classes': 1}
    # Pased (hours):  0.9585778581433826 	 2021-07-19 17:41:29.684016

    #  no Y normalization
    # Concordance_index
    # NN: 0.4969198279918154

    # cudda and no Y norm




    accuracy_ = pd.DataFrame(data={'accuracy_score': [round(accuracy_score(y_test, predictions), 2)]})
    f1_ = pd.DataFrame(data={'f1_score': [f1_score(y_test, predictions, average='macro')]})
    # 'micro' = total true positives, 'macro' = for each label,unweighted
    # 'weighted' = account for label imbalance,
    # TP / TP + FP >>> total predicted positives (column)
    precision_ = pd.DataFrame(data={'precision_score': [precision_score(y_test, predictions, average='weighted')]})
    # TP / TP + FN >>> total Actual positives (row)
    recall_ = pd.DataFrame(data={'recall_score': [recall_score(y_test, predictions, average='weighted')]})
    print("precision_score", precision_score(y_test, predictions, average='weighted'))
    finalResults = pd.concat([accuracy_, f1_, precision_, recall_], axis=1)
    confusion_ = confusion_matrix(y_test_binary.values.argmax(axis=1),
                                  pred_prob.argmax(axis=1))  # .values.argmax(axis=1)
    for class_ in range(len(confusion_)):
        class_accuracy = confusion_[class_][class_] / float(sum(confusion_[class_]))
        class_str = "class_" + str(class_) + "_accuracy"
        print(class_str, class_accuracy)
        class_accuracy_df = pd.DataFrame(data={class_str: [class_accuracy]})
        finalResults = pd.concat([finalResults, class_accuracy_df], axis=1)
    print('confusion_matrix: ', confusion_matrix(y_test_binary.values.argmax(axis=1), pred_prob.argmax(axis=1)))

    cf_matrix = confusion_matrix(y_test_binary.values.argmax(axis=1), pred_prob.argmax(axis=1))
    vmin = np.min(cf_matrix)
    vmax = np.max(cf_matrix)
    off_diag_mask = np.eye(*cf_matrix.shape, dtype=bool)
    # fig = plt.figure(figsize=(10,20))
    # # fig, axes = plt.subplots(2, 1,figsize=(10,20)) # axes[0]
    # ax1 = plt.subplot2grid((6,3),(0,1),rowspan = 1,colspan = 1)
    # ax2 = plt.subplot2grid((6,3),(2,0), rowspan = 4,colspan = 3)
    # sns.set_style("whitegrid")
    sns.heatmap(ax=ax1, data=cf_matrix, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax, fmt='2.0f')
    sns.heatmap(ax=ax1, data=cf_matrix, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax,
                cbar_kws=dict(ticks=[]), fmt='2.0f')
    # sns.heatmap(confusion_matrix(y_test,predictions),annot=True,cmap='Blues',fmt='3.0f')
    ax1.set_title('Confusion matrix')  # , y=1.05, size=15)

    # # Permutate
    # # results = permutation_importance(model, X_test, y_test, scoring='neg_mean_squared_error')
    # # importance = results.importances_mean

    # bar=progressbar.ProgressBar(max_value=len(X.columns))
    print("Pased (hours): ", round((time.time() - start), 2) / float(60 * 60), "\t", datetime.now(), "\t iterations:",
          max_eval, " per iter.:", round((time.time() - start), 2) / float(60 * 60))
    importance = {c: [] for c in X_test.columns}
    print("Permutate: ", end=" ")
    count = 1
    for c in X_test.columns:
        print(len(importance) - count, end=" ")
        X_test_c = X_test.copy(deep=True)
        accuracy_ = []
        for _ in range(5):  # 10
            temp = X_test_c[c].tolist()
            random.shuffle(temp)
            X_test_c[c] = temp
            # predictions = model.predict(X_test_c)
            pred_prob = model.predict(X_test_c)
            predictions = pred_prob.argmax(axis=-1)
            accuracy_.append(accuracy_score(y_test, predictions))
        importance[c].append(1 - np.mean(accuracy_))
        count += 1
    print("")
    # bar.update(X.columns.tolist().index(c))

    # print(importance[0:5])
    importance_df = pd.DataFrame(data=importance, index=["values"])  # index =X_test.columns,columns=["values"]
    importance_df.sort_values(by="values", axis=1, ascending=False, inplace=True)
    print("importance_df:\n", importance_df.head())
    # sns.barplot(ax=ax2,data = importance_df, y=importance_df.index,x='values', orient='h') #,order=df.sort_values('y',ascending = False).Education)###
    # https://seaborn.pydata.org/tutorial/categorical.html
    importance_df = importance_df.T
    importance_df['col'] = importance_df.index
    # importance_df['values'] = np.log10(importance_df['values'])
    print(importance_df.head())
    sns_log = sns.stripplot(ax=ax2, data=importance_df, y='col', x='values')  # ,x='values', orient='h')
    ax2.xaxis.grid(True)
    # sns_log.set(xscale="log")
    # sns_log.set_xscale("log")
    ax2.set_title('Feature importance by permutation (measured by reduction in Accuracy)')  # , y=1.05, size=15)

    # df_test = pd.DataFrame(data=X_test,columns=data_df.columns)
    # df_target = {'improved':y_test}
    # df_target = pd.DataFrame(data=df_target)
    # df_predict = {'predictions':predictions}
    # df_predict = pd.DataFrame(data=df_predict)
    # finalResults = pd.concat([df_test, df_target, df_predict], axis=1)

    plt.tight_layout()
    plt.show()
    plt.show()
    end = time.time()
    print("End-Start (hours): ", (end - start) / float(60 * 60), "\t", datetime.now())
    return (finalResults)

####################################################################################################################################
# if __name__ == '__main__':
def main():
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
    print("Find Main Covariates:")
    find_covariates(covariates_main)
    ###########################################################################
    dataLocation = "C:\\Users\\hrmor\\OneDrive - University of Mississippi Medical Center\\04_Projects\\Project__JHS_Morris\\JHS data (Moradi-Morris)\\"
    if platform.system() != 'Windows':
        dataLocation = "/home/hmoradi/Downloads/Data/JHS/"
    env_var = pd.read_csv(dataLocation + "env_var" + ".csv",header=None)
    covariates_env =  env_var.values.flatten()
    # fake_stcotrk --->>> replaced with --->>> FakeCensusTractID
    print("\nFind Env. Covariates:")
    find_covariates(covariates_env)
    ############################################################################
    covariates_all=[]
    for x in covariates_main:
        covariates_all.append(x)
    for x in covariates_env:
        covariates_all.append(x)
    print("\nFeature Collector:")
    feature_collector(covariates_all)
    ############################################################################
    covariates_all = [x.strip().lower() for x in covariates_all]
    for x in ['mihx','chdhx','strokehx','cvdhx','cardiacprochx']:
        covariates_all.remove(x)
    print("\nOutcome Collector:")
    outcome_collector(covariates_all)
    ############################################################################
    print("\nModeling:")
    for x in ['depression', 'weeklystress', 'visitdate', 'exam', 'subjid', 'subjid']:  #
        covariates_all.remove(x)
    outcomes=["chd_inc","chd_date","chd_year","chd_years","chd_days"
                ,"hf_inc", "hf_date", "hf_year", "hf_years", "hf_days"
                ,"st_inc", "st_date", "st_year", "st_years", "st_days"]
    # modeling(covariates_all,outcomes)
####################################################################################################################################