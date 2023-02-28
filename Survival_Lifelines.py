from Param_JHS import *
from Libraries_JHS import *

########################################################################################################################
glob_x_train, glob_x_val, glob_x_test = [],[],[]
glob_y_train, glob_y_val, glob_y_test = [],[],[]
final_features = []
labtrans= []
########################################################################################################################
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
    # cols_standardize.remove('duration')
    # cols_standardize.remove('event')  # covariates_all
    global final_features
    final_features = cols_standardize
    # print(cols_standardize)
    standardize = [([col], MinMaxScaler()) for col in cols_standardize] # StandardScaler
    cols_leave = [x for x in df_train.columns.values if x not in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize)  # + leave

    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_train = pd.DataFrame(data=x_train, columns=df_train.columns)
    x_val = x_mapper.transform(df_val).astype('float32')
    x_val = pd.DataFrame(data=df_val, columns=df_val.columns)
    x_test = x_mapper.transform(df_test).astype('float32')
    x_test = pd.DataFrame(data=df_test, columns=df_test.columns)

    num_durations = 10
    # labtrans = DeepHitSingle.label_transform(num_durations)

    # get_target = lambda df: (df['duration'].values, df['event'].values)
    # y_train = labtrans.fit_transform(*get_target(df_train))
    # y_val = labtrans.transform(*get_target(df_val))
    # y_test = labtrans.transform(*get_target(df_test))

    glob_x_train, glob_x_val, glob_x_test = x_train, x_val, x_test
    # glob_y_train, glob_y_val, glob_y_test = y_train, y_val, y_test
########################################################################################################################
def modeling(covariates_all,outcomes,cleaned_main_cov):

    global glob_x_train, glob_x_val, glob_x_test
    global glob_y_train, glob_y_val, glob_y_test
    global labtrans

    print_log("\nLifelines_rf")
    print_log("==================")
    print(name+".csv\n")

    jhs_analysis = pd.read_csv(dataLocation + name + ".csv") # Data_set_hf_rfImpute-True # _rfImpute-True # Data_set_hf_10y-False # Data_set_hf
    # for d in ["chd_date","hf_date","st_date"]:  # "visitdate",
    #     jhs_analysis[d] = pd.to_datetime(jhs_analysis[d], infer_datetime_format=True)
    print(jhs_analysis.columns)
    # exit()
    covariates_all = [x.strip().lower() for x in covariates_all]

    # categorical
    for var in covariates_all:
        items = jhs_analysis[var].unique()
        if len (items)<20:
            print(var,":",str(items))
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
    #### >>>>>
    # param__
    max_eval=20
    optimize = False
    # optimize = True

    # num_layers = hp.choice('num_layers', [1, 2, 3, 4, 5])
    # num_neurons = hp.choice('num_neurons', [25, 50, 75, 100,150])
    # batch_norm = hp.choice('batch_norm', [True, False])
    # dropout = hp.choice('dropout', [0.1, 0.2, 0.3, 0.4, 0.5]) # 0.1
    # learning_rate = hp.choice('learning_rate', [0.1, 0.01, 0.001, 0.0001])  # 0.01
    # batch_size = hp.choice('batch_size', [4, 16, 32, 64])
    # epochs = hp.choice('epochs', [4, 16, 32, 64, 128, 256, 512, 1024]) # 100

    # momentu = param_grid['']
    # actiovation_func = param_grid['']
    # loss =  param_grid['']
    # optimizer = param_grid['']

    # param_grid = dict(num_layers=num_layers,
    #                   num_neurons=num_neurons,
    #                   batch_norm=batch_norm, dropout=dropout,
    #                   learning_rate=learning_rate, batch_size=batch_size,
    #                   epochs=epochs)
    # print_log('param_grid: ', param_grid)

    # tpe_trials = Trials()
    tpe_best = []
    model = []
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
        # model = eval_model(param_grid)['model']



    print("glob_x_train.columns", len(glob_x_train.columns))
    print("cleaned_main_cov",len(cleaned_main_cov))
    cleaned_main_cov = [x for x in glob_x_train.columns if any([True for y in cleaned_main_cov if y.startswith(x)])]
    cleaned_main_cov.append('duration')
    cleaned_main_cov.append('event')
    cleaned_main_cov = glob_x_train.columns
    print("cleaned_main_cov",len(cleaned_main_cov))


    Xs=pd.concat([glob_x_train[cleaned_main_cov], glob_x_val[cleaned_main_cov], glob_x_test[cleaned_main_cov]], axis=0,ignore_index=True)
    # Ys=pd.concat([glob_y_train, glob_y_val, glob_y_test], axis=0)
    # XY=pd.concat([Xs,Ys], axis=1)

    cph = CoxPHFitter(penalizer=0.0001, l1_ratio=0.001)
    cph.fit(Xs, 'duration', event_col='event')
    cph.print_summary()
    # exit(0)
    scores = k_fold_cross_validation(cph, Xs, 'duration', event_col='event', k=10)
    print(scores)
    print(np.average(scores))
    exit(0)
    # [-0.5242949384176823, -0.5381298690443433, -0.5137557201395361, -0.5702555786222618, -0.66149334608674,
    #  -0.5555031588657328, -0.5264151333587714, -0.5557490592499997, -0.5380915925666223, -0.5976553516119979]
    # mean: -0.5581343747963687



    # # global glob_x_train, glob_x_val, glob_x_test
    # # global glob_y_train, glob_y_val, glob_y_test
    # # global labtrans
    # set_global_data(df_train, df_test)

    print('\nShap') #########################
    calc_shap = False
    graph_shap = False
    if graph_shap:
        shap_values = []
        #
        # model = eval_model(param_grid)['model']
        # explainer = shap.KernelExplainer(model.predict, glob_x_train)

        if calc_shap:
            model = eval_model(param_grid)['model']
            explainer = shap.KernelExplainer(model.predict, glob_x_train)
            # explainer = shap.DeepExplainer(model, glob_x_train) # sample(n=int(glob_x_train.shape[0] // 1))

            shap_values = explainer.shap_values(glob_x_test, nsamples=100) # [:5,:]
            # shap_values = explainer.shap_values(glob_x_test)  # , nsamples=10) # sample(n=int(testing.shape[0] // 1)).

            print('training size:', len(glob_x_train), len(glob_x_train[0]))
            print('features:', len(final_features))
            print('\nD1 Classes:', len(shap_values), '\nD2 samples:', len(shap_values[0]), '\nD3 Columns/features:', len(shap_values[0][0]), '\nvalue:', shap_values[0][0][0])
            print('type: ',type(shap_values))
            print('type [0]: ', type(shap_values[0]))
            # write ################################
            for i in range(len(shap_values)):
                np.savetxt("shap_"+str(i)+".csv", shap_values[i])
            np.savetxt("shape.csv",np.array([len(shap_values)]))

        # read ###################################
        arr_shape = np.loadtxt("shape.csv")
        shap_values = [ [] for i in range(int(arr_shape))]
        for i in range(int(arr_shape)):
            shap_values[i] = np.loadtxt("shap_"+str(i)+".csv")

        print('\nD1 Classes:',len(shap_values),'\nD2 samples:', len(shap_values[0]),'\nD3 Columns/features:',len(shap_values[0][0]),'\nvalue:',shap_values[0][0][0])
        print('type: ',type(shap_values))
        print('type [0]: ', type(shap_values[0]))

        # shap.summary_plot(shap_values, glob_x_train, feature_names=final_features, plot_type="bar")
        overall_classes = np.array( [ 0 for _ in range(len(shap_values[0][0]))  ] )
        overall_classes = np.array( [overall_classes  for _ in range(len(shap_values[0]))   ] )
        for x in range(len(shap_values)) :
            overall_classes = np.add(overall_classes,shap_values[x] )
        overall_classes = np.divide(overall_classes, len(shap_values))
        print('overall_classes.shape', overall_classes.shape)
        # shap.summary_plot(overall_classes, glob_x_train, feature_names=final_features, plot_type="bar")
        overall_variables = np.array([0 for _ in range(len(shap_values[0][0]))])
        for x in range(overall_classes.shape[0]) :
            overall_variables = np.add(overall_variables,overall_classes[x] )
        overall_variables = np.divide(overall_variables, overall_classes.shape[0])
        print('overall_variables.shape',overall_variables.shape)
        # shap.summary_plot(overall_variables, glob_x_train, feature_names=final_features, plot_type="bar")
        df = pd.DataFrame(overall_variables, columns=['values'],index=final_features)



        ### Ploting
        print("\nPloting")
        df[['label','color']] = np.nan
        df = df.sort_values(by='values', ascending=False, )
        print(df.head(2),"\n")
        env_disc = pd.read_csv(dataLocation + "env_disc" + ".csv")
        env_disc['Variable Name'] = env_disc['Variable Name'].str.lower()
        print(env_disc.head(2),"\n")

        palette = sns.color_palette("bright", 10) # pastel

        no_desc = 0
        df = df.drop('occupation_2.58509567', 0)
        geo_risk_dict = {'fakecensustractid':'Fake census tract ID'}
        standard_risk_dict = {'sbp':'Systolic blood pressure', 'dbp':'Diastolic blood pressure', 'abi':'Ankle brachial index', 'bpmeds':'Blood pressure medication status',
                            'totchol':'Total cholesterol','ldl':'LDL cholesterol','hdl':'HDL cholesterol ', 'trigs':'Triglycerides ', 'fpg':'Fasting glucose',
                            'hba1c':'Hemoglobin A1C', 'alc':'Alcohol drinking in the past 12 months','alcw':'Average number of drinks per week',
                            'currentsmoker':'Cigarette Smoking Status','eversmoker':'History of Cigarette Smoking',
                            'pa3cat': 'Physical Activity', 'activeindex':'Physical activity during leisure time','nutrition3cat':'Nutrition Categorization',
                            'depression':'Depressive Symptoms Score','weeklystress':'Weekly stress score', 'perceivedstress':'Global Stress Score',
                            }
        # demo + insurance + psycho
        scio_risk_dict = {'fmlyinc':'Family income', # Psychosocial
                          'occupation_1.0': 'occupation - Management/Professional', 'occupation_2.0': 'occupation - Service', 'occupation_3.0': 'occupation - Sales',
                          'occupation_4.0': 'occupation - Farming', 'occupation_5.0': 'occupation - Construction',
                          'occupation_6.0': 'occupation - Production', 'occupation_7.0': 'occupation - Military', 'occupation_8.0': 'occupation - Sick',
                          'occupation_9.0': 'occupation - Unemployed', 'occupation_10.0': 'occupation - Homemaker', 'occupation_11.0': 'occupation - Retired',
                          'occupation_12.0': 'occupation - Student', 'occupation_13.0': 'occupation - Other',
                          'edu3cat_0':'Education - Less thank high school','edu3cat_1':'Education - LHigh school graduate/GED','edu3cat_2':'Education - attended vocational or colledge',
                          'dailydiscr':'Daily discrimination','lifetimediscrm':'Lifetime discrimination','discrmburden':'Discrimination burden',

                          'insured':'Insured', # insurance
                          'privatepublicins_0': 'Insurance status - Uninsured', 'privatepublicins_1': 'Insurance status - Private Only',
                          'privatepublicins_2': 'Insurance status - Public Only', 'privatepublicins_3': 'Insurance status - Private & Public',

                          'sex': 'Sex', 'age': 'Age','waist':'Waist','bmi':'BMI', #  Demographic
                          }
        for x in df.index:
            if x in env_disc['Variable Name'].values:
                # print(env_disc.loc[env_disc['Variable Name']==x] )
                df.loc[x,'label']= str(env_disc.loc[env_disc['Variable Name']==x,'Label'].values[0]).split('\n')[0] + ' ('+str(x)+')'
                df.loc[x, 'color'] = 2
                df.loc[x, 'color_label'] = 'Environmental'
            elif x.split('_')[0] in env_disc['Variable Name'].values:
                print('First part exist: ',x)
                # print(env_disc.loc[env_disc['Variable Name'] == x.split('_')[0]])
                df.loc[x, 'label'] = str(env_disc.loc[env_disc['Variable Name']==x.split('_')[0],'Label'].values[0]).split('\n')[0] + ' ('+str(x)+')'
                df.loc[x, 'color'] = 2
                df.loc[x, 'color_label'] = 'Environmental'
            elif x in geo_risk_dict.keys():
                df.loc[x, 'label'] = geo_risk_dict[x]+' ('+str(x)+')'
                df.loc[x, 'color'] = 2
                df.loc[x, 'color_label'] = 'Environmental'
            elif x in standard_risk_dict.keys():
                df.loc[x, 'label'] = standard_risk_dict[x]+' ('+str(x)+')'
                df.loc[x, 'color'] = 3
                df.loc[x, 'color_label'] = 'Standard'
            elif x in scio_risk_dict.keys():
                df.loc[x, 'label'] = scio_risk_dict[x]+' ('+str(x)+')'
                df.loc[x, 'color'] = 0
                df.loc[x, 'color_label'] = 'Psychosocial'
            else:
                no_desc+=1
                print('no_desc: '+str(no_desc)+" "+x)


        df['label'] = df['label'].str.replace("  "," ").replace("\n"," ").replace("\t"," ")
        # print(df.dropna().head())
        # exit()

        palette = {"Environmental": palette[2], "Standard": palette[3], "Psychosocial": palette[0]}
        hue_order = ["Environmental", "Standard", "Psychosocial"]
        ### Positive
        my_dpi = 100
        df_filtered = df[df['values']>=0].sort_values(by='values', ascending=False).reset_index(drop=True).copy()
        for i in range(0,df_filtered.shape[0],51):
            partial_df = df_filtered.iloc[i:i+50].copy()
            print("partial_df: ", partial_df.shape)
            plt.figure(figsize=(1200 / my_dpi, (900 / my_dpi)*((partial_df.shape[0]+10)/(51+10))   ), dpi=my_dpi)
            # sns.set(style="ticks")
            sns.set_style("darkgrid",{"axes.facecolor": ".9"})
            # sns.set_context("paper")
            sns.barplot(y='label', x="values", data=partial_df,hue='color_label',dodge=False,palette=palette,hue_order=hue_order)#,palette=[palette[i] for i in partial_df.color.astype(int)])
            # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
            # plt.tight_layout()
            plt.legend(title="Predictors",loc='lower right')
            plt.xlabel('Mean SHAP (average impact on model output magnitude)',Fontsize=12 )
            plt.ylabel('Variables',Fontsize=12)
            # plt.show()
            plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
            plt.grid()
            plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
            plt.savefig(dataLocation+"Figs/"+"Positive-"+str(i),  bbox_inches="tight",pad_inches=0.3) # facecolor='y', , transparent=True, dpi=200
            plt.clf()
            # exit()

        ### Negative
        plt.clf()
        plt.close()
        df_filtered = df[df['values'] < 0].sort_values(by='values', ascending=False).reset_index(drop=True).copy()
        for i in range(0, df_filtered.shape[0], 51):
            partial_df = df_filtered.iloc[i:i + 50].copy()
            print("partial_df: ", partial_df.shape)
            plt.figure(figsize=(1200 / my_dpi, (900 / my_dpi)*((partial_df.shape[0]+10)/(51+10))   ), dpi=my_dpi)
            # sns.set(style="ticks")
            sns.set_style("darkgrid", {"axes.facecolor": ".9"})
            # sns.set_context("paper")
            sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
                        dodge=False,palette=palette,hue_order=hue_order)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
            # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
            # plt.tight_layout()
            plt.legend(title="Predictors", loc='upper left')
            plt.xlabel('Mean SHAP (average impact on model output magnitude)', Fontsize=12)
            plt.ylabel('Variables', Fontsize=12)
            # plt.show()
            plt.xlim(df_filtered['values'].min()* 1.02 , df_filtered['values'].max() + 0.0002)
            plt.grid()
            plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
            plt.savefig(dataLocation + "Figs/" + "Negative-" + str(i), bbox_inches="tight",
                        pad_inches=0.3)  # facecolor='y', , transparent=True, dpi=200
            plt.clf()
            # exit()

        ### only positive
        plt.clf()
        plt.close()
        df_filtered = df.copy()
        df_filtered['values'] = df_filtered['values'].abs()
        df_filtered = df_filtered.sort_values(by='values', ascending=False).reset_index(drop=True)
        for i in range(0, df_filtered.shape[0], 51):
            partial_df = df_filtered.iloc[i:i + 50].copy()
            print("partial_df: ", partial_df.shape)
            plt.figure(figsize=(1200 / my_dpi, (900 / my_dpi)*((partial_df.shape[0]+10)/(51+10))  ), dpi=my_dpi)
            # sns.set(style="ticks")
            sns.set_style("darkgrid", {"axes.facecolor": ".9"})
            # sns.set_context("paper")
            sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
                        dodge=False,hue_order=hue_order,palette=palette)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
            # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
            # plt.tight_layout()
            plt.legend(title="Predictors", loc='lower right')
            plt.xlabel('Mean |SHAP| (average impact on model output magnitude)', Fontsize=12)
            plt.ylabel('Variables', Fontsize=12)
            # plt.show()
            plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
            plt.grid()
            plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
            plt.savefig(dataLocation + "Figs/" + "Abs-" + str(i), bbox_inches="tight",
                        pad_inches=0.3)  # facecolor='y', , transparent=True, dpi=200
            plt.clf()
            # exit()

        ### Similarity
        my_dpi = 100
        df_filtered = df.sort_index(ascending=False).reset_index(drop=True).copy()
        for i in range(0, df_filtered.shape[0], 51):
            partial_df = df_filtered.iloc[i:i + 50].copy()
            print("partial_df: ", partial_df.shape)
            plt.figure(figsize=(1200 / my_dpi,  (900 / my_dpi)*((partial_df.shape[0]+10)/(51+10))   ), dpi=my_dpi)
            # sns.set(style="ticks")
            sns.set_style("darkgrid", {"axes.facecolor": ".9"})
            # sns.set_context("paper")
            sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
                        dodge=False,hue_order=hue_order,palette=palette)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
            # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
            # plt.tight_layout()
            plt.legend(title="Predictors", loc='lower right')
            plt.xlabel('Mean SHAP (average impact on model output magnitude)', Fontsize=12)
            plt.ylabel('Variables', Fontsize=12)
            # plt.show()
            plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
            plt.grid()
            plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
            plt.savefig(dataLocation + "Figs/" + "Similarity-" + str(i), bbox_inches="tight",
                        pad_inches=0.3)  # facecolor='y', , transparent=True, dpi=200
            plt.clf()
            # exit()


        df = df.sort_index(ascending=False) # key=lambda x: x.str.lower(),
        # df['labels'] = df.index.str.lower()
        # df = df.sort_values('labels').drop('labels', axis=1)
        sns.barplot(y=df.index, x="values", data=df)
        plt.xticks(fontsize=8, rotation=90)  # ax.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        # plt.show()



        df.to_csv(resultsLocation + "Result_shap" + ".csv", index=True)
    # shap.initjs()

    # shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], glob_x_train[0, :], feature_names=final_features)

    # https://github.com/smazzanti/tds_black_box_models_more_explainable/blob/master/
    # Shap2Probas.ipynb (sexy recruter, racist chatbot) #########################3
    # https://github.com/slundberg/shap
    # DeepExplainer(DEEP SHAP), GradientExplainer
    # LIME creates a surrogate model locally around the unit who’s prediction you wish to understand. Thus it is inherently local.
    # Shapely values ‘decompose’ the final prediction into the contribution of each attribute





    print_log("Final Model")
    print_log("Final Model Accuracy: ", eval_model(param_grid)['acc'])

    #### >>>>>
    k = 10 ################################################################################################
    kfold_accuracy = pd.DataFrame(columns= [ str(k+1)+'-fold' for k in range(10)],index=['overall'])

    ### OVERAL
    df_jhs = jhs_analysis.copy()
    df_jhs_tested = jhs_analysis.copy()
    for i in range(k):
        df_test = df_jhs_tested.sample(n=int(df_jhs.shape[0]/10.0))
        df_jhs_tested= df_jhs_tested.drop(df_test.index)
        df_train = df_jhs.drop(df_test.index)
        set_global_data(df_train,  df_test)
        model_accuracy = eval_model(param_grid)['acc']
        print_log("Model Accuracy: ", model_accuracy)
        kfold_accuracy.loc['overall',str(i+1)+'-fold']= model_accuracy
        kfold_accuracy['mean'] = kfold_accuracy.mean(axis=1)
        kfold_accuracy.to_csv(resultsLocation + "Result_10fold_main" + ".csv", index=True)
        print_log(kfold_accuracy.tail(5))
    print_log("not tested: df_jhs_tested ", df_jhs_tested.shape[0])

    exit(0)
    ### Conditional
    k = 10
    cond = [    #{'var': 'age', 'op': '<=', 'val': 39} ]

                {'var':'sex', 'op':'==','val':1} ,
                {'var': 'sex', 'op': '!=', 'val': 1},

                # labtrans.transform(*get_target(df_test)) # line 286
                # File "/home/hmoradi/Downloads/PycharmProject/venv/lib/python3.9/site-packages/pycox/preprocessing/discretization.py",
                # line 58, in bin_numerical if bins.max() == right_cuts.size:
                # File  "/home/hmoradi/Downloads/PycharmProject/venv/lib/python3.9/site-packages/numpy/core/_methods.py", line 39, in _amax
                # return umr_maximum(a, axis, None, out, keepdims, initial, where)
                # ValueError: zero - size  array   to   reduction   operation    maximum    which    has    no    identity

                {'var': 'age', 'op': 'bet', 'val': [40,59]},
                {'var': 'age', 'op': '>=', 'val': 60},

                {'var': 'fmlyinc', 'op': '==', 'val': 1},
                {'var': 'fmlyinc', 'op': '==', 'val': 2},
                {'var': 'fmlyinc', 'op': '==', 'val': 3},
                {'var': 'fmlyinc', 'op': '==', 'val': 4},

                {'var': 'bpmeds', 'op': '==', 'val': 1},
                {'var': 'bpmeds', 'op': '!=', 'val': 1}
          ]
    jhs_analysis = jhs_analysis.sample(frac=1,random_state=12)
    for c in cond: # [cond[0]]
        df_jhs = jhs_analysis.copy()
        df_jhs_tested = jhs_analysis.copy()
        for i in range(k):
            print_log("\nFold: ", i+1)
            df_test = df_jhs_tested.sample(n=int(df_jhs.shape[0] / 10.0))
            df_jhs_tested = df_jhs_tested.drop(df_test.index)
            df_train = df_jhs.drop(df_test.index)
            ### test cond
            print_log("Cond: ", c['var'] + " " + c['op'] + " " + str(c['val']))
            if c['op'] == '==':
                df_test = df_test[df_test[ c['var'] ] == c['val'] ]
            elif c['op'] == '!=':
                df_test = df_test[df_test[ c['var'] ] != c['val'] ]
            elif c['op'] == '<=':
                df_test = df_test[df_test[ c['var'] ] <= c['val'] ]
            elif c['op'] == 'bet':
                df_test = df_test[( (df_test[ c['var'] ] >= c['val'][0])  &  (df_test[ c['var'] ] <= c['val'][1])  )]
            elif c['op'] == '>=':
                df_test = df_test[df_test[ c['var'] ] >= c['val'] ]
            else:
                print_log("Condition Error")
                exit()
            if (df_test.shape[0] == 0 or df_train.shape[0] == 0):
                print_log("variable not exist in this fold")
                continue
            print(df_test.head(2))
            print(df_train.head(2))
            print_log("set_global_data")
            set_global_data(df_train, df_test)
            model_accuracy = eval_model(param_grid)['acc']
            print_log("Model Accuracy: ", model_accuracy)
            kfold_accuracy.loc[c['var']+" "+c['op']+" "+ str(c['val']), str(i + 1) + '-fold'] = model_accuracy
            kfold_accuracy['mean'] = kfold_accuracy.mean(axis=1)
            kfold_accuracy.to_csv(resultsLocation + "Result_10fold_main" + ".csv", index=True)
            print_log(kfold_accuracy.tail(5))

    ### Permutation
    for feature in covariates_all: # [covariates_all[0]]
        df_jhs = jhs_analysis.copy()
        df_jhs[feature] = np.random.permutation(df_jhs[feature])
        df_jhs_tested = df_jhs.copy()
        for i in range(k):
            df_test = df_jhs_tested.sample(n=int(df_jhs.shape[0] / 10.0))
            df_jhs_tested = df_jhs_tested.drop(df_test.index)
            df_train = df_jhs.drop(df_test.index)
            set_global_data(df_train, df_test)
            model_accuracy = eval_model(param_grid)['acc']
            print_log("Model Accuracy: ", model_accuracy)
            kfold_accuracy.loc["Perm. "+feature, str(i + 1) + '-fold'] = model_accuracy
            kfold_accuracy.loc["Perm. "+feature, str(i + 1) + '-fold'] = model_accuracy
            kfold_accuracy['mean'] = kfold_accuracy.mean(axis=1)
            kfold_accuracy.to_csv(resultsLocation + "Result_10fold_permutation" + ".csv", index=True)
            print_log(kfold_accuracy.tail(5))

    kfold_accuracy['mean'] = kfold_accuracy.mean(axis=1)
    kfold_accuracy.to_csv(resultsLocation + "Result_10fold_permutation" + ".csv", index=True)
    print_log(kfold_accuracy.head(7),"\n",kfold_accuracy.head(7))

    # https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/deephit.ipynb
    # https://github.com/chl8856/DeepHit
    # https://github.com/havakv/pycox
    # https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/#dataset

    return()
####################################################################################################################################
def main(): #if __name__ == '__main__':
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
    covariates_main = [x.strip().lower() for x in covariates_main]
    covariates_env = [x.strip().lower() for x in covariates_env]
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


    cleaned_main_cov = [x for x in covariates_all if not x in covariates_env]
    print("\nmain",len(covariates_main),covariates_main)
    print("all",len(covariates_all),covariates_all)
    print("env",len(covariates_env),covariates_env)
    print("new",len(cleaned_main_cov),cleaned_main_cov)
    # exit()
    modeling(covariates_all,outcomes,cleaned_main_cov)
########################################################################################################################################################################################################################################################################