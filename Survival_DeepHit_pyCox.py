from Param_JHS import *
from Libraries_JHS import *

########################################################################################################################
glob_x_train, glob_x_val, glob_x_test = [],[],[]
glob_y_train, glob_y_val, glob_y_test = [],[],[]
final_features = []
labtrans= []
########################################################################################################################

def print_log(*args):
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
    log = model.fit(glob_x_train, glob_y_train, batch_size, epochs, callbacks, val_data=(glob_x_test,glob_y_test)) # <<<<<<<< val_data=(glob_x_val,glob_y_val)
    # _ = log.plot()

    surv = model.predict_surv_df(glob_x_test)
    # surv = model.interpolate(10).predict_surv_df(glob_x_test)

    durations_test, events_test = glob_y_test

    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    acc =  ev.concordance_td('antolini')

    return({'loss': -acc, 'status': STATUS_OK, 'test':['test'], 'log':log, 'acc':acc, 'model':model})
########################################################################################################################
def set_global_data(df_train,df_test):

    global glob_x_train, glob_x_val, glob_x_test
    global glob_y_train, glob_y_val, glob_y_test
    global labtrans

    all_features = list(x for x in df_train.columns.values if x not in outcome_features)

    # main_features # psychosocial_features # all_features # outcome_features #
    if feature_set == "main":
        df_train = df_train[(main_features+outcome_features)]
        df_test = df_test[(main_features+outcome_features)]
    elif feature_set == "psycho+main":
        df_train = df_train[(psychosocial_features + main_features+outcome_features)]
        df_test = df_test[(psychosocial_features + main_features+outcome_features)]
    elif feature_set == "psycho":
        df_train = df_train[(psychosocial_features+outcome_features )]
        df_test = df_test[(psychosocial_features+outcome_features )]
    elif feature_set == "all":
        df_train = df_train[(all_features+outcome_features)]  # Defaults <<<<
        df_test = df_test[(all_features+outcome_features)]
    elif feature_set == "env":
        main_plus_psycho = main_features + psychosocial_features
        others = [x for x in all_features if x not in main_plus_psycho]
        df_train = df_train[(others+outcome_features)]  # Defaults <<<<
        df_test = df_test[(others+outcome_features)]
    elif feature_set == "env+psycho":
        others = [x for x in all_features if x not in main_features]
        df_train = df_train[(others+outcome_features)]  # Defaults <<<<
        df_test = df_test[(others+outcome_features)]
    else:
        print("feature name error!")


    cols_standardize = df_train.columns.values.tolist()
    cols_standardize.remove('duration')
    cols_standardize.remove('event')
    global final_features
    final_features = cols_standardize
    standardize = [([col], MinMaxScaler()) for col in cols_standardize] # StandardScaler
    cols_leave = [x for x in df_train.columns.values if x not in cols_standardize]
    leave = [(col, None) for col in cols_leave]
    x_mapper = DataFrameMapper(standardize)  # + leave
    x_train = x_mapper.fit(df_train)
    df_val = 0 # df_train.sample(frac=0.2) # change from 0.3 # <<<<<<<<<<
    # df_train = df_train.drop(df_val.index)
    x_train = x_mapper.transform(df_train).astype('float32')
    x_val = 0 # x_mapper.transform(df_val).astype('float32') # <<<<<<<<<<
    x_test = x_mapper.transform(df_test).astype('float32')

    num_durations = max_incident_years
    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = 0 # labtrans.transform(*get_target(df_val)) # <<<<<<<<<<
    y_test = labtrans.transform(*get_target(df_test))

    glob_x_train, glob_x_val, glob_x_test = x_train, x_val, x_test
    glob_y_train, glob_y_val, glob_y_test = y_train, y_val, y_test
########################################################################################################################
def modeling():

    global glob_x_train, glob_x_val, glob_x_test
    global glob_y_train, glob_y_val, glob_y_test
    global labtrans

    global outcome_features
    jhs_analysis = pd.read_csv(dataLocation + name + ".csv")
    jhs_analysis.columns = jhs_analysis.columns.str.strip().str.lower()
    all_features = list(x for x in jhs_analysis.columns.values if x not in outcome_features)
    jhs_analysis = jhs_analysis.rename(columns={'outcome': 'duration', 'any_inc': 'event'})
    outcome_features = ['duration','event']

    start = datetime.now()
    print_log("datetime.now()", datetime.now())

    for var in all_features:
        items = jhs_analysis[var].unique()
        if len(items) < 5:
            _ = 0
            # print(var,":",str(items))

    print("\npts onehot")  # .apply(lambda x:1 if i<100 else 2 if i>100 else 0)

    # pts.loc[pts['bmi'].isna(), 'bmi'] = pts['bmi'].mean()  # 50%
    # print(jhs_analysis['occupation'],jhs_analysis['edu3cat'],jhs_analysis['privatepublicins'])

    # jhs_analysis= jhs_analysis.astype({"occupation": 'str', "edu3cat": 'str'})

    jhs_analysis['occupation'] = jhs_analysis['occupation'].apply(lambda x: "employed" if x <= 7 else "not_employed")
    jhs_analysis['edu3cat'] = jhs_analysis['edu3cat'].apply(lambda x: "HSgrad" if x > 0 else "less_HSgrad") # .apply(lambda x: "employed" if x <= 7 else "not_employed")

    # jhs_analysis.loc[jhs_analysis['occupation'].astype(float) <=7 , 'occupation'] = "employed"
    # jhs_analysis.loc[jhs_analysis['occupation'].astype(float) > 7, 'occupation'] = "not_employed"

    # jhs_analysis.loc[jhs_analysis['edu3cat'].astype(float) >0 , 'edu3cat'] = "HSgrad"
    # jhs_analysis.loc[jhs_analysis['edu3cat'].astype(float) == 0, 'edu3cat'] = "less_HSgrad"

    # jhs_analysis['privatepublicins'] = jhs_analysis['privatepublicins'].astype(str)
    # jhs_analysis.loc[(jhs_analysis['privatepublicins'].astype(float) == 1) | (jhs_analysis['privatepublicins'].astype(float) ==3 ) , 'privatepublicins'] = "private"
    # jhs_analysis.loc[(jhs_analysis['privatepublicins'].astype(float) == 2) | (jhs_analysis['privatepublicins'].astype(float) == 3), 'privatepublicins'] = "public"

    for col in ['occupation', 'edu3cat', 'privatepublicins']:   # psychosocial_features
        cats= sorted(jhs_analysis[col].unique())
        print(cats)
        onehot = pd.get_dummies(jhs_analysis[col].astype(pd.CategoricalDtype(categories=cats)), drop_first=False, prefix=str(col))
        jhs_analysis = pd.concat([jhs_analysis, onehot], axis=1)
        jhs_analysis = jhs_analysis.drop([col], axis=1)

        for y in onehot.columns.values: psychosocial_features.append(y)
        psychosocial_features.remove(col)
    all_features = list(x for x in jhs_analysis.columns.values if x not in outcome_features)
    print(all_features)
    print(len(all_features))
    # exit()

    df_train = jhs_analysis.copy()

    if eval_correlation:
        # plt.matshow(jhs_analysis.corr())
        # plt.show()

        # corr = jhs_analysis.corr()
        # plt.matshow(corr.style.background_gradient(cmap='coolwarm'))
        # plt.show()

        corr_get = jhs_analysis.corr()[ (jhs_analysis.corr()>0.5) | (jhs_analysis.corr()<(-0.5))  ].iloc[:30]
        # corr_get = corr.iloc[:30]
        sns.heatmap(corr_get,
                    xticklabels=corr_get.columns.values,
                    yticklabels=corr_get.columns.values)
        plt.show()
        exit()

    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)
    set_global_data(df_train, df_test)

    # TODO: param

    num_layers = hp.choice('num_layers', [1, 2, 3, 4, 5])
    num_neurons = hp.choice('num_neurons', [25, 50, 75, 100,150])
    batch_norm = hp.choice('batch_norm', [True, False])
    dropout = hp.choice('dropout', [0.1, 0.2, 0.3, 0.4, 0.5,0.6])
    learning_rate = hp.choice('learning_rate', [0.01, 0.001, 0.0001])
    batch_size = hp.choice('batch_size', [4, 16, 32, 64, 128,256])
    epochs = hp.choice('epochs', [4, 16, 32, 64, 128, 256 ])
    # momentu = param_grid['']
    # actiovation_func = param_grid['']
    # loss =  param_grid['']
    # optimizer = param_grid['']

    param_grid = dict(num_layers=num_layers,
                      num_neurons=num_neurons,
                      batch_norm=batch_norm, dropout=dropout,
                      learning_rate=learning_rate, batch_size=batch_size,
                      epochs=epochs)

    tpe_trials = Trials()
    tpe_best = []
    model = []
    if optimize:
        tpe_best = fmin(fn=eval_model, space=param_grid, algo=tpe.suggest, max_evals=max_eval, trials=tpe_trials,
                    rstate=np.random.RandomState(14))
        param_grid= space_eval(param_grid, tpe_best)
        print_log("Best: ", tpe_best, getXfromBestModelfromTrials(tpe_trials, 'acc'), param_grid)
        model = getXfromBestModelfromTrials(tpe_trials, 'model')
        print_log("datetime.now(): ", datetime.now())
    else:
        param_grid = {'batch_norm': True, 'batch_size': 64, 'dropout': 0.4, 'epochs': 128, 'learning_rate': 0.001, 'num_layers': 4, 'num_neurons': 50}
        if feature_set== 'all':
            param_grid ={'batch_norm': True, 'batch_size': 256, 'dropout': 0.1, 'epochs': 64, 'learning_rate': 0.01, 'num_layers': 4, 'num_neurons': 75}
            # model = eval_model(param_grid)['model']

    # TODO: ### SHAP
    if graph_shap:
        print('\n***Shap')
        if calc_shap:
            set_global_data(jhs_analysis.copy(), jhs_analysis.copy())
            res = eval_model(param_grid)
            model=res['model']
            print_log("overTrained Model Accuracy: ", res['acc'])
            explainer = shap.KernelExplainer(model.predict, shap.kmeans(glob_x_train, kmeans_k)  )
            # 11
            number_of_rows = glob_x_test.shape[0]
            random_indices = np.random.choice(number_of_rows, size=number_of_rows//rows_devideby_to_use, replace=False)
            random_rows = glob_x_test[random_indices, :]

            shap_values = explainer.shap_values(random_rows)#    glob_x_test,nsamples=100     ) # [:5,:]

            print('training size:', len(glob_x_train), len(glob_x_train[0]))
            print('features:', len(final_features))
            print('\nD1 Classes:', len(shap_values), '\nD2 samples:', len(shap_values[0]), '\nD3 Columns/features:', len(shap_values[0][0]), '\nvalue:', shap_values[0][0][0])
            print('type: ',type(shap_values))
            print('type [0]: ', type(shap_values[0]))
            # write ################################
            for i in range(len(shap_values)):
                np.savetxt("shap_"+str(i)+".csv", shap_values[i])
            np.savetxt("shape.csv",np.array([len(shap_values)]))
            exit()
        # read ###################################
        set_global_data(jhs_analysis.copy(), jhs_analysis.copy())
        arr_shape = np.loadtxt("Results/shape.csv")
        shap_values = [ [] for i in range(int(arr_shape))]
        for i in range(int(arr_shape)):
            shap_values[i] = np.loadtxt("Results/shap_"+str(i)+".csv")

        print('\nD1 Classes:',len(shap_values),'\nD2 samples:', len(shap_values[0]),'\nD3 Columns/features:',len(shap_values[0][0]),'\nvalue:',shap_values[0][0][0])
        print('type: ',type(shap_values))
        print('type [0]: ', type(shap_values[0]))

        # shap.summary_plot(shap_values, glob_x_train, feature_names=final_features, plot_type="bar")
        overall_classes = np.array( [ 0 for _ in range(len(shap_values[0][0]))  ] )
        overall_classes = np.array( [overall_classes  for _ in range(len(shap_values[0]))   ] )
        for x in range(len(shap_values)) :
            overall_classes = np.add(overall_classes,shap_values[x] )  # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        overall_classes = np.divide(overall_classes, len(shap_values)) # ToDO: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        print('overall_classes.shape', overall_classes.shape)

        overall_variables = np.array([0 for _ in range(len(shap_values[0][0]))])
        for x in range(overall_classes.shape[0]) :
            overall_variables = np.add(overall_variables,overall_classes[x] )
        overall_variables = np.divide(overall_variables, overall_classes.shape[0])
        print('overall_variables.shape',overall_variables.shape)

        df = pd.DataFrame(overall_variables, columns=['values'],index=final_features)
        print(df.head(),"\n",str(list(df.index)))



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
        # df = df.drop('occupation_2.58509567', 0)
        # geo_risk_dict = {'fakecensustractid':'Fake census tract ID'}
        standard_risk_dict = {'sbp':'Systolic blood pressure', 'dbp':'Diastolic blood pressure', 'abi':'Ankle brachial index', 'bpmeds':'Blood pressure medication status',
                            'totchol':'Total cholesterol','ldl':'LDL cholesterol','hdl':'HDL cholesterol ', 'trigs':'Triglycerides ', 'fpg':'Fasting glucose',
                            'hba1c':'Hemoglobin A1C', 'alc':'Alcohol drinking in the past 12 months','alcw':'Average number of drinks per week',
                            'currentsmoker':'Cigarette Smoking Status','eversmoker':'History of Cigarette Smoking',
                            'pa3cat': 'Physical Activity', 'activeindex':'Physical activity during leisure time','nutrition3cat':'Nutrition Categorization',

                            'sex': 'Sex - Male', 'age': 'Age','waist':'Waist','bmi':'BMI', #  Demographic
                            }
        # demo + insurance + psycho
        scio_risk_dict = { # Psychosocial
                          # .apply(lambda x: "HSgrad" if x > 0 else "less_HSgrad") # .apply(lambda x: "employed" if x <= 7 else "not_employed")

                          # 'occupation_1.0': 'occupation - Management/Professional', 'occupation_2.0': 'occupation - Service', 'occupation_3.0': 'occupation - Sales',
                          # 'occupation_4.0': 'occupation - Farming', 'occupation_5.0': 'occupation - Construction',
                          # 'occupation_6.0': 'occupation - Production', 'occupation_7.0': 'occupation - Military', 'occupation_8.0': 'occupation - Sick',
                          # 'occupation_9.0': 'occupation - Unemployed', 'occupation_10.0': 'occupation - Homemaker', 'occupation_11.0': 'occupation - Retired',
                          # 'occupation_12.0': 'occupation - Student', 'occupation_13.0': 'occupation - Other',

                          # 'edu3cat_0':'Education - Less thank high school','edu3cat_1':'Education - LHigh school graduate/GED','edu3cat_2':'Education - attended vocational or colledge',

                          'dailydiscr':'Daily discrimination','lifetimediscrm':'Lifetime discrimination','discrmburden':'Discrimination burden',


                          'depression': 'Depressive Symptoms Score', 'weeklystress': 'Weekly stress score', 'perceivedstress': 'Global Stress Score',
                          }
        socioeconomic = {'insured':'Insured', # insurance
                          'privatepublicins_0.0': 'Insurance status - Uninsured', 'privatepublicins_1.0': 'Insurance status - Private Only',
                          'privatepublicins_2.0': 'Insurance status - Public Only', 'privatepublicins_3.0': 'Insurance status - Private & Public',
                         'fmlyinc': 'Family income',
                         'occupation_employed': 'Employment Status (Employed)',
                         'occupation_not_employed': 'Employment Status (Not Employed)',
                         'edu3cat_HSgrad': 'Education - High school graduated',
                         'edu3cat_less_HSgrad': 'Education - Less than high school Graduate',
        }
        for x in df.index:
            if x in env_disc['Variable Name'].values:
                # print(env_disc.loc[env_disc['Variable Name']==x] )
                df.loc[x,'label']= str(env_disc.loc[env_disc['Variable Name']==x,'Label'].values[0]).split('\n')[0] #+ ' ('+str(x)+')'
                df.loc[x, 'color'] = 2
                df.loc[x, 'color_label'] = 'Environmental'
            elif x.split('_')[0] in env_disc['Variable Name'].values:
                print('First part exist: ',x)
                # print(env_disc.loc[env_disc['Variable Name'] == x.split('_')[0]])
                df.loc[x, 'label'] = str(env_disc.loc[env_disc['Variable Name']==x.split('_')[0],'Label'].values[0]).split('\n')[0] #+ ' ('+str(x)+')'
                df.loc[x, 'color'] = 2
                df.loc[x, 'color_label'] = 'Environmental'
            # elif x in geo_risk_dict.keys():
            #     df.loc[x, 'label'] = geo_risk_dict[x]+' ('+str(x)+')'
            #     df.loc[x, 'color'] = 2
            #     df.loc[x, 'color_label'] = 'Environmental'
            elif x in standard_risk_dict.keys():
                df.loc[x, 'label'] = standard_risk_dict[x]#+' ('+str(x)+')'
                df.loc[x, 'color'] = 3
                df.loc[x, 'color_label'] = 'Standard'
            elif x in scio_risk_dict.keys():
                df.loc[x, 'label'] = scio_risk_dict[x]#+' ('+str(x)+')'
                df.loc[x, 'color'] = 0
                df.loc[x, 'color_label'] = 'Psychosocial'
            elif x in socioeconomic.keys():
                df.loc[x, 'label'] = socioeconomic[x]#+' ('+str(x)+')'
                df.loc[x, 'color'] = 5
                df.loc[x, 'color_label'] = 'Socioeconomic'
            else:
                no_desc+=1
                print('no_desc: '+str(no_desc)+" "+x)


        df['label'] = df['label'].str.replace("  "," ").replace("\n"," ").replace("\t"," ")
        # print(df.dropna().head())
        # exit()

        palette = {"Environmental": palette[2], "Standard": palette[3], "Psychosocial": palette[0], "Socioeconomic":palette[9]} # 7 grey 5 dark red
        hue_order = [ "Standard", "Environmental", "Socioeconomic", "Psychosocial"]
        # ### Positive
        # my_dpi = 200
        # df_filtered = df[df['values']>=0].sort_values(by='values', ascending=False).reset_index(drop=True).copy()
        # for i in range(0,df_filtered.shape[0],51):
        #     partial_df = df_filtered.iloc[i:i+50].copy()
        #     print("partial_df: ", partial_df.shape)
        #     plt.figure(figsize=(400 / my_dpi, (1400 / my_dpi)*((partial_df.shape[0]+10)/(51+10))   ), dpi=my_dpi)
        #     # sns.set(style="ticks")
        #     sns.set_style("darkgrid",{"axes.facecolor": ".9"})
        #     # sns.set_context("paper")
        #     sns.barplot(y='label', x="values", data=partial_df,hue='color_label',dodge=False,palette=palette,hue_order=hue_order)#,palette=[palette[i] for i in partial_df.color.astype(int)])
        #     # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
        #     # plt.tight_layout()
        #     plt.legend(title="Predictors",loc='lower right')
        #     plt.xlabel('Mean SHAP (average impact on model output magnitude)',Fontsize=12 )
        #     plt.ylabel('Variables',Fontsize=12)
        #     # plt.show()
        #     plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
        #     plt.grid()
        #     plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
        #     plt.savefig(dataLocation+"Figs/"+"Positive-"+str(i),  bbox_inches="tight",pad_inches=0.3) # facecolor='y', , transparent=True, dpi=200
        #     plt.clf()
        #     # exit()
        #
        # ### Negative
        # plt.clf()
        # plt.close()
        # df_filtered = df[df['values'] < 0].sort_values(by='values', ascending=False).reset_index(drop=True).copy()
        # for i in range(0, df_filtered.shape[0], 51):
        #     partial_df = df_filtered.iloc[i:i + 50].copy()
        #     print("partial_df: ", partial_df.shape)
        #     plt.figure(figsize=(400 / my_dpi, (1400 / my_dpi)*((partial_df.shape[0]+10)/(51+10))   ), dpi=my_dpi)
        #     # sns.set(style="ticks")
        #     sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        #     # sns.set_context("paper")
        #     sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
        #                 dodge=False,palette=palette,hue_order=hue_order)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
        #     # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
        #     # plt.tight_layout()
        #     plt.legend(title="Predictors", loc='upper left')
        #     plt.xlabel('Mean SHAP (average impact on model output magnitude)', Fontsize=12)
        #     plt.ylabel('Variables', Fontsize=12)
        #     # plt.show()
        #     plt.xlim(df_filtered['values'].min()* 1.02 , df_filtered['values'].max() + 0.0002)
        #     plt.grid()
        #     plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
        #     plt.savefig(dataLocation + "Figs/" + "Negative-" + str(i), bbox_inches="tight",
        #                 pad_inches=0.3)  # facecolor='y', , transparent=True, dpi=200
        #     plt.clf()
        #     # exit()


        my_dpi = 200
        ### abs
        plt.clf()
        plt.close()
        df_filtered = df.copy()
        df_filtered['values'] = df_filtered['values'].abs()
        df_filtered = df_filtered.sort_values(by='values', ascending=False).reset_index(drop=True)
        df_filtered['label']= df_filtered['label'].apply(lambda x: x.capitalize())
        df_filtered['label'] = df_filtered['label'].apply(lambda x: x.replace("Ldl","LDL").replace(" a1c "," A1C ")
                                                          .replace("+instructional+w","+w").replace("(not employed)","- not employed")
                                                          .replace("Hdl", "HDL")
                                                          )
        print(df_filtered.head())
        for i in range(0, df_filtered.shape[0], 51):

            import matplotlib as mpl
            mpl.rcParams['font.family'] = 'Arial'
            sns.set(font="Arial")

            partial_df = df_filtered.iloc[i:i + 50].copy()
            print("partial_df: ", partial_df.shape)
            plt.figure(figsize=(600 / my_dpi, (2000 / my_dpi)*((partial_df.shape[0]+10)/(51+10))  ), dpi=my_dpi)
            # sns.set(style="ticks")
            sns.set_style("darkgrid", {"axes.facecolor": ".9"})
            # sns.set_context("paper")
            ax = sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
                        dodge=False,hue_order=hue_order,palette=palette)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
            # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
            # plt.tight_layout()
            plt.legend(title="Predictors", loc='lower right', prop={'size': 23})
            plt.xlabel('Mean |SHAP| (average impact on model output magnitude)', Fontsize=12)
            plt.ylabel('Variables', Fontsize=12, rotation=0)
            # plt.show()
            plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
            plt.grid()

            import matplotlib as mpl
            mpl.rcParams['font.family'] = 'Arial'
            sns.set(font="Arial")
            legend = plt.legend( loc='lower right',prop={'size': 13})
            frame = legend.get_frame()
            frame.set_facecolor('white')
            plt.axhline(y=14.5, color='r', linestyle='-')
            ax.yaxis.set_label_coords(-1.1,1.02)
            # ax.set_ylabel() # position=(x, y)
            # ax.tick_params(axis='y', rotation=90)

            print("write")
            plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
            plt.savefig(dataLocation + "Figs\\" + "Abs-" + str(i)+'.svg', bbox_inches="tight",
                        pad_inches=0.3, format='svg')  # facecolor='y', , transparent=True, dpi=200 , format='eps'
            # plt.savefig(dataLocation + "Figs/" + "Abs-" + str(i), bbox_inches="tight",
            #             pad_inches=0.3)
            plt.clf()
        categories = [ "Standard", "Environmental", "Socioeconomic", "Psychosocial"]
        for x in [ "Standard", "Environmental", "Socioeconomic", "Psychosocial"]:
            # df.loc[x,'label']= str(env_disc.loc[env_disc['Variable Name']==x,'Label']
            # df.loc[x, 'color_label'] = 'Socioeconomic'
            df_category = df_filtered[df_filtered['color_label']==x]
            for i in range(0, df_category.shape[0], 51):
                partial_df = df_category.iloc[i:i + 50].copy()

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                print("partial_df: ", partial_df.shape)
                print(len(palette),len(categories),categories.index(x))
                plt.figure(figsize=(600 / my_dpi, (2000 / my_dpi) * ((partial_df.shape[0] + 10) / (51 + 10))),
                           dpi=my_dpi)
                # sns.set(style="ticks")
                sns.set_style("darkgrid", {"axes.facecolor": ".9"})
                # sns.set_context("paper")
                ax = sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
                                 dodge=False, #hue_order=hue_order,
                                 palette=palette)  # [categories.index(x)+1] ,palette=[palette[i] for i in partial_df.color.astype(int)])
                # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
                # plt.tight_layout()
                plt.legend(title="Predictors", loc='lower right', prop={'size': 23})
                plt.xlabel('Mean |SHAP| (average impact on model output magnitude)', Fontsize=12)
                plt.ylabel('Variables', Fontsize=12, rotation=0)
                # plt.show()
                plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
                plt.grid()

                import matplotlib as mpl
                mpl.rcParams['font.family'] = 'Arial'
                sns.set(font="Arial")
                legend = plt.legend(loc='lower right', prop={'size': 13})
                frame = legend.get_frame()
                frame.set_facecolor('white')
                # plt.axhline(y=14.5, color='r', linestyle='-')
                ax.yaxis.set_label_coords(-1.1, 1.02)
                # ax.set_ylabel() # position=(x, y)
                # ax.tick_params(axis='y', rotation=90)

                print("write")
                plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
                plt.savefig(dataLocation + "Figs\\" + "cat-"+str(x)+"-" + str(i) + '.svg', bbox_inches="tight",
                            pad_inches=0.3, format='svg')
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                break


        df_filtered.to_csv(dataLocation +"Figs/" + "Result_shap_abs_sorted" + ".csv", index=True)

            # exit()





        # ### Similarity
        # my_dpi = 100
        # df_filtered = df.sort_index(ascending=False).reset_index(drop=True).copy()
        # for i in range(0, df_filtered.shape[0], 51):
        #     partial_df = df_filtered.iloc[i:i + 50].copy()
        #     print("partial_df: ", partial_df.shape)
        #     plt.figure(figsize=(400 / my_dpi,  (1400 / my_dpi)*((partial_df.shape[0]+10)/(51+10))   ), dpi=my_dpi)
        #     # sns.set(style="ticks")
        #     sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        #     # sns.set_context("paper")
        #     sns.barplot(y='label', x="values", data=partial_df, hue='color_label',
        #                 dodge=False,hue_order=hue_order,palette=palette)  # ,palette=[palette[i] for i in partial_df.color.astype(int)])
        #     # plt.xticks(fontsize=8,rotation=90)# ax.tick_params(axis='both', which='major', labelsize=10)
        #     # plt.tight_layout()
        #     plt.legend(title="Predictors", loc='lower right')
        #     plt.xlabel('Mean SHAP (average impact on model output magnitude)', Fontsize=12)
        #     plt.ylabel('Variables', Fontsize=12)
        #     # plt.show()
        #     plt.xlim(df_filtered['values'].min() - 0.0001, df_filtered['values'].max() * 1.02)
        #     plt.grid()
        #     plt.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)  # right=0.9, top=0.9, bottom=0.1
        #     plt.savefig(dataLocation + "Figs/" + "Similarity-" + str(i), bbox_inches="tight",
        #                 pad_inches=0.3)  # facecolor='y', , transparent=True, dpi=200
        #     plt.clf()
        #     # exit()

        # shap.summary_plot(shap_values[target_shap_class], shap_testing, feature_names=pts_col)
        plt.figure(figsize=(1200 / my_dpi, (1200 / my_dpi)), dpi=my_dpi)
        print(overall_classes.shape)
        print(glob_x_test.shape)
        print(glob_x_train.shape)
        print(len(final_features))
        # shap.summary_plot(overall_classes, glob_x_test, feature_names=final_features)
        # 11

        df.to_csv(resultsLocation + "Result_shap_" + name +".csv", index=True)
        exit()

        df = df.sort_index(ascending=False) # key=lambda x: x.str.lower(),
        # df['labels'] = df.index.str.lower()
        # df = df.sort_values('labels').drop('labels', axis=1)
        sns.barplot(y=df.index, x="values", data=df)
        plt.xticks(fontsize=8, rotation=90)  # ax.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        # plt.show()



        df.to_csv(resultsLocation + "Result_shap_"+name + ".csv", index=True)
        exit()





    # print_log("Final Model")
    # print_log("Final Model Accuracy: ", eval_model(param_grid)['acc'])

    # TODO: ### Cross-validation

    kfold_accuracy = pd.DataFrame(columns= [ str(k+1)+'-fold' for k in range(10)],index=['overall'])
    df_jhs = jhs_analysis.copy()
    df_jhs_tested = jhs_analysis.copy()
    for i in range(k_fold_croos_validation):
        df_test = df_jhs_tested.sample(n=int(df_jhs.shape[0]/10.0))
        df_jhs_tested= df_jhs_tested.drop(df_test.index)
        df_train = df_jhs.drop(df_test.index)
        set_global_data(df_train,  df_test)
        model_accuracy = eval_model(param_grid)['acc']
        print_log("Model Accuracy: ", model_accuracy)
        kfold_accuracy.loc['overall',str(i+1)+'-fold']= model_accuracy
        kfold_accuracy['mean'] = kfold_accuracy.mean(axis=1)
        kfold_accuracy.to_csv(resultsLocation + "Result_10fold_main_"+name +"_" +algorithm+"_" +feature_set + ".csv", index=True)
        print_log(kfold_accuracy.tail(5))
    print_log("not tested: df_jhs_tested ", df_jhs_tested.shape[0])
    print("DeepHitAnalysis")
    # with HF
    # main: 0.77
    # psycho: 0.76
    # all: 0.76
    # NO HF
    # main_features # 0.77
    # psychosocial_features+main_features # 0.77
    # all_features # 0.76


    exit(0)

    # TODO: ### Conditional
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
        for i in range(k_fold_croos_validation):
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
            kfold_accuracy.to_csv(resultsLocation + "Result_10fold_main_conditions_"+name + ".csv", index=True)
            print_log(kfold_accuracy.tail(5))

    # TODO: ### Permutation
    for feature in covariates_all: # [covariates_all[0]]
        df_jhs = jhs_analysis.copy()
        df_jhs[feature] = np.random.permutation(df_jhs[feature])
        df_jhs_tested = df_jhs.copy()
        for i in range(k_fold_croos_validation):
            df_test = df_jhs_tested.sample(n=int(df_jhs.shape[0] / 10.0))
            df_jhs_tested = df_jhs_tested.drop(df_test.index)
            df_train = df_jhs.drop(df_test.index)
            set_global_data(df_train, df_test)
            model_accuracy = eval_model(param_grid)['acc']
            print_log("Model Accuracy: ", model_accuracy)
            kfold_accuracy.loc["Perm. "+feature, str(i + 1) + '-fold'] = model_accuracy
            kfold_accuracy.loc["Perm. "+feature, str(i + 1) + '-fold'] = model_accuracy
            kfold_accuracy['mean'] = kfold_accuracy.mean(axis=1)
            kfold_accuracy.to_csv(resultsLocation + "Result_10fold_permutation_"+name+ ".csv", index=True)
            print_log(kfold_accuracy.tail(5))

    kfold_accuracy['mean'] = kfold_accuracy.mean(axis=1)
    kfold_accuracy.to_csv(resultsLocation + "Result_10fold_permutation_"+name + ".csv", index=True)
    print_log(kfold_accuracy.head(7),"\n",kfold_accuracy.head(7))

    return()
####################################################################################################################################
def main(): #if __name__ == '__main__':
    ############################################################################
    print("\nDeep_hit:")
    modeling()
####################################################################################################################################