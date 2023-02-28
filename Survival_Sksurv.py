from Param_JHS import *
from Libraries_JHS import *

####################################################################################################################################
def modeling():


    jhs_analysis = pd.read_csv(dataLocation + name + ".csv")
    jhs_analysis.columns = jhs_analysis.columns.str.strip().str.lower()
    all_features = list(x for x in jhs_analysis.columns.values if x not in outcome_features)
    for var in all_features:
        items = jhs_analysis[var].unique()
        if len (items)<5:
            _ = 0
            # print(var,":",str(items))

    print("\npts onehot")

    jhs_analysis['occupation'] = jhs_analysis['occupation'].apply(lambda x: "employed" if x <= 7 else "not_employed")
    jhs_analysis['edu3cat'] = jhs_analysis['edu3cat'].apply(lambda x: "HSgrad" if x > 0 else "less_HSgrad")

    for col in ['occupation', 'edu3cat', 'privatepublicins']:  # psychosocial_features
        cats= sorted(jhs_analysis[col].unique())
        onehot = pd.get_dummies(jhs_analysis[col].astype(pd.CategoricalDtype(categories=cats)), drop_first=False, prefix=str(col))
        for y in onehot.columns.values: psychosocial_features.append(y)
        psychosocial_features.remove(col)
        jhs_analysis = pd.concat([jhs_analysis, onehot], axis=1)
        jhs_analysis = jhs_analysis.drop([col], axis=1)

    all_features = list(x for x in jhs_analysis.columns.values if x not in outcome_features)

    col_exceptions = []
    # standardize = [([col], MinMaxScaler()) for col in all_features if col not in col_exceptions]  # StandardScaler
    # leave = [(col, None) for col in col_exceptions]
    # x_mapper = DataFrameMapper(standardize + leave)  # + leave
    # jhs_analysis_scaled = x_mapper.fit_transform(jhs_analysis).astype('float32')
    standardize = [ col for col in all_features if col not in col_exceptions ]
    scaler = MinMaxScaler().fit(jhs_analysis[standardize])
    jhs_analysis[standardize] = scaler.transform(jhs_analysis[standardize] )

    # main_features # psychosocial_features # all_features # outcome_features #
    data_df = []
    if feature_set=="main": data_df = jhs_analysis[main_features]
    elif feature_set=="psycho": data_df = jhs_analysis[(psychosocial_features+main_features)]
    elif feature_set=="all": data_df = jhs_analysis[all_features]  # Defaults <<<<
    else: print("feature name error!")
    target_df = jhs_analysis[['outcome','any_inc']]

    eve_sur = pd.concat([target_df[['any_inc']].astype('bool'),target_df[['outcome']].astype('float64')],axis=1)
    eve_sur= [tuple(i) for i in eve_sur.values]
    eve_sur = np.asarray(eve_sur,dtype=[('any_inc', np.bool), ('outcome', np.float64)])

    if algorithm == 'sksurv_cox_rf':
        print("RandomSurvivalForest")
        # rsf = RandomSurvivalForest(n_estimators=100, # 1000
        #                            min_samples_split=10,
        #                            min_samples_leaf=15,
        #                            max_features="sqrt",
        #                            random_state=7)
        rsf = RandomSurvivalForest(max_features="sqrt",random_state=7)

        scores = cross_val_score( rsf, data_df, eve_sur, cv = 10, scoring = score_survival_model)
        np.savetxt(resultsLocation + "Result_10fold_main_" + name + "_" + algorithm + "_" + feature_set + ".csv",scores, delimiter=",")
        print(scores)
        print(np.average(scores))

    if algorithm == 'sksurv_cox_cox':
        print("CoxPHSurvivalAnalysis")
        # estimator = CoxPHSurvivalAnalysis(alpha=0.001) # reduce one zero
        estimator = CoxPHSurvivalAnalysis(alpha=0.0001)

        scores = cross_val_score( estimator, data_df, eve_sur, cv = 10, scoring = score_survival_model)
        np.savetxt(resultsLocation + "Result_10fold_main_"+name +"_" +algorithm+"_" +feature_set + ".csv",scores, delimiter=",")
        print(scores)
        print(np.average(scores))

    return()
####################################################################################################################################
from sksurv.metrics import concordance_index_censored
def score_survival_model(model, X, y):
    # Harrell’s concordance index
    prediction = model.predict(X)
    result = concordance_index_censored(y['any_inc'], y['outcome'], prediction)
    return result[0]
####################################################################################################################################
def main(): # if __name__ == '__main__':

    print("\nSksurv main:")
    modeling()
####################################################################################################################################