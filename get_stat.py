import numpy as np

from Param_JHS import *
from Libraries_JHS import *

####################################################################################################################################

def main():
    jhs_analysis = pd.read_csv(dataLocation + name + "_for-Get-Info.csv")
    jhs_analysis = jhs_analysis.rename(columns={'outcome': 'duration', 'any_inc': 'event'})
    print("jhs_analysis missing: \n", pd.DataFrame([jhs_analysis.isna().sum().values], columns=jhs_analysis.columns.values))
    jhs_analysis.columns = jhs_analysis.columns.str.strip().str.lower()
    all_features = list(x for x in jhs_analysis.columns.values if x not in outcome_features)
    for var in all_features:
        items = jhs_analysis[var].unique()
        if len (items)<5:
            _ = 0
            # print(var,":",str(items))

    # print("\npts onehot")

    jhs_analysis['occupation'] = jhs_analysis['occupation'].apply(lambda x: "employed" if x <= 7 else ("not_employed" if x>7 else np.nan ) )
    jhs_analysis['edu3cat'] = jhs_analysis['edu3cat'].apply(lambda x: "HSgrad" if x > 0 else ("less_HSgrad" if x==0 else np.nan ))

    # for col in ['occupation', 'edu3cat', 'privatepublicins']:  # psychosocial_features
    #     cats= sorted(jhs_analysis[col].unique())
    #     onehot = pd.get_dummies(jhs_analysis[col].astype(pd.CategoricalDtype(categories=cats)), drop_first=False, prefix=str(col))
    #     for y in onehot.columns.values: psychosocial_features.append(y)
    #     psychosocial_features.remove(col)
    #     jhs_analysis = pd.concat([jhs_analysis, onehot], axis=1)
    #     jhs_analysis = jhs_analysis.drop([col], axis=1)

    all_features = list(x for x in jhs_analysis.columns.values if x not in outcome_features)

    col_exceptions = []

    # standardize = [ col for col in all_features if col not in col_exceptions ]
    # scaler = MinMaxScaler().fit(jhs_analysis[standardize])
    # jhs_analysis[standardize] = scaler.transform(jhs_analysis[standardize] )


    data_df = []
    if feature_set=="main": data_df = jhs_analysis[main_features]
    elif feature_set=="psycho": data_df = jhs_analysis[(psychosocial_features+main_features)]
    elif feature_set=="all": data_df = jhs_analysis[all_features]  # Defaults <<<<
    else: print("feature name error!")

    print("\nGet Stat:")
    print(all_features)
    for x in ['sex','age','currentsmoker', 'eversmoker','bmi']:
        print("\n\n####### ",x)
        print("head",jhs_analysis[x].head(20).values)

        print(jhs_analysis[x].describe())
        print("median",jhs_analysis[x].median())
        print("sum",jhs_analysis[x].sum())
        # df['condition']. value_counts()
        print("# event:")
        print(jhs_analysis.loc[jhs_analysis['event']==1,x].describe())
        print("median", jhs_analysis.loc[jhs_analysis['event']==1,x].median())
        print("sum", jhs_analysis.loc[jhs_analysis['event']==1,x].sum())
        print("# not event:")
        print(jhs_analysis.loc[jhs_analysis['event'] != 1, x].describe())
        print("median", jhs_analysis.loc[jhs_analysis['event'] != 1, x].median())
        print("sum", jhs_analysis.loc[jhs_analysis['event'] != 1, x].sum())



    for x in ['edu3cat','occupation','insured']:
        print("\n\n####### ", x)
        print('# cat')
        print(jhs_analysis[x].value_counts())
        print("# event:")
        print(jhs_analysis.loc[jhs_analysis['event'] == 1, x].value_counts())
        print("# not event:")
        print(jhs_analysis.loc[jhs_analysis['event'] != 1, x].value_counts())

        # table = pd.DataFrame()
        # # pts_col_no_treat
        # print("\n\n\n ####Generate Teble #### \n\n\n")
        # counter = 0
        # for x in ['age', 'bmi']:
        #     counter += 1
        #     print(x, end=' ,')
        #     print("mean, ", pts[x].mean(), end=" ,")
        #     print("std, ", pts[x].std(), end=" ,")
        #
        #     table.loc[counter, 'name'] = x
        #     table.loc[counter, 'mean-std'] = str(str(format(pts[x].mean(), ".1f")) + " " + str(
        #         "(" + str(format(pts[x].std(), ".1f")) + ")"))  # pts[x].mean()
        #     # table.loc[counter, 'std'] = pts[x].std()
        #
        #     print()
        #
        #
        # for x in ['gender_female']:
        #     counter += 1
        #     print(x, end=' ,')
        #     # print(pts[x].value_counts())
        #     table.loc[counter, 'name'] = x
        #
        #     for index, value in pts[x].value_counts().iteritems():
        #         tot = pts[x].value_counts().sum()
        #         print(str(index), ", ", str(int(value)), end=", ")
        #         table.loc[counter, str(int(index))] = str(int(value)) + " " + str(
        #             "(" + str(format(((int(value) / int(tot)) * 100), ".1f")) + ")")
        #         # table.loc[counter, str(int(index))+" %"] = format( ((int(value)/int(tot))*100) , ".1f")
        #     print()
        #
        # print('# cat')
        # print(jhs_analysis[x].value_counts())
        # print("# event:")
        # print(jhs_analysis.loc[jhs_analysis['event'] == 1,x].value_counts())
        # print("# not event:")
        # print(jhs_analysis.loc[jhs_analysis['event'] != 1,x].value_counts())
        # for x in ['edu3cat','occupation','insured']:
        #     print("\n\n####### ", x)
        #     print('# cat')
        #     print(jhs_analysis[x].value_counts())
        #     print("# event:")
        #     print(jhs_analysis.loc[jhs_analysis['event'] == 1, x].value_counts())
        #     print("# not event:")
        #     print(jhs_analysis.loc[jhs_analysis['event'] != 1, x].value_counts())

