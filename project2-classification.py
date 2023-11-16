import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, zero_one_loss
from tabulate import tabulate
import toolbox_02450


"""
      _       _                   _               
     | |     | |                 | |              
   __| | __ _| |_ __ _   ___  ___| |_ _   _ _ __  
  / _` |/ _` | __/ _` | / __|/ _ \ __| | | | '_ \ 
 | (_| | (_| | || (_| | \__ \  __/ |_| |_| | |_) |
  \__,_|\__,_|\__\__,_| |___/\___|\__|\__,_| .__/ 
                                           | |    
                                           |_|    
"""
#region data setup
df = pd.read_csv('/Users/miroslav/dtu/23E/ML/repo/water_potability.csv').dropna(how='any')
#df.head()

# we have to standardize to use KNN properly
scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]]) # do not scale the class
#df.head()

X = df.drop(columns=['Potability'])
#X.head()
y = df['Potability']
#y.head()

# check prevalence - this is then noted in the text
#y.value_counts()

# for reusable randomness across runs
random_state = 420
#endregion

"""
  _              _                    _               _  __ _           _   _             
 | |            (_)                  | |             (_)/ _(_)         | | (_)            
 | |_ _ __ _   _ _ _ __   __ _    ___| | __ _ ___ ___ _| |_ _  ___ __ _| |_ _  ___  _ __  
 | __| '__| | | | | '_ \ / _` |  / __| |/ _` / __/ __| |  _| |/ __/ _` | __| |/ _ \| '_ \ 
 | |_| |  | |_| | | | | | (_| | | (__| | (_| \__ \__ \ | | | | (_| (_| | |_| | (_) | | | |
  \__|_|   \__, |_|_| |_|\__, |  \___|_|\__,_|___/___/_|_| |_|\___\__,_|\__|_|\___/|_| |_|
            __/ |         __/ |                                                           
           |___/         |___/                                                                                                                                                        
"""
#region classification
# make some demo splits so we can play around
demo_X_train, demo_X_test, demo_y_train, demo_y_test = model_selection.train_test_split(X, y, test_size=.33, random_state=random_state)

# BASELINE MODEL
demo_baseline = DummyClassifier(strategy='most_frequent')
demo_baseline = demo_baseline.fit(demo_X_train, demo_y_train)
print(f'demo_baseline score is {demo_baseline.score(demo_X_test, demo_y_test)}')

# LOGISTIC REGRESSION MODEL
demo_logreg = LogisticRegression().fit(demo_X_train, demo_y_train)
print(f'demo logreg score is {demo_logreg.score(demo_X_test, demo_y_test)} with coefficients')
print(demo_logreg.coef_)


# METHOD 2 - KNN
demo_knn = KNeighborsClassifier().fit(demo_X_train, demo_y_train)
print(f'demo knn score is {demo_knn.score(demo_X_test, demo_y_test)}')
print()
#endregion

"""
  ___        _                _    _______      __
 |__ \      | |              | |  / ____\ \    / /
    ) |_____| | _____   _____| | | |     \ \  / / 
   / /______| |/ _ \ \ / / _ \ | | |      \ \/ /  
  / /_      | |  __/\ V /  __/ | | |____   \  /   
 |____|     |_|\___| \_/ \___|_|  \_____|   \/    
"""
#region cv

#region utility
classifier_names = ['baseline', 'logreg', 'knn']
classifier_params = {
    'baseline': None,
    'logreg': 'C',
    'knn': "n_neighbors"
}
def get_classifier(key):
    if key == "baseline":
        return DummyClassifier(strategy='most_frequent')
    if key == "logreg":
        return LogisticRegression()
    if key == "knn":
        return KNeighborsClassifier()
#endregion utility

K1 = 10
K2 = 5
KFold1 = model_selection.StratifiedKFold(n_splits = K1, shuffle=True, random_state=random_state)
KFold2 = model_selection.StratifiedKFold(n_splits = K2, shuffle=True, random_state=random_state)    

param_grid = {
    'baseline': [{'strategy':['most_frequent']}],
    'logreg': [
        {
            # partially based on https://stackoverflow.com/a/62159222, but adjusted
            'C': [pow(10., exp) for exp in range(3, -3, -1)]
        }
    ],
    'knn': [
        {
            'n_neighbors': range(2, 50)
            # max K was determined by trying range(2,int(pow(len(X), 1/2) * 2)) and never seeing >50
        }
    ]
}

# custom scorer to match the assignment, since https://datascience.stackexchange.com/a/94263
scorer = make_scorer(zero_one_loss, greater_is_better=False)

#region: mostly a reimplementation of Algorithm 6 from the lecture notes (section 10.1.5)
# + generate a table for the report
i = 0
rows = [
    ['Outer fold', 'Method 2 (KNN)', None, 'Logistic Regression', None, 'Baseline'],
    ['i', 'k*', 'Etest', 'λ*', 'Etest', 'Etest']
]

for par_indices, test_indices in KFold1.split(X, y):
    i += 1
    print(f'Computing outer CV fold #{i}')
    row = []

    X_par, y_par = X.iloc[par_indices, :], y.iloc[par_indices]
    X_test, y_test = X.iloc[test_indices, :], y.iloc[test_indices]

    for estimator_key in classifier_names:
        estimator = get_classifier(estimator_key)
        gs = model_selection.GridSearchCV(estimator, param_grid[estimator_key], scoring=scorer, cv=KFold2)
        gs.fit(X_par, y_par)

        y_est = gs.predict(X_test)
        loss = zero_one_loss(y_test, y_est)
        
        print(f'\t{estimator_key}: E_test={loss}\t{gs.best_params_}')

        row.append(round(loss, 3))
        if estimator_key != 'baseline':
            param_value = gs.best_params_[classifier_params[estimator_key]]
            if estimator_key == 'logreg':
                # convert C to lambda
                param_value = 1/param_value
            row.append(param_value)
    
    row.append(i)
    row.reverse()
    rows.append(row)
print()
#endregion 2-level-impl

# now print the nicely formatted table
print(tabulate(rows))

#endregion cv

"""
   _____ _        _   _     _   _           _   ______          _             _   _             
  / ____| |      | | (_)   | | (_)         | | |  ____|        | |           | | (_)            
 | (___ | |_ __ _| |_ _ ___| |_ _  ___ __ _| | | |____   ____ _| |_   _  __ _| |_ _  ___  _ __  
  \___ \| __/ _` | __| / __| __| |/ __/ _` | | |  __\ \ / / _` | | | | |/ _` | __| |/ _ \| '_ \ 
  ____) | || (_| | |_| \__ \ |_| | (_| (_| | | | |___\ V / (_| | | |_| | (_| | |_| | (_) | | | |
 |_____/ \__\__,_|\__|_|___/\__|_|\___\__,_|_| |______\_/ \__,_|_|\__,_|\__,_|\__|_|\___/|_| |_|
"""
#region StatEval

# this is very much based on Method 11.4.1 from the lecture notes (section 11.4.2)
def compare_pairwise(modelA, modelB, loss):
    r = []
    for train_indices, test_indices in KFold1.split(X, y):
        n_j = len(test_indices)

        X_train = X.iloc[train_indices, :]
        y_train = y.iloc[train_indices]

        X_test = X.iloc[test_indices, :]
        y_test = y.iloc[test_indices]

        mA = modelA.fit(X_train, y_train)
        mB = modelB.fit(X_train, y_train)

        yHatA = mA.predict(X_test)
        yHatB = mB.predict(X_test)

        r_j = 0
        for _ in test_indices:
            r_j += loss(y_test, yHatA, yHatB)
        r_j /= n_j
        r.append(r_j)
    
    return toolbox_02450.correlated_ttest(r, 1/K1)

def eval_loss(yReal, yHatA, yHatB):
    return zero_one_loss(yReal, yHatA) - zero_one_loss(yReal, yHatB)

baseline_vs_logreg = compare_pairwise(get_classifier('baseline'), get_classifier('logreg'), eval_loss)
baseline_vs_knn = compare_pairwise(get_classifier('baseline'), get_classifier('knn'), eval_loss)
logreg_vs_knn = compare_pairwise(get_classifier('logreg'), get_classifier('knn'), eval_loss)

eval_rows = [
    [ 'Model 1', 'Model 2', 'p-value', 'CI' ],
    [ 'Baseline', 'Logistic Regression', round(baseline_vs_logreg[0], 5), f'({round(baseline_vs_logreg[1][0], 3)};{round(baseline_vs_logreg[1][1], 3)})' ],
    [ 'Baseline', 'KNN', round(baseline_vs_knn[0], 5), f'({round(baseline_vs_knn[1][0], 3)};{round(baseline_vs_knn[1][1], 3)})' ],
    [ 'Logistic Regression', 'KNN', round(logreg_vs_knn[0], 5), f'({round(logreg_vs_knn[1][0], 3)};{round(logreg_vs_knn[1][1], 3)})' ],
]

print()
print(tabulate(eval_rows))
#endregion StatEval


sec5_logreg = LogisticRegression(C=0.1)
sec5_X_train, sec5_X_test, sec5_y_train, sec5_y_test = model_selection.train_test_split(X, y, test_size=.33, random_state=random_state)

sec5_logreg.fit(sec5_X_train, sec5_y_train)

print()
print('logistic regression coefficients:')
print(sec5_logreg.coef_)