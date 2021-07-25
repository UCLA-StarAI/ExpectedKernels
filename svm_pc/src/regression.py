import numpy as np
from sklearn import svm, tree
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
import json

# DATASET = ['abalone', 'delta-ailerons', 'insurance']
DATASET = ['elevators']

def predict(X, sv, coef, gamma, b):
    # diff = sv - X
    norm2 = np.array([np.linalg.norm(sv[n] - X, axis=1) for n in range(np.shape(sv)[0])])
    dec_func_vec = (coef.dot(np.exp(-gamma*(norm2**2))) + b)
    return dec_func_vec

for dataset in DATASET:
    print(dataset)
    train_x = np.loadtxt('/path/to/repo/svm_pc/data/{}/{}_train_x.csv'.format(dataset, dataset), delimiter=',')
    train_y = np.loadtxt('/path/to/repo/svm_pc/data/{}/{}_train_y.csv'.format(dataset, dataset), delimiter=',')
    valid_x = np.loadtxt('/path/to/repo/svm_pc/data/{}/{}_valid_x.csv'.format(dataset, dataset), delimiter=',')
    valid_y = np.loadtxt('/path/to/repo/svm_pc/data/{}/{}_valid_y.csv'.format(dataset, dataset), delimiter=',')

    if dataset is 'elevators':
        print(train_x.shape)
        train_x = train_x[:3000]
        train_y = train_y[:3000]

    gsc = GridSearchCV(
        estimator=svm.SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    print(train_x.shape)
    grid_result = gsc.fit(train_x, train_y)
    best_params = grid_result.best_params_
    # with open(dataset, 'w') as f:
    #     json.dump(best_params, f)

    best_svr = svm.SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                   coef0=0.1, shrinking=True,
                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)
    
    best_svr.fit(train_x, train_y)
    predict_y = best_svr.predict(valid_x)
    print('\tSVR: ', mean_squared_error(valid_y, predict_y, squared=False))
    print('\tn_SV: ', best_svr._n_support)

    best_params["sv"] = best_svr.support_vectors_.tolist()
    best_params["coef"] = best_svr.dual_coef_.tolist()
    best_params["b"] = best_svr.intercept_.tolist()

    with open(dataset, 'w') as f:
        json.dump(best_params, f)

