import numpy as np 
import pandas as pd 
import pickle
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score, 
    roc_curve,
    confusion_matrix
)


pkl_file = './experiment.pkl.2'
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)


def evaluate(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    j_best_idx = np.argmax(youden_j)
    best_threshold = thresholds[j_best_idx]
    y_pred_best = (y_pred >= 0.5).astype(int)
    # y_pred_best = (y_pred >= best_threshold).astype(int)
    f1 = f1_score(y_true, y_pred_best)
    acc = accuracy_score(y_true, y_pred_best)
    return auc, f1, acc

df_data = []
for k in data:
    a, b, c, seed = k.split('_')
    results = data[k]['outputs']
    df = data[k]['df']
    # evaluate eta0
    ind_valid = (df['domain'] == 1) & (df['a'] == 0)
    y_true = df[ind_valid]['y']
    y_pred_eta0 = results[0][0][ind_valid]  
    y_pred_eta0_bench = results[0][1][ind_valid]  
    eta0_auc, eta0_f1, eta0_acc = evaluate(y_true, y_pred_eta0)
    eta0_bench_auc, eta0_bench_f1, eta0_bench_acc = evaluate(y_true, y_pred_eta0_bench)

    # evaluate eta1
    ind_valid = (df['domain'] == 1) & (df['a'] == 1)
    y_true = df[ind_valid]['y']
    y_pred_eta1 = results[1][0][ind_valid]  
    y_pred_eta1_bench = results[1][1][ind_valid]  
    eta1_auc, eta1_f1, eta1_acc = evaluate(y_true, y_pred_eta1)
    eta1_bench_auc, eta1_bench_f1, eta1_bench_acc = evaluate(y_true, y_pred_eta1_bench)

    # evaluate eta
    ind_valid = df['domain'] == 1
    y_true = df[ind_valid]['y']
    y_pred_eta = results[2][0][ind_valid]  
    y_pred_eta_bench = results[2][1][ind_valid]  
    eta_auc, eta_f1, eta_acc = evaluate(y_true, y_pred_eta)
    eta_bench_auc, eta_bench_f1, eta_bench_acc = evaluate(y_true, y_pred_eta_bench)

    # evaluate gemma
    ind_valid = df['domain'] == 1
    y_true = df[ind_valid]['y']
    y_pred_gemma = results[3][ind_valid]  
    gemma_auc, gemma_f1, gemma_acc = evaluate(y_true, y_pred_gemma)

    df_data.extend([
        [a, b, c, seed, 'eta0', 'AUC', eta0_auc],
        [a, b, c, seed, 'eta0', 'F', eta0_f1],
        [a, b, c, seed, 'eta0', 'ACC', eta0_acc],
        [a, b, c, seed, 'eta0_bench', 'AUC', eta0_bench_auc],
        [a, b, c, seed, 'eta0_bench', 'F', eta0_bench_f1],
        [a, b, c, seed, 'eta0_bench', 'ACC', eta0_bench_acc],
        [a, b, c, seed, 'eta1', 'AUC', eta1_auc],
        [a, b, c, seed, 'eta1', 'F', eta1_f1],
        [a, b, c, seed, 'eta1', 'ACC', eta1_acc],
        [a, b, c, seed, 'eta1_bench', 'AUC', eta1_bench_auc],
        [a, b, c, seed, 'eta1_bench', 'F', eta1_bench_f1],
        [a, b, c, seed, 'eta1_bench', 'ACC', eta1_bench_acc],
        [a, b, c, seed, 'eta', 'AUC', eta_auc],
        [a, b, c, seed, 'eta', 'F', eta_f1],
        [a, b, c, seed, 'eta', 'ACC', eta_acc],
        [a, b, c, seed, 'eta_bench', 'AUC', eta_bench_auc],
        [a, b, c, seed, 'eta_bench', 'F', eta_bench_f1],
        [a, b, c, seed, 'eta_bench', 'ACC', eta_bench_acc],
        [a, b, c, seed, 'gemma', 'AUC', gemma_auc],
        [a, b, c, seed, 'gemma', 'F', gemma_f1],
        [a, b, c, seed, 'gemma', 'ACC', gemma_acc],
        ])
    

columns = ['a', 'b', 'c', 'seed', 'model', 'metric', 'value']
df = pd.DataFrame(df_data, columns=columns)
df.to_csv('experiments.csv.2', index=None)
print(df)
