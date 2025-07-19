import numpy as np 
import pandas as pd 
from run import evaluate
import pickle

pkl_file = './experiment_switch_a_y.pkl'
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

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
        [a, b, c, seed, 'eta_bench', 'ACC', eta_bench_acc]
        ])
    

columns = ['a', 'b', 'c', 'seed', 'model', 'metric', 'value']
df = pd.DataFrame(df_data, columns=columns)
df.to_csv('experiments_swtich_a_y.csv', index=None)
print(df)
