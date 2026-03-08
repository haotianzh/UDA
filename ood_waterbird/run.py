import os
import pickle
import numpy as np
import pandas as pd 
import multiprocessing as mp
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from ood import ood, ood_predict, predict_probability
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

def generate_metadata_df(metadata_name, a, b, c, seed):
    np.random.seed(seed)
    df = pd.read_csv(metadata_name)
    df['cls'] = df['y'].astype(str) + df['a'].astype(str)
    # check initial distribution on training and test
    def assign_R(df, a, b, c):
        df = df.copy()
        df['R'] = 0
        source_ind_01 = np.random.choice(df[df['cls'] == '01'].index, int(df[df['cls'] == '01'].shape[0]*a), replace=False)
        source_ind_10 = np.random.choice(df[df['cls'] == '10'].index, int(df[df['cls'] == '10'].shape[0]*b), replace=False)
        source_ind_00 = np.random.choice(df[df['cls'] == '00'].index, int(df[df['cls'] == '00'].shape[0]*c), replace=False)
        df.loc[source_ind_01, 'R'] = 1
        df.loc[source_ind_10, 'R'] = 1
        df.loc[source_ind_00, 'R'] = 1
        return df
    df = assign_R(df, a, b, c)
    return df

def load_data(embeddings, metadata_df):
    df_source = metadata_df[metadata_df['R'] == 1]
    df_source.loc[:, 'split'] = 0
    df_target = metadata_df[metadata_df['R'] == 0]
    df_target.loc[:, 'split'] = 1
    # reorganize embeddings 
    new_ind = np.concatenate([df_source.index, df_target.index])
    embeddings = embeddings[new_ind]
    y = np.concatenate([df_source['y'].values, df_target['y'].values])
    a = np.concatenate([df_source['a'].values, df_target['a'].values])
    R = np.concatenate([df_source['R'].values, df_target['R'].values])
    domain = np.concatenate([df_source['split'].values, df_target['split'].values])
    df = pd.DataFrame({'y':y, 'a':a, 'R': R, 'domain': domain})
    df['cls'] = df['y'].astype(str) + df['a'].astype(str)
    return embeddings, df

def train_logistic_model_1(embeddings, df):
    # logistic for model 1
    ind_model_1 = (df['domain'] == 0) & (df['cls'].isin(['00', '10'])) 
    x = embeddings[ind_model_1]
    y = df['y'][ind_model_1]
    # model_1 = LogisticRegression(penalty='l1', solver='saga', max_iter=500).fit(x, y)
    model_1 = LogisticRegression(penalty='l2', max_iter=1000).fit(x, y)
    model_1_outputs_logistic = model_1.predict_proba(embeddings)[:, 1]
    return model_1_outputs_logistic

def train_logistic_model_2(embeddings, df):
    ind_model_2_1 = (df['domain'] == 1) & (df['cls'].isin(['01', '11'])) 
    ind_model_2_2 = (df['domain'] == 0) & (df['cls'].isin(['01']))
    ind_model_2 = ind_model_2_1 | ind_model_2_2 
    x = embeddings[ind_model_2]
    y = df['R'][ind_model_2]
    # model_2 = LogisticRegression(penalty='l1', solver='saga', max_iter=500).fit(x, y)
    model_2 = LogisticRegression(penalty='l2', max_iter=1000).fit(x, y)
    model_2_outputs_logistic = model_2.predict_proba(embeddings)[:, 1]
    return model_2_outputs_logistic

def train_logistic_model_3(embeddings, df):
    ind_model_3 = (df['domain'] == 1) 
    x = embeddings[ind_model_3]
    y = df[ind_model_3]['a']
    # model_3 = LogisticRegression(penalty='l1', solver='saga', max_iter=500).fit(x, y)
    model_3 = LogisticRegression(penalty='l2', max_iter=1000).fit(x, y)
    model_3_outputs_logistic = model_3.predict_proba(embeddings)[:, 1]  
    return model_3_outputs_logistic

def train_logistic_model_4(embeddings, df):
    ind_model_4 = (df['domain'] == 0) & (df['cls'].isin(['00', '01', '10'])) 
    x = embeddings[ind_model_4]
    y = df['y'][ind_model_4]
    # model_4 = LogisticRegression(penalty='l1', solver='saga', max_iter=500).fit(x, y)
    model_4 = LogisticRegression(penalty='l2', max_iter=1000).fit(x, y)
    model_4_outputs_logistic = model_4.predict_proba(embeddings)[:, 1]
    return model_4_outputs_logistic

def train_logistic_model_5(embeddings, df):
    ind_model_5 = (df['domain'] == 0) 
    x = embeddings[ind_model_5]
    y = df['a'][ind_model_5]
    # model_5 = LogisticRegression(penalty='l1', solver='saga', max_iter=500).fit(x, y)
    model_5 = LogisticRegression(penalty='l2', max_iter=1000).fit(x, y)
    model_5_outputs_logistic = model_5.predict_proba(embeddings)[:, 1]  
    return model_5_outputs_logistic


def get_gemma(logits, alphas, betas):
    beta00, beta01, beta10, beta11 = betas
    alpha00, alpha01, alpha10 = alphas
    res0 = logits * ((beta10 + beta11) / (alpha10))
    res1 = (1 - logits) * ((beta00 + beta01) / (alpha00 + alpha01))
    return res0 / (res0 + res1)
 
def calculate(embeddings, df, model_outputs):
    target = {'x': embeddings[df['domain'] == 1], 'A': df[df['domain'] == 1]['a']}
    source_ind = (df['domain'] == 0) & (df['cls'].isin(['00', '01', '10'])) 
    source = {'x': embeddings[source_ind], 'y': df[source_ind]['y'].values.reshape(-1,1), 'A': df[source_ind]['a']}
    beta0010dm = ood(source=source, target=target, method='distribution')
    beta00dm, beta10dm = beta0010dm
    beta11dm = sum((df['domain'] == 1) & (df['cls']=='11')) / sum(df['domain'] == 1)
    beta01dm = 1 - beta00dm - beta10dm - beta11dm
    betadmvec = [beta00dm, beta01dm, beta10dm, beta11dm]
    # print(betadmvec)
    # print(df[df['domain'] == 1].groupby('cls')['a'].count() / df[df['domain'] == 1].groupby('cls')['a'].count().sum())
    # estimate eta0
    nsource = sum(source_ind)
    alpha10 = (sum((df['domain'] == 0) & (df['cls']=='10')) / nsource)
    alpha00 = (sum((df['domain'] == 0) & (df['cls']=='00')) / nsource)
    alpha01 = (sum((df['domain'] == 0) & (df['cls']=='01')) / nsource)
    ba10 = betadmvec[2] / alpha10
    ba00 = betadmvec[0] / alpha00
    ba01 = betadmvec[1] / alpha01
    eta0 = ba10 * model_outputs[0] / (ba10 * model_outputs[0] + ba00 * (1 - model_outputs[0]))
    # estimate eta1
    pihat = sum(df['domain'] == 0) / (sum(df['domain'] == 0) + sum(df['domain'] == 1))
    eta1 = 1 - ba01 * (1 - pihat) / pihat *  model_outputs[1] / (1 -  model_outputs[1])
    eta = eta1 * model_outputs[2] + eta0 * (1 - model_outputs[2])
    # benchmark eta0
    eta0_bench = model_outputs[0]
    # benchmark eta
    eta_bench = model_outputs[3]
    # benchmark eta1
    tau1hat = model_outputs[4]
    # new benchmark gemma
    gemma = get_gemma(model_outputs[3], [alpha00, alpha01, alpha10], betadmvec)
    eta1_bench = (eta_bench - eta0_bench * (1 - tau1hat)) / tau1hat
    return [(eta0, eta0_bench), 
            (eta1, eta1_bench), 
            (eta, eta_bench),
            gemma]


def run(a, b, c, seed):
    # metadata_name = './embeds/metadata_vit16.csv'
    metadata_name = './embeds/metadata_resnet50.csv'
    # embeddings = np.load('./embeds/embeds_resnet18.npy')
    # embeddings = np.load('./embeds/embeds_vit16.npy')
    embeddings = np.load('./embeds/embeds_resnet50.npy')
    metadata_df = generate_metadata_df(metadata_name=metadata_name, a=a, b=b, c=c, seed=seed)
    # print(metadata_df.groupby(['R', 'y', 'a']).count())
    x, df = load_data(embeddings, metadata_df)
    models = [train_logistic_model_1, train_logistic_model_2, train_logistic_model_3, train_logistic_model_4, train_logistic_model_5]
    model_outputs = []
    for model in models:
        model_outputs.append(model(x, df))
    results = calculate(x, df, model_outputs)
    print(results)
    return a, b, c, seed, results, df


if __name__ == "__main__":

    mp.set_start_method('spawn')
    seeds = np.random.choice(np.arange(10000), 5, replace=False)
    res = []
    ranges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    with mp.Pool(24) as pool:
        for a in [0.3, 0.5, 0.7]:
            for b in ranges:
                for seed in seeds:
                    d = pool.apply_async(run, (a, b, 0.5, seed))
                    res.append(d)
            for c in ranges:
                for seed in seeds:
                    d = pool.apply_async(run, (a, 0.5, c, seed))
                    res.append(d)
        pool.close()
        pool.join()


    data = {}
    for d in res:
        a, b, c, seed, results, df = d.get()
        data[f'{a}_{b}_{c}_{seed}'] = {}
        data[f'{a}_{b}_{c}_{seed}']['outputs'] = results
        data[f'{a}_{b}_{c}_{seed}']['df'] = df

    with open('experiment_resnet50.pkl', 'wb') as out:
        pickle.dump(data, out, protocol=4)
    