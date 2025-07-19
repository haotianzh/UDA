import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar


def predict_probability(model, new_data_matrix, col_names):
    new_data_df = pd.DataFrame(new_data_matrix, columns=col_names)
    predicted_probs = model.predict_proba(new_data_df)[:, 1]
    return predicted_probs

def ood(target, source, method='distribution'):
    xtg = target['x']
    xsc = source['x']
    ysc = source['y']
    Atg = target['A']
    Asc = source['A']
    phat = np.sum(Atg == 0) / (xtg.shape[0] + xsc.shape[0])
    pihat = xsc.shape[0] / (xsc.shape[0] + xtg.shape[0])

    if method == 'distribution':
        b1hat = 1 / np.mean(ysc[Asc == 0])
        b0hat = 1 / (1 - np.mean(ysc[Asc == 0]))
        yreg = ysc[Asc == 0]
        xreg = xsc[Asc == 0, :]
        datareg = pd.DataFrame(xreg)
        datareg.insert(0, 'y', yreg)
        modelreg = LogisticRegression(solver='lbfgs', max_iter=1000)
        modelreg.fit(datareg.drop(columns='y'), datareg['y'])
        col_names = list(datareg.drop(columns='y').columns)
        eta1 = predict_probability(modelreg, xtg[Atg == 0, :], col_names)
        eta0 = 1 - eta1
        def f(b00):
            return -np.mean(np.log(eta0 * b0hat * b00 + eta1 * b1hat * (phat / (1 - pihat) - b00)))
        result = minimize_scalar(f, bounds=(0, phat / (1 - pihat)), method='bounded')
        return [result.x, phat / (1 - pihat) - result.x]
    
    elif method.startswith('moment'):
        if method == 'moment1':
            xmean1 = np.mean(xsc[(Asc == 0) & (ysc == 1), 0])
            xmean0 = np.mean(xsc[(Asc == 0) & (ysc == 0), 0])
            Hmat = np.array([[1, 1], [xmean1, xmean0]])
            Xmean3 = np.mean(xtg[Atg == 0, 0])
            vmat = np.array([1, Xmean3])
        elif method == 'moment2':
            x1mean1 = np.mean(xsc[(Asc == 0) & (ysc == 1), 0]**2)
            x1mean0 = np.mean(xsc[(Asc == 0) & (ysc == 0), 0]**2)
            x3mean1 = np.mean(xsc[(Asc == 0) & (ysc == 1), 2])
            x3mean0 = np.mean(xsc[(Asc == 0) & (ysc == 0), 2])
            Hmat = np.array([[x1mean1, x1mean0], [x3mean1, x3mean0]])
            x1mean2 = np.mean(xtg[Atg == 0, 0]**2)
            x3mean2 = np.mean(xtg[Atg == 0, 2])
            vmat = np.array([x1mean2, x3mean2])
        elif method == 'moment3':
            x1mean1 = np.mean(xsc[(Asc == 0) & (ysc == 1), 0]**4)
            x1mean0 = np.mean(xsc[(Asc == 0) & (ysc == 0), 0]**4)
            x3mean1 = np.mean(xsc[(Asc == 0) & (ysc == 1), 2]**2)
            x3mean0 = np.mean(xsc[(Asc == 0) & (ysc == 0), 2]**2)
            Hmat = np.array([[x1mean1, x1mean0], [x3mean1, x3mean0]])
            x1mean2 = np.mean(xtg[Atg == 0, 0]**4)
            x3mean2 = np.mean(xtg[Atg == 0, 2]**2)
            vmat = np.array([x1mean2, x3mean2])
        else:
            raise ValueError("Invalid method provided")
        Hmatinv = np.linalg.inv(Hmat)
        betahat = (1 / (1 - pihat)) * phat * Hmatinv.dot(vmat)
        return list(reversed(betahat.tolist()))
    else:
        raise ValueError("Please provide a correct method!")

def ood_predict(source, target, betavec):
    beta00, beta01, beta10, beta11 = betavec
    xtg = target['x']
    xsc = source['x']
    ysc = source['y']
    Atg = target['A']
    Asc = source['A']
    alpha00 = np.sum((ysc == 0) & (Asc == 0)) / xsc.shape[0]
    alpha01 = np.sum((ysc == 0) & (Asc == 1)) / xsc.shape[0]
    alpha10 = np.sum((ysc == 1) & (Asc == 0)) / xsc.shape[0]
    pihat = xsc.shape[0] / (xsc.shape[0] + xtg.shape[0])
    xksi = xsc[Asc == 0, :]
    yksi = ysc[Asc == 0]
    datareg = pd.DataFrame(xksi)
    datareg.insert(0, 'y', yksi)
    modelreg = LogisticRegression(solver='lbfgs', max_iter=1000)
    modelreg.fit(datareg.drop(columns='y'), datareg['y'])
    col_names = list(datareg.drop(columns='y').columns)
    ksi0 = predict_probability(modelreg, xtg, col_names)
    xscr = np.hstack((np.ones((np.sum(Asc == 1), 1)), xsc[Asc == 1, :]))
    xtgr = np.hstack((np.zeros((np.sum(Atg == 1), 1)), xtg[Atg == 1, :]))
    datareg = pd.DataFrame(np.vstack((xscr, xtgr)))
    datareg.columns = ['y'] + [f'x{i}' for i in range(1, datareg.shape[1])]
    modelreg = LogisticRegression(solver='lbfgs', max_iter=1000)
    modelreg.fit(datareg.drop(columns='y'), datareg['y'])
    col_names = list(datareg.drop(columns='y').columns)
    ksi1 = predict_probability(modelreg, xtg, col_names)
    eta0 = (beta10 / alpha10) * ksi0 / ((beta10 / alpha10) * ksi0 + (beta00 / alpha00) * (1 - ksi0))
    eta1 = 1 - (beta01 / alpha01) * ((1 - pihat) / pihat) * ksi1 / (1 - ksi1)
    return {'eta0': eta0, 'eta1': eta1}
