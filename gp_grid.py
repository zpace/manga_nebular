import pymc3 as pm
import numpy as np
import theano

def solve_single_gp(lengthscales, X0, samples, noise=1.0e-3):
    cov = pm.gp.cov.ExpQuad(len(lengthscales), ls=lengthscales)
    K = cov(X0.T)
    K_noise = K + pm.gp.cov.WhiteNoise(noise)(X0.T)

    L = np.linalg.cholesky(K_noise.eval())
    alpha = np.linalg.solve(
        L.T, np.linalg.solve(L, samples.flatten()))

    return cov, alpha

def gp_predict(cov, alpha, X0, X1):
    K_s = cov(X0.T, X1.T)
    post_mean = np.dot(K_s.T.eval(), alpha)
    return post_mean

def gp_predictt(cov, alpha, X0, X1):
    K_s = cov(X0.T, X1.T)
    post_mean = theano.tensor.dot(K_s.T, alpha)
    return post_mean

def gp_learn_emline_space(sparsegrids, grid_preds, ls_mult=1.5, noise=1.0e-4):
    '''
    use a GP model to learn the emission-line prediction space for one observable
    '''
    XX0 = np.meshgrid(*[g for g in sparsegrids], indexing='ij')
    X0 = np.row_stack([a.flatten() for a in XX0])
    ls = ls_mult * np.array(list(map(
        lambda a: np.mean(np.diff(a)), sparsegrids)))

    covs, alphas = zip(*[solve_single_gp(ls, X0, pred.flatten(), noise=noise)
                         for pred in grid_preds])

    return X0, covs, alphas
