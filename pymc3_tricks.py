from pymc3.step_methods.hmc.quadpotential import QuadPotentialFull
import pymc3 as pm
import numpy as np

import time

def get_step_for_trace(trace=None, model=None,
                       regular_window=5, regular_variance=1e-3,
                       **kwargs):
    model = pm.modelcontext(model)

    # If not given, use the trivial metric
    if trace is None:
        potential = QuadPotentialFull(np.eye(model.ndim))
        return pm.NUTS(potential=potential, **kwargs)

    # Loop over samples and convert to the relevant parameter space;
    # I'm sure that there's an easier way to do this, but I don't know
    # how to make something work in general...
    samples = np.empty((len(trace) * trace.nchains, model.ndim))
    i = 0
    for chain in trace._straces.values():
        for p in chain:
            samples[i] = model.bijection.map(p)
            i += 1

    # Compute the sample covariance
    cov = np.cov(samples, rowvar=0)

    # Stan uses a regularized estimator for the covariance matrix to
    # be less sensitive to numerical issues for large parameter spaces.
    # In the test case for this blog post, this isn't necessary and it
    # actually makes the performance worse so I'll disable it, but I
    # wanted to include the implementation here for completeness
    N = len(samples)
    cov = cov * N / (N + regular_window)
    cov[np.diag_indices_from(cov)] += \
        regular_variance * regular_window / (N + regular_window)

    # Use the sample covariance as the inverse metric
    potential = QuadPotentialFull(cov)
    return pm.NUTS(potential=potential, **kwargs)

def densemass_sample(model, nstart=200, nburn=200, ntune=5000, **kwargs):
    '''
    DFM's mass-matrix-learning routine

    https://dfm.io/posts/pymc3-mass-matrix/
    '''
    nwindow = nstart * 2 ** np.arange(np.floor(np.log2((ntune - nburn) / nstart)))
    nwindow = np.append(nwindow, ntune - nburn - np.sum(nwindow))
    nwindow = nwindow.astype('int')
    print('Sequentially scaling mass matrix...')
    print('Window sizes:', nwindow)

    with model:
        start = None
        burnin_trace = None
        for steps in nwindow:
            branch_start_time = time.time()
            step = get_step_for_trace(burnin_trace, regular_window=0)
            burnin_trace = pm.sample(
                start=start, tune=steps, draws=steps, step=step,
                compute_convergence_checks=False, discard_tuned_samples=False,
                **kwargs)
            branch_duration = time.time() - branch_start_time

            start = burnin_trace.point(-1)

        step = get_step_for_trace(burnin_trace, regular_window=0)

        return step, start
