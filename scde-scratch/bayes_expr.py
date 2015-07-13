"""
Python implementation of
'Bayesian approach to single-cell differential expression analysis'
 (Nature Methods 2014)

kieran.campbell@sjc.ox.ac.uk
"""

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm

import cPickle
import warnings

from pandas import DataFrame
from scipy.stats.distributions import poisson
from scipy.stats.distributions import nbinom
from scipy.special import gammaln
from scipy.optimize import minimize


def load_reduced_expression():
    expression_pickle = "/Users/kieranc/oxford/projects/data/reduced_expression.pkl"

    # Can alternatively use
    # expr = pd.read_csv(fname, index_col = 0)

    pkl_file = open(expression_pickle, 'rb')
    expr = cPickle.load(pkl_file)
    return expr.T # make sample-by-variable always
    
def load_full_twocell_expression():
    expr_file = "/Users/kieranc/oxford/projects/data/two_cells_full.csv"
    two_cell_expr = pd.read_csv(expr_file, index_col = 0)
    return two_cell_expr

def _sanitize_median_expr(x, thresh):
    X = x[x>thresh]
    """
    If no counts exist, give pseudocount of 1 ??
    """
    if X.size == 0:
        return 1
    else:
        return np.median(X)

def get_median_expr(expr, thresh=5):
    e = expr.apply(lambda x: _sanitize_median_expr(x, thresh))
    return e.astype(int)




"""
Convert a sample-by-gene pandas data frame
from fragments per kilobase transcript per million reads mapped
(FPKM) to transcripts per million (tpm)

Converted by TPM_i = FPKM_i / \sum_j FPKM_j * 10^6
"""
def fkpm_to_tpm(expr):
    # no confusion about division
    tpm_expr = expr.apply(lambda x: x / x.sum(), axis=1)
    tpm_expr = tpm_expr * 1000000
    return tpm_expr.astype(int)


def generate_pois_mix(lambda_0 = 0.1, lambda_1 = 3, pi = 0.2, N=1):
    print "Generating data with mixing %f and means %i %i" % (pi, lambda_0, lambda_1)
    result = np.zeros(N)

    for i in range(N):
        if np.random.random() < pi:
        # generate from lambda_0
            result[i] =  np.random.poisson(lambda_0)
        else:
            result[i] =  np.random.poisson(lambda_1)


    return result




"""
Calculates pi_k * pois(x, lambda)
for a given mixture model component.
Note - is vectorized
"""
def component_likelihood(x, lambda_i, pi_k):
    return pi_k * poisson.pmf(x, lambda_i)

def model_likelihood(x, lambd, pi):
    return component_likelihood(x, lambd[0], pi[0]) + \
        component_likelihood(x, lambd[1], pi[1])

"""
Fits a two-poisson mixture model with the mean of one of the distributions
held constant (set with lambda_0)
"""
def two_poisson_em_const(N=100, data = "none", lambda_0 = 1):

    if data == "none":
        lambda_0, lambda_1, pi = [1,5,0.6]
        data = generate_pois_mix(lambda_0, lambda_1, pi, N=N)

    # want to estimate pi and lambda_1 (lambda_0 fixed at 0.1)
    gamma = np.zeros((2, data.size))
    N_ = np.zeros(2)

    p0 = np.array([0.5, 10])
    pi_0, lambda_1 = p0
    pi = np.array([pi_0, 1 - pi_0])
    lambd = np.array([lambda_0, lambda_1])
    
    # em loop
    counter = 0
    converged = False
    max_iter = 100
    while not converged:
        # compute responsibility function 
        for k in [0,1]:
            gamma[k,:] = component_likelihood(data, lambd[k], pi[k]) / \
                         model_likelihood(data, lambd, pi)
            N_[k] = 1.*gamma[k].sum()

        # lambda_1 and pi_1 estimation
        lambd[1] = sum(gamma[1,:] * data) / N_[1]
        pi_0 = N_[0] / data.size


        pi = np.array([pi_0, 1 - pi_0])
        assert abs(N_.sum() - data.size) / float(data.size) < 1e-6
        assert abs(pi.sum() - 1) < 1e-6

        counter += 1
        converged = counter > max_iter
        
    model = [pi_0, lambd[1]]
    return model, gamma
    
def do_em_rep(n=100):
    res = np.zeros((n, 2))
    for i in range(n):
        res[i,:] = two_poisson_em()

    return res.mean(0)

"""
Fits a two-poisson mixture model
"""
def two_poisson_em(N=100, data = "none"):

    if data == "none":
        lambda_0, lambda_1, pi = [1,5,0.6]
        data = generate_pois_mix(lambda_0, lambda_1, pi, N=N)

    # want to estimate pi and lambda_1 (lambda_0 fixed at 0.1)
    gamma = np.zeros((2, data.size))
    N_ = np.zeros(2)

    p0 = np.array([0.5, 5, 10])
    pi_0, lambda_0, lambda_1 = p0
    pi = np.array([pi_0, 1 - pi_0])
    lambd = np.array([lambda_0, lambda_1])
    
    # em loop
    counter = 0
    converged = False
    max_iter = 100
    while not converged:
        # compute responsibility function 
        for k in [0,1]:
            gamma[k,:] = component_likelihood(data, lambd[k], pi[k]) / \
                         model_likelihood(data, lambd, pi)
            N_[k] = 1.*gamma[k].sum()


        
        # lambda and pi estimation
        for k in [0,1]:
            lambd[k] = sum(gamma[k,:] * data) / N_[k]
            if lambd[k] == 0:
                lambd[k] = 0.1 # force lambda never to reach 0 - nonsensical PP
            
        pi_0 = N_[0] / data.size
        pi = np.array([pi_0, 1 - pi_0])

        print "lambda: " + str(lambd)
        print "pi: " + str(pi)
        assert abs(N_.sum() - data.size) / float(data.size) < 1e-6
        assert abs(pi.sum() - 1) < 1e-6

        counter += 1
        converged = counter > max_iter
        
    model = [pi_0, lambd[0], lambd[1]]

    return model, gamma

        

"""
Objective function that needs minimised to find the
MLE estimate of a negative binomial in a mixture model
Returns minus twice the log likelihood (see Murphy eq. 11.29)

Note doesn't return log-likelihood (constant terms removed), so if
it's much greater than 1, don't panic
"""
def NB_mixture_objective_function(r, x, gam, m):
    value = 0

    # convert to floats otherwise math.log(m / (m+r)) -> -inf
    r = float(r)
    m = float(m)

    for i in range(len(x)):
        x_i = x[i]
        g_i = gam[i]
        expr = gammaln(r + x_i)
        expr -= gammaln(r)
        expr -= r * math.log(1 + m / r)
        expr += x_i * math.log(m / (m + r))
        #expr -= math.log(math.factorial(x_i)) (not needed for opt over r)
        value += g_i * expr

    return -2 * value

def pois_likelihood(x, lambd):
    return poisson.pmf(x, lambd)

def nb_likelihood(x, m, r):
    p = float(r) / (r + m)
    return nbinom.pmf(x, r, p)

def pois_nb_likelihood(x, pi, lambd, m, r):
    expr = pi * pois_likelihood(x, lambd)
    expr += (1 - pi) * nb_likelihood(x, m, r)
    return expr
    
def fixed_pois_NB_mixture(N=100, data = "none", lambda_0 = 0.1):

    if data == "none":
        data = generate_pois_NB_mix(N=N)

    # want to estimate pi and lambda_1 (lambda_0 fixed at 0.1)
    gamma = np.zeros((2, data.size))
    N_ = np.zeros(2)

    p0 = np.array([0.5, 10, 50])
    pi_0, r, m = p0
    pi = np.array([pi_0, 1 - pi_0])
    
    # em loop
    counter = 0
    converged = False
    max_iter = 100
    while not converged:
        # compute responsibility function 
        total_likelhood = pois_nb_likelihood(data, pi[0], lambda_0, m, r)

        gamma[0,:] = pi[0] * pois_likelihood(data, lambda_0) / \
                     total_likelhood
        gamma[1,:] = pi[1] * nb_likelihood(data, m, r) / \
                     total_likelhood

        for k in [0,1]:
            N_[k] = 1.*gamma[k,:].sum()

        # m estimation (still valid - see notes)
        m = sum(gamma[1,:] * data) / N_[1]

        """
        r estimation: minimise NB_mixture_objective_function
        (see proof in notes). We use the previous r as the next
        guess (starting point). 400 used as arbitrary upper bound.
        Uses truncated newton algorithm (TNC), identical to newton's
        method but with bounds.
        """
        upper_bound = 400
        minimization = minimize(NB_mixture_objective_function,
                               x0 = r,
                               args = (data, gamma[1,:], m),
                               method = 'TNC',
                               bounds = [(0.001, upper_bound)])
        r = minimization['x'][0]
        if r == upper_bound:
            warnings.warn("Estimate has reached upper bound implying boundary optimization. Consider chaning boundaries")
        
        pi_0 = N_[0] / data.size
        pi = np.array([pi_0, 1 - pi_0])

        assert abs(N_.sum() - data.size) / float(data.size) < 1e-6
        assert abs(pi.sum() - 1) < 1e-6

        counter += 1
        converged = counter > max_iter
        
    model = {'pi': pi_0, 'm': m, 'r': r}
    return model, gamma


"""
Generate a poisson-negative binomial mixture model, where the
poisson is generated by lambda and NB by (m,r) = mean & shape,
with mixing parameter pi and number of samples N.
Overall pmf:
p(x | l, m, r, pi) + pi * Pois(l) + (1-pi)*NB(m,r)
"""
def generate_pois_NB_mix(lambd = 0.1, m = 10, r = 3, pi = 0.2, N=1):
    print "Generating poisson-NB mix with lambda: %f, m: %f, r: %f, pi: %f" % (lambd, m, r, pi)
    result = np.zeros(N)

    # numpy generates random NB in terms of probability, so calculate that:
    p = float(r) / (r + m)

    for i in range(N):
        if np.random.random() < pi: 
        # generate poisson 
            result[i] =  np.random.poisson(lambd)
        else:
            result[i] =  np.random.negative_binomial(r, p)


    return result


"""
Full model specification:
Poisson-NB *regression* mixture using concomitant variables (mixture of experts)
20-11-2014
"""

"""
Can use pois_likelihood from before
"""

"""
Negative binomial regression likelihood
"""
def nb_regression_likelihood(y, x, (a_0, a_1), r):
    m = a_0 + a_1 * x
    p = float(r) / (r + m)

    return nbinom.pmf(y, r, p)

def mix_from_params(x, (w0, w1)): # just a sigmoid function
    pi = np.zeros((x.size,2))
    for i in range(x.size):
        pi[i,0] = 1.0/(1 + np.exp(w0 + w1*x[i]))
    pi[:,1] = 1 - pi[:,0]
    return pi

def NB_regression_objective_function((a0, a1, r), y, x, gam):
    value = 0

    # convert to floats otherwise math.log(m / (m+r)) -> -inf
    r = float(r)
    m = a0 + a1 * x

    #print (r)

    for i in range(len(x)):
        y_i = y[i]
        m_i = m[i]
        if m_i / (m_i + r) <= 0:
            print (m_i, r)
        g_i = gam[i]
        expr = gammaln(r + y_i)
        expr -= gammaln(r)
        expr -= r * math.log(1 + m_i / r)
        expr += y_i * math.log(m_i / (m_i + r))
        #expr -= math.log(math.factorial(x_i)) (not needed for opt over r)
        value += g_i * expr

    return -2 * value

def NB_regression_objective_function2((a0, a1, r), y, x, gam):
    """
    Same as NB_regression_objective_function, but around
    three orders of magnitude faster
    USE THIS
    """
    m = a0 + a1 * x
    glr = gammaln(r)

    expr = gammaln(r + y)
    expr -= glr 
    expr -= r * np.log(1 + m / r)
    expr += y * np.log(m / (m + r))
    
    expr = expr * gam

    return -2 * expr.sum()

"""
Main Poisson-NB mixture model with
regression and concomitant (mixture of experts)
variables

y: response vector in question
e: median non-zero expression for gene i
N: repetitions
lambda_0: poisson noise mean
"""
def fixed_pois_NB_concomitant_regression((y,e), N=100, lambda_0 = 0.1):

    if y.size != e.size:
        print "y & e must have same size for regression"
        return 

    gamma = np.zeros((y.size,2)) #r_ki variables
    N_ = np.zeros(2) # number beloning to each class

    """
    Parameter set:
    (a0,a1) : regression coefficients for NB
    r : dispersion parameter for nb
    (w0, w1) : logistic regression coefficients for mixing

    Initial guesses:
    """
    (a0, a1) = (0, 1)
    r = 10
    (w0, w1) = (0,0) # give each class equal prior

    counter = 0
    converged = False
    max_iter = 100
    while not converged:

        # compute responsibility functions
        pi = mix_from_params(y, (w0, w1))
        pois_lik = pois_likelihood(y, lambda_0)
        nb_lik = nb_regression_likelihood(y, e, (a0, a1), r)


        gamma[:,0] = pois_lik * pi[:,0]
        gamma[:,1] = nb_lik * pi[:,1]

        loglik = gamma.sum()
        print loglik

        # normalize the gammas
        gamma = gamma/gamma.sum(1)[:,np.newaxis]

        if not np.allclose(gamma.sum(1), np.ones(gamma.shape[0])):
            print "Rowsums of gamma are not 1"
            return

        for k in [0,1]:
            N_[k] = 1.*gamma[:,k].sum()

        """
        Now need to estimate parameters
        """
        # (w0, w1) optimisation
        rik = gamma[:,1]
        X = sm.tools.add_constant(e)
        X.columns = ['w0','w1']
        logit = sm.Logit(rik, X)
        result = logit.fit()
        w0 = result.params['w0']
        w1 = result.params['w1']

        # (a0, a1, r) estimation
        minimization = minimize(NB_regression_objective_function2,
                                x0 = np.array([a0, a1, r]),
                                bounds = [(0, None),
                                            (0, None),
                                            (0.1, None)],
                                args = (y, e, gamma[:,1]))
        (a0, a1, r) = minimization['x']

        if a0 == 0 or a1 == 0:
            print "Warning: optimised a lies on boundary"   

        counter += 1
        converged = counter > max_iter

    model = {'a' : (a0, a1), 'w': (w0, w1), 'r': r}
    return model








expr = load_reduced_expression()

expr_tpm = fkpm_to_tpm(expr)
e = get_median_expr(expr)
y = expr.ix[0,:].astype(int)

# expr_full = load_full_twocell_expression().T
# expr_full = fkpm_to_tpm(expr_full)
# e_full = get_median_expr(expr_full)

# y_full = expr_full.ix[0,:]








