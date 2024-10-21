import numpy as np
from itertools import combinations, count
from numpy.linalg import det, inv, norm
from dsci_project_assignment import Gradient_Descent
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import multivariate_normal as multi_n
from scipy.stats import uniform as uni
import pandas as pd

rng = np.random.default_rng(seed=0)
pd.set_option('display.max_columns', 20)
# Fetching Data:
data  = fetch_california_housing(as_frame=True)['frame'].copy()
data = np.c_[np.ones((data.shape[0],1)),data.to_numpy()]

# data.AveBedrms = np.ceil(data.AveBedrms.mul(data.Population))
# data.AveOccup = np.ceil(data.AveOccup.mul(data.Population))
# data['Bedrms_per_Hhld'] = data.AveBedrms.div(data.AveOccup)
# data.AveRooms = np.ceil(data.AveRooms.mul(data.Population))
# data.drop(columns =['Population'], inplace=True)
# data.iloc[:, [7,8]] = data.iloc[:, [8,7]]
# data.rename(columns={'MedHouseVal':'Bedrms_per_Hhld', 'Bedrms_per_Hhld':'MedHouseVal'}, inplace=True)


# Train/Val/Test Split:
val = train_test_split(data[:,:-1],data[:,-1], train_size=0.8, random_state=0)


# Kernel:
def kernel(data: pd.DataFrame):
    rows,columns= data.shape
    K = np.zeros((columns,columns))
    for i in range(columns):
        for j in range(columns):
            K[i,j] = np.exp((-1/2)*norm(data[i]-data[j])**2)
    return K
# print(kernel(val[0]))
def weight_prior(weights: np.ndarray, dims: int, mean: np.ndarray[float], kernel: np.ndarray[float]):
    return multi_n.logpdf(weights, mean=mean, cov=kernel)
# Multivariate Posterior 
def observe_variance(y_pred: np.ndarray, y_true: np.ndarray, dim: int):
    n = len(y_pred)
    residuals = np.sum(np.square(y_true-y_pred))/(n-dim)
    return residuals
def maximum_likelihood(data: np.ndarray, y_true: np.ndarray, w_mean: np.ndarray, w_var: np.ndarray):
    rows, columns = data.shape
    weights = multi_n.rvs(mean=w_mean, cov=w_var, size=1)
    y_pred = data.dot(weights)
    obs_var = np.square(y_pred-y_true).mean()
    lhood = np.sum(np.log((1/np.sqrt(2*np.pi*obs_var))*np.exp(-np.square(y_true-y_pred)/(2*obs_var))))
    obs_var = np.square(data.dot(weights)-y_true).mean()
    return weights, obs_var
def multi_likelihood(y_pred: np.ndarray, y_true: list, obs_var: float):
    '''
    Returns p(y|X,w)
    '''
    n = len(y_pred)
    post  = np.sum(np.log((1/np.sqrt(2*np.pi*obs_var))*np.exp(-np.square(y_true-y_pred)/(2*obs_var))))
    return post
def marginal_likelihood(data: np.ndarray, y_true: np.ndarray, obs_var: float, dims: int, w_samp: int, w_mean: np.ndarray, w_var: np.ndarray):
    '''
    Novice attempt at approximating the marginal likelihood.
    Returns p(y|X): summing over the weights
    **This approach requires mcmc for better accuracy; thus, don't use this callable in practice.**
    '''
    n = len(data)
    weight_samp = multi_n.rvs(mean=w_mean, cov=w_var, size=w_samp)
    marg = -np.inf
    for w in weight_samp:
        # print(w)
        marg = np.logaddexp(marg, multi_likelihood(data.dot(w), y_true, obs_var) + weight_prior(w, dims, w_mean, w_var))
    return marg - np.log(w_samp)

def alt_weight_post(data: np.ndarray, y: np.ndarray, w: np.ndarray, obs_var: float, w_var: float):
    sigma_n = np.linalg.inv(data.T.dot(data)/obs_var + np.linalg.inv(w_var))
    wn = np.dot(sigma_n, np.dot(data.T, y)/obs_var +np.dot(np.linalg.inv(w_var), w))
    return wn, sigma_n


def mcmc(y_true, y_pred, obs_var, dims):
    weight_i1 = multi_n.rvs(mean=np.zeros((dims, )), cov=np.eye(dims))
    weight = []
    for i in count():
        weight_i2 = uni.rvs(loc = np.zeros((dims,)), scale=1)
        prior_i1 = multi_n.pdf()
        lhood_i = multi_likelihood(y_pred, y_true,obs_var)
    pass 


transformer = QuantileTransformer(n_quantiles=5000, output_distribution='normal', random_state=0)

# Scaled Training Data:
transformer.fit(val[0])
XS = transformer.transform(val[0])

# # normali = lambda x: (x-x.min())/(x.max()-x.min())
# # XS = val[0].copy()
# # Performing Gradient Descent:
# gd = Gradient_Descent(XS, val[2], 1e-10)
# newtheta,*_ = gd.fit(1e-3)

# w, obs_var = maximum_likelihood(XS, val[2], np.zeros((9,)), np.eye(9))
# wn, sigma_n = alt_weight_post(XS, val[2], w, obs_var, np.eye(9))

# w_var = np.eye(9)
# w_mean = np.zeros((9,))
# weight_set = list()
# train_var = list()
# for i in range(1000):
#     w, obs_var = maximum_likelihood(XS, val[2],w_mean,w_var)
#     train_var.append(obs_var)
#     wn, sigma_n = alt_weight_post(XS, val[2], w, obs_var, w_var)
#     w_var = sigma_n
#     w_mean = wn
#     weight_set.append(w)  

# print(f'Initial Weights:{weight_set[0]} with model variance: {np.square(XS.dot(weight_set[0])-val[2]).mean()}')
# best_var = min(train_var)
# best_param = weight_set[train_var.index(best_var)]
# print(f' Last Weight:{best_param} with model variance:{np.square(XS.dot(best_param)-val[2]).mean()}')
# prior_w = multi_n.logpdf(newtheta, mean=np.zeros((9,)), cov=np.eye(9))
# XBIAS = np.c_[np.ones((XS.shape[0],1)), XS]
# print(gd.loglikelihood(8))
# sigma2 = observe_variance(XBIAS.dot(newtheta), val[2].to_numpy(), 9)
# w_mean, w_vari = alt_weight_post(XBIAS, val[2].to_numpy(), newtheta, sigma2, np.eye(9))
# print(multi_n.pdf(XBIAS,mean=w_mean, cov=w_vari))
# print(f'Observed Variance:{sigma2}')
# print(f'Likelihood:{multi_likelihood(XBIAS.dot(newtheta), val[2].to_numpy(), sigma2)}')
# print(f'Marginal Likelihood:{marginal_likelihood(XBIAS, val[2].to_numpy(), sigma2, 9,w_samp=5000, w_mean=np.zeros((9,)),w_var=np.eye(9))}')
# print(f'Exponential Probability of Parameters:{(multi_likelihood(XBIAS.dot(newtheta), val[2].to_numpy(), sigma2)+prior_w)-(marginal_likelihood(XBIAS, val[2].to_numpy(), sigma2, 9,w_samp=5000, w_mean=np.zeros((9,)),w_var=np.eye(9)))}')
# print(kernel(XBIAS.dot(newtheta)))
# post_y = multi_likelihood(XBIAS.dot(newtheta), val[2].to_numpy(), sigma2, 8)


