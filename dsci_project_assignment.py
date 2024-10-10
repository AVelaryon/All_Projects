# -*- coding: utf-8 -*-
"""DSCI_Project_Assignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qN81v8Wwl8R55YexdRFHHNkO5L7rtrm4
"""
import random as rand
import numpy as np
import pandas as pd
import itertools as it
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.preprocessing import QuantileTransformer, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scipy.stats.qmc as ssq
import scipy as sy
from scipy.stats import spearmanr
import seaborn as sns

"""## Data Exploration and Visualization"""

fetch_data = fetch_california_housing(as_frame=True)
data = fetch_data['frame'].copy()

data.describe()

min_max = lambda x: (x-x.min())/(x.max()-x.min())
# sns.displot(data=data.iloc[:,:2].apply(min_max), kind='kde', fill=True)
# plt.savefig('pairplot0')

"""# Data Cleaning (Preprocessing)"""

def split_scaler(df: pd.DataFrame, transformer: callable):
  X_train, x_test, Y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], train_size=0.7,random_state=0)
  trans = transformer
  trans.fit(X_train)
  X_train = pd.DataFrame(trans.transform(X_train), columns=X_train.columns)
  x_test = pd.DataFrame(trans.transform(x_test), columns=x_test.columns)
  return X_train, x_test, Y_train, y_test

# data.drop(columns=['Population'], inplace=True)

# Cleaning using z-score: I removed values with a z-score>=4
def remov_out(obj):
  z_score = lambda x: (x-x.mean())/(x.std())
  z_scores = obj.apply(z_score).to_numpy()
  ind_x, *_ = (np.abs(z_scores)>4.5).nonzero()
  return obj.drop(index=ind_x)
data_new = remov_out(data) #Defining new datadet (cleaned)

data_new.describe().to_latex()



"""### Splitting the dataset into training and testing set"""

val = train_test_split(data_new.iloc[:,:-1], data_new.iloc[:,-1], train_size=0.7, random_state=0)

"""## Feature Engineering & Visualization

>### Creating Pipelines: (i.) QuantileTransform, StandardScaler, and RobustScaler for Random Forest regression, and (ii.) StandardScaler for Logistic Regression and SVM
"""

class Gradient_Descent:
  '''
  This class implements the Gradient Descent algorithm for Linear Regression. User has multiple options
  for determining the best parameters that best fit the model\'s output: two optimization algorithms and
  one maximum likelihood calibration. The two optimization algorithms can be called by calling 'fit' or
  'optimization', both of which require the user to supply args

  fit: alpha (arg)
      'alpha' is the learning rate

  optimization: x_test, y_test
                'x_test' is the testing set of the feature dataset
                'y_test' is simply the true target testing set (intended for comparision)

  The maximum log-likelihood estimation requires no arguments. In future, this method will be improved as it was
  developed specifically for this dataset; it makes a few assumptions about the distribution of the dataset
  '''
  def __init__(self,X_train, Y_train, epsilon):
    self.Xtr = X_train.to_numpy()
    self.Ytr = Y_train.to_numpy()
    self.columns = X_train.columns
    # self.mins, self.maxes = np.split(X_train.agg(['min','max']).to_numpy().T,2,1)
    self.epsilon = epsilon
    self.n_feat = self.Xtr.shape[1]
    self.Xbias = np.c_[np.ones((self.Xtr.shape[0],)), self.Xtr]
    self.weights = np.random.randn(self.n_feat+1,)

  @staticmethod
  def mean_squared_error(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

  @staticmethod
  def accuracy(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    r2 = pow(corr,2)
    return r2

  def fit(self,alpha: float) -> tuple[np.ndarray[float],float, float]:
    n = self.Xtr.shape[0]
    oldtheta = self.weights
    for i in it.count():
      reduction = (1/n)*(self.Xbias.T).dot(self.Xbias.dot(oldtheta) - self.Ytr)
      newtheta = oldtheta - alpha*reduction
      if np.mean(np.square(newtheta - oldtheta))<self.epsilon:
        break
      else:
        oldtheta = newtheta
    self.weights = newtheta
    error = np.mean(np.square(self.Xbias.dot(self.weights) - self.Ytr))
    accuracy = spearmanr(self.Xbias.dot(self.weights),self.Ytr)
    # accuracy = self.accuracy(self.Ytr, self.Xbias.dot(self.weights))
    return newtheta, accuracy.statistic, error

  def optimization(self,x_test, y_test):
    alpha = np.linspace(0.0001,0.1,num=1000)
    alph_set = {'train_accuracies': [],'test_accuracies': []}
    weights = dict()
    for i in alpha:
      theta0, accu0, error0 = self.fit(i)
      weights[i] = theta0
      alph_set['train_accuracies'].append(accu0)
      pred, acc_1 = self.predict(x_test, y_test)
      alph_set['test_accuracies'].append(acc_1)
    for k in range(1000):
      if np.isclose(alph_set['train_accuracies'][k], alph_set['test_accuracies'][k], atol=0.01):
        print(f"Training Score:\n{alph_set['train_accuracies'][k]}\n Testing Score:\n{alph_set['test_accuracies'][k]}")
        print(f'Learning Rate = {alpha[k]}')
        print(f'Weights = {weights[alpha[k]]}')
        break
      else:
        continue
  def loglikelihood(self,dims):

    min = [-1]*self.Xbias.shape[1]
    max = [1]*self.Xbias.shape[1]
    engine = ssq.LatinHypercube(d=self.Xbias.shape[1], seed=0)
    X = ssq.scale(engine.random(n=2*self.Xbias.shape[0]), min, max)
    llhood = np.zeros(2*self.Xbias.shape[0])

    for (i, x) in enumerate(X):
      obs_var = np.sum(np.square(self.Xbias.dot(x)-self.Ytr))/(self.Xbias.shape[0]-dims)
      llhood[i] = np.sum(np.log((1/np.sqrt(2*np.pi*obs_var))*np.exp(-np.square(self.Ytr-self.Xbias.dot(x))/(2*obs_var))))

    X_filt = X[np.isfinite(llhood),:]
    llhood = llhood[np.isfinite(llhood)]
    idx_mle = np.argmax(llhood)
    print(f'Maximum Likelihood:{llhood[idx_mle]}')
    best_params = pd.Series(X_filt[idx_mle, :])
    r2 = self.accuracy(self.Ytr, self.Xbias.dot(best_params))
    print(f'Pearson\'s R2:{r2}')
    return best_params, r2

  def predict(self, data, y_true):
    n, m = data.to_numpy().shape
    n1,m1 = self.Xbias.shape
    if (m1 - 1) != m:
      raise Exception('Unequal dimensions')
    else:
      datnew = np.c_[np.ones((n,)), data.to_numpy()]
      pred = datnew.dot(self.weights)
      r2 = spearmanr(pred, y_true)
    # r2 = self.accuracy(y_true, pred)
    return pred, r2.statistic


"""> A visual of the newly cleaned dataset"""

# data_new.hist(bins=100)
# plt.tight_layout()

"""> The remaining outliers present in the data"""

# data.plot(kind='box')
# plt.tight_layout()

""">> Converting the target variable of the *California Housing Dataset* into something more suitable for classification.
*   Use binning: the min-max for the target variable is between 0 and 5, respectively; thus, I perform quartile binning



"""
# sns.heatmap(data_new.corr())
# plt.show()
# data_new.drop(columns=['AveRooms','Population'], inplace=True)
#Converting Regression model to Classification via binning
class_data = data_new.copy()
class_data['MedHouseVal'] = pd.qcut(data_new['MedHouseVal'], 4, labels=False)


val = train_test_split(data_new.iloc[:,:-1], data_new.iloc[:, -1], train_size=0.8, random_state=0)
val2 = train_test_split(class_data.iloc[:,:-1], class_data.iloc[:,-1], train_size=0.3, random_state=0)

pip = Pipeline([
     ('Transformer', StandardScaler()),
    ('RFModel', RandomForestRegressor(random_state=0))
])
pip1 = Pipeline([
    ('StandardScaler', StandardScaler()),
    ('Logistic Regression', LogisticRegression(random_state=3))
])
pip2 = Pipeline([
    ('StandardScaler', StandardScaler()),
    ('SVM', SVC(random_state=3))
])

"""## Model Building and Regularization

> We first need to perform a GridSearchCV to ascertain the correct hyperparmeters that offer the best performance
"""



# parameters = {
#     'n_estimators':[100,200,300,400],
#     'min_samples_split':[2,3,4,5],
#     'min_samples_leaf':[1,2,3,4,5],
#     'ccp_alpha':[0.0001,0.0002,0.0003, 0.0005]
# }
# parameters1 = {
#     'penalty': ['l1','l2'],
#     'C': [0.001,0.01,0.1,1,10,100,200]
# }
# parameters2 = {
#     'C': [0.001,0.01,0.1,1,10,100,200]
# }
# """>>Performing the GridSearchCV using the parameters of their respective models"""

# grid = GridSearchCV(RandomForestRegressor(random_state=0), parameters, n_jobs=-1)
# grid1 = GridSearchCV(LogisticRegression(random_state=3), parameters1)
# grid2 = GridSearchCV(SVC(random_state=3), parameters2)

"""> Hyperparameter tuning for the RandomForestRegressor model"""

# grid.fit(val[0],val[2])
# print(grid.best_params_)
# breakpoint
"""> Hyperparameter tuning for LogisticRegression model"""


"""> Hyperparameter tuning for the SuppostVectorClassifier (i.e., SVC) model"""



""">> Reparameterizing our Pipelines with the best model parameters:"""

# pip = Pipeline([
#      ('Transformer', QuantileTransformer(n_quantiles=4000, output_distribution='normal', random_state=0)),
#     ('RFModel', RandomForestRegressor(n_estimators=400,ccp_alpha=0.001, min_samples_leaf=1, min_samples_split=2,max_samples=2500, n_jobs=-1,random_state=0))
# ])
# # n_estimators=400,ccp_alpha=0.001, min_samples_leaf=1, min_samples_split=2,max_samples=2500, random_state=0, n_jobs=-1
# pip1 = Pipeline([
#     ('StandardScaler', StandardScaler()),
#     ('Logistic Regression', LogisticRegression(penalty='l2', C=100,random_state=3))
# ])
# pip2 = Pipeline([
#     ('StandardScaler', StandardScaler()),
#     ('SVM', SVC(C =10, random_state=3))
# ])
# parameters = {
#     'n_estimators':[100,200,300,400],
#     'min_samples_split':[2,3,4,5],
#     'min_samples_leaf':[1,2,3,4],
#     'ccp_alpha':[0.0001,0.0002,0.0003, 0.0005],
#     'max_samples':[1000,1500,2000,2500]
# }
# parameters1 = {
#     'penalty': ['l1','l2'],
#     'C': [0.001,0.01,0.1,1,10,100,200]
# }
# parameters2 = {
#     'C': [0.001,0.01,0.1,1,10,100,200]
# }
# grid = GridSearchCV(RandomForestRegressor(random_state=0),parameters, n_jobs=-1)
# grid.fit(val[0], val[2])
# print(grid.best_params_)
# pip3 = Pipeline([
#     ('Transformer', QuantileTransformer(n_quantiles=4000, output_distribution='normal',random_state=0)),
#     ('RFModel', Lasso(alpha=0.01))
# ])
# pip4 = Pipeline([
#     ('Transformer', QuantileTransformer(n_quantiles=4000, output_distribution='normal',random_state=0)),
#     ('RFModel', XGBRegressor())
# ])

"""## Dimensionality Reduction, Performance Measurement, and Model Evaluation"""

# pca = PCA()
# regmodel = pca.fit(val[0])
# xtst = pca.transform(val[1])

# pip.fit(val[0], val[2])
# print(f'PiPTraining Score: \n{pip.score(val[0], val[2])}')
# print(f'Test Score: \n{pip.score(val[1], val[3])}')

# pip3.fit(val[0],val[2])
# print(f'Training Score: \n{pip3.score(val[0], val[2])}')
# print(f'Test Score: \n{pip3.score(val[1], val[3])}')

# xgb = XGBRegressor(n_estimators=300, learning_rate=0.1)
# xgb.fit(val[0],val[2])
# print(f'Training Score: \n{xgb.score(val[0],val[2])}')
# print(f'Test Score: \n{xgb.score(val[1],val[3])}')





# The new Correlation matrix: MedInc and AveRooms remain the predominant predictor variables, which are highly correlated
# Based on VIF values and percentage of variance, the main predictors are 'Median Income Value', 'HouseAge', 'Population', and 'Average Occupation'

"""### Feature Selection"""

# FS = data_new.copy()
# fs_val = train_test_split(FS.iloc[:,:-1], FS.iloc[:,-1], train_size=0.7, random_state=1)

"""##Split Data & Define Model

## Model Fitting
"""

# print(f'Validation score for Regression: \n {cross_val_score(pip, val[0], val[2], cv=10)}')
# print(f'Validation score for Classification:\n {cross_val_score(pip2, val2[0], val2[2], cv=10)}')

# pip.fit(val[0], val[2])
# print(f'Training score: \n {pip.score(val[0], val[2])}')
# print(f'Test score:\n {pip.score(val[1], val[3])}')
# qt = QuantileTransformer(n_quantiles=6000, output_distribution='normal', random_state=0)
# rt = RobustScaler()
# data_new.drop(columns=['AveRooms','AveBedrms', 'Population'], inplace=True)





"""## GridSearch

## Fitting Model With New Parameters
"""

rng = np.random.default_rng(seed=0)
def fom(f: callable, data: np.ndarray[float]):
  y_pred = f(data)
  Vy = np.var(y_pred)
  n,m = data.shape
  Si = np.zeros(m)
  for i in range(m):
    X = data.copy()
    x = X[:,i]
    X = np.roll(X, 1, axis=0)
    X[:,i] = x
    Si[i] += 1 - np.mean(np.square(y_pred - f(X)))/(2*Vy)
  return Si
def som(f: callable, data: np.ndarray[float], fom: np.ndarray[float]):
  y_pred = f(data)
  Vy = np.var(y_pred)
  n,m = data.shape
  Si = np.zeros((m,m))
  for i in range(m):
    for k in range(m):
      X = data.copy()
      x = X[:,[i,k]]
      X = np.roll(X,1, axis=0)
      X[:,[i,k]] = x
      Si[i,k] += 1 - np.mean(np.square(y_pred-f(X)))/(2*Vy) - fom[i] - fom[k]
  return Si
qt = QuantileTransformer(n_quantiles=4000, output_distribution='normal', random_state=0)
# val = train_test_split(pd.DataFrame(qt.fit_transform(data_new.iloc[:,:-1]), columns=data_new.columns[:-1]), data_new.iloc[:, -1], train_size=0.7, random_state=0)
# RF = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, min_samples_split=2, max_samples=2000, ccp_alpha=0.0001,n_jobs=-1,random_state=0)
# # n_estimators=400, min_samples_leaf=3, min_samples_split=2, max_samples=2000, ccp_alpha=0.0001,
# RF.fit(val[0], val[2])
# print(f'Training Score:\n {RF.score(val[0], val[2])}\n')
# print(f'Testing Score:\n {RF.score(val[1], val[3])}')
# print(f'Cross-Validation Score:\n {np.mean(cross_val_score(RF, val[1],val[3], cv=10, n_jobs=-1))}')

# gd = Gradient_Descent(val[0], val[2], 0.000001)
# weight, r2, error = gd.fit(0.01)
# pred, r21 = gd.predict(val[1], val[3])

# An Idea

# i,j =(data_new['Divide'].to_numpy() == 0).nonzero()
# print(i,j)
# train, validate, test = np.split(pd.DataFrame(qt.fit_transform(data_new.sample(frac=1)), columns=data_new.columns), [int(0.6*len(data_new)), int(0.8*len(data_new))])
# gd = Gradient_Descent(train.iloc[:,:-1], train.iloc[:,-1], 1e-15)
# weights, r, error = gd.fit(1e-5)
# print(f'Weights:\n{weights}\n')
# print(f'Accuracy:\n{r}\n')
# print(f'Error:\n{error}\n')
# pred , r2 = gd.predict(validate.iloc[:,:-1], validate.iloc[:,-1])
# print(f'Validation Score:\n{r2}\n')

def SGD(data, alpha, epsilon):
  n, m = data.to_numpy().shape
  rand.seed(a=0)
  theta = np.random.randn(m)
  Xbias = np.c_[np.ones((n,)), data]
  


# pred, r2 = gd.predict(val2[1],val2[3])
# print(f'Training Score:\n {r}')
# print(f'Testing Score:\n {r2}')
# gd = Gradient_Descent(val[0], val[2], 0.000001)
# weight, r2, error = gd.fit(0.01)
# pred, r21 = gd.predict(val[1], val[3])
# print(f'\n Gradient Descent:')

# print(f'\n Training Score:\n{r2}\n')
# print(f'Testing Score:\n{r21}')

# lr = LinearRegression(n_jobs=-1)
# lr.fit(val[0], val[2])

# print(f'\n Training Score:\n{lr.score(val[0], val[2])}\n')
# print(f'Testing Score:\n{lr.score(val[1], val[3])}')

# Si = fom(RF.predict, val[0])
# corr_matrix = np.corrcoef(val[2], val[0].dot(Si))
# corr = corr_matrix[0,1]
# r2 = pow(corr,2)
# print(r2)
# print(f'\n{Si}')
# Sik = som(RF.predict, val[0], Si)
# print(f'\n:{Sik}')

