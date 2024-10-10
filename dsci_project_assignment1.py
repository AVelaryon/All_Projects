
import numpy as np
import pandas as pd
import itertools as it
import random as rand
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import SGDRegressor, ARDRegression
from sklearn.preprocessing import QuantileTransformer, normalize, StandardScaler, RobustScaler, FunctionTransformer, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import scipy.stats.qmc  as ssq
import scipy as sy
import matplotlib.pyplot as plt
import seaborn as sns

"""## Data Exploration and Visualization"""

fetch_data = fetch_california_housing(as_frame=True)
data = fetch_data['frame'].copy()

data.describe()

"""> Dataset distribution before pre-processing"""



"""# Data Cleaning (Preprocessing)"""

def split_scaler(df: pd.DataFrame, transformer: callable) -> tuple[pd.DataFrame]:
  '''
  This function takes a DataFrame and Transformer as arguments, splits the DataFrame into
  training and testing sets, and scales the training and testing sets using the provided
  transformer.
  '''
  X_train, x_test, Y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], train_size=0.7,random_state=0)
  trans = transformer
  trans.fit(X_train)
  X_train = pd.DataFrame(trans.transform(X_train), columns=X_train.columns)
  x_test = pd.DataFrame(trans.transform(x_test), columns=x_test.columns)
  return X_train, x_test, Y_train, y_test

def remov_out(obj: pd.DataFrame, z: float) -> pd.DataFrame:
  '''
  This function admits two arguments: DataFrame and z-score (in that order). The purpose of
  this function is to detect and remove outliers using the provided z-score value.
  '''
  z_score = lambda x: (x-x.mean())/(x.std())
  z_scores = obj.apply(z_score).to_numpy()
  ind_x, *_ = (np.abs(z_scores)>z).nonzero()
  return obj.drop(index=ind_x)

data_new = remov_out(data, 6) #Defining new datadet (cleaned)
unique, indices = np.unique(data_new.Latitude, return_index=True)
new_data = data_new.iloc[indices, :].copy()

"""> Data distribution post-processing."""



"""> Data description of the post-processed dataset"""


"""> Creating a separate dataset for classification. Given the distribution of the ```MedHouseVal``` target variable, I will employ quartile binning using ```pd.qcut```.

>> Converting the target variable of the *California Housing Dataset* into something more suitable for classification.
*   Use binning: the min-max for the target variable is between 0 and 5, respectively; thus, I perform quartile binning
"""

classification  = data_new.copy()
class_bins = pd.qcut(classification.iloc[:,-1], 4, labels=False)
classification.MedHouseVal = class_bins

"""### Splitting the dataset into training and testing set"""

XtrR, xtsR, YtrR, ytsR = split_scaler(data_new,RobustScaler()) #For Regression
XtrC, xtsC, YtrC, ytsC = split_scaler(classification, QuantileTransformer(n_quantiles=4000, output_distribution='normal', random_state=0)) # For Classification

"""## Feature Engineering & Visualization

>### Creating Pipelines: (i.) QuantileTransform, StandardScaler, and RobustScaler for Random Forest regression, and (ii.) StandardScaler for Logistic Regression and SVM
"""

class Gradient_Descent:
  '''
  This class implements the Gradient Descent algorithm for Linear Regression. The user has multiple options
  for determining the best parameters that best fit the model\'s output: two optimization algorithms and
  one maximum likelihood calibration. The two optimization algorithms can be called by calling 'fit' or
  'optimization', both of which require the user to supply args

  fit: alpha (arg)
      'alpha' is the learning rate

  optimization: x_test, y_test
                'x_test' is the testing set of the feature dataset.
                'y_test' is simply the true target testing set (intended for comparision).
                This class method is an attempt to optimize the learning rate by successively iterating
                through a range of learning rates ([0.0001, 0.1]), returning the best learning rate, training score and
                testing score.

  log-likelihood: requires no arguments. This calibration makes the relatively general assumption that the
                  data is uniformly distributed. In future, the user will be able to specify what type of
                  distribution.
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
  def root_squared_error(y_pred, y_true):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

  @staticmethod
  def accuracy(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    r2 = pow(corr,2)
    return r2

  def fit(self,alpha: float) -> tuple[np.ndarray[float],float, float]:
    n = self.Xtr.shape[0]
    theta = self.weights
    errorset = []
    olderror = self.root_squared_error(self.Xbias.dot(theta), self.Ytr)
    errorset.append(olderror)
    for i in it.count():
      reduction = (2/n)*(self.Xbias.T).dot(self.Xbias.dot(theta) - self.Ytr)
      theta -= alpha*reduction
      newerror = self.root_squared_error(self.Xbias.dot(theta), self.Ytr)
      if abs(newerror - olderror)<self.epsilon:
        break
      else:
        errorset.append(newerror)
        olderror = newerror
    self.weights = theta
    error = self.root_squared_error(self.Xbias.dot(self.weights),self.Ytr)
    # accuracy = spearmanr(self.Xbias.dot(self.weights),self.Ytr)
    return theta, error, errorset

  def optimization(self,x_test, y_test):
    alpha = np.linspace(0.0001,0.1,num=1000)
    alph_set = {'train_error': [],'test_error': []}
    weights = dict()
    for i in alpha:
      theta0,error0, errorset = self.fit(i)
      weights[i] = theta0
      alph_set['train_error'].append(error0)
      pred, error = self.predict(x_test, y_test)
      alph_set['test_error'].append(error)
    for k in range(1000):
      if np.isclose(alph_set['train_error'][k], alph_set['test_error'][k], atol=0.01):
        print(f"Training Error:\n{alph_set['train_error'][k]}\n Testing Error:\n{alph_set['test_error'][k]}")
        print(f'Learning Rate = {alpha[k]}')
        print(f'Weights = {weights[alpha[k]]}')
        break
      else:
        continue

  def loglikelihood(self):
    min = [-1]*self.Xbias.shape[1]
    max = [1]*self.Xbias.shape[1]
    engine = ssq.LatinHypercube(d=self.Xbias.shape[1], seed=0)
    X = ssq.scale(engine.random(n=2*self.Xbias.shape[0]), min, max)
    llhood = np.zeros(2*self.Xbias.shape[0])

    for (i, x) in enumerate(X):
      model = self.Xbias.dot(x)
      llhood[i] = np.sum(np.log(sy.stats.norm.pdf((model - self.Ytr))))

    X_filt = X[np.isfinite(llhood),:]
    llhood = llhood[np.isfinite(llhood)]
    idx_mle = np.argmax(llhood)
    print(idx_mle)
    best_params = pd.Series(X_filt[idx_mle, :])
    r2 = self.accuracy(self.Ytr, self.Xbias.dot(best_params))
    return best_params, r2

  def predict(self, data, y_true):
    n, m = data.to_numpy().shape
    n1,m1 = self.Xbias.shape
    if (m1 - 1) != m:
      raise Exception('Unequal dimensions')
    else:
      datnew = np.c_[np.ones((n,)), data.to_numpy()]
      pred = datnew.dot(self.weights)

    error = self.root_squared_error(pred, y_true)
    return pred, error

def SGD(X_train, Y_train, alpha,batch_size, epsilon):
  n, m = X_train.to_numpy().shape
  weights = np.random.randn(m+1)
  X_train = X_train.to_numpy()
  Y_train = Y_train.to_numpy()
  dat_bias = np.c_[np.ones((n,1)), X_train]
  error_set = []
  for i in it.count():
    indx = np.random.choice(n, size=n)
    dat_bias = dat_bias[indx]
    Y_train = Y_train[indx]
    for k in range(0,n, batch_size):
      j = min(k+batch_size, n)
      X_samp = dat_bias[k:j]
      Y_samp = Y_train[k:j]
      error = np.sqrt(np.square(X_samp.dot(weights)-Y_samp).mean())
      grad  = (2/batch_size)*(X_samp.T).dot(X_samp.dot(weights)-Y_samp)
      weights -=alpha*grad
      newerr = np.sqrt(np.square(X_samp.dot(weights)-Y_samp).mean())
      error_set.append(newerr - error)

      if abs(newerr - error)<epsilon or i==1000:
        break
      else:
        error = newerr
  return weights, error_set


# gd = Gradient_Descent(XtrR, YtrR, 1e-9)
# gd.loglikelihood()

# ard = ARDRegression()
# ard.fit(XtrR, YtrR)
# ard.score(xtsR, ytsR)



# # This Training set is just so that there is a unscaled dataset
qt = QuantileTransformer(n_quantiles=4000, output_distribution='normal', random_state=0)
val = train_test_split(data_new.iloc[:,:-1], data_new.iloc[:,-1], train_size=0.7, random_state=0)
val2 = train_test_split(classification.iloc[:,:-1],classification.iloc[:,-1], train_size=0.7, random_state=0)
train, validate, test = np.split(pd.DataFrame(qt.fit_transform(data_new.sample(frac=1)), columns=data_new.columns), [int(0.6*len(data_new)), int(0.8*len(data_new))])

"""## Model Building

### Gradient Descent Model

> A visual of the newly cleaned scaled dataset
"""



"""> Performing Gradient Descent and Stochastic Gradient Descent using ```sklearn.linear_model.SGDRegressor```
>> Taking your suggestion of using a different metric of performance: Root Mean Squared Error (RMSE)
"""



""">Gradient Descent's weight parameters are: [ 2.02573908, 0.79519682, 0.32807362, -0.07464749,0.11852886,0.13573368, -0.25860996 -0.38521398,-0.29344148]\
The first quantity in the above weight vector denotes the bias, which happens to be pretty large relative to other parameters. I should note, though, that the data was scaled using the ```QuantileTransformer```

"""



"""> The above plot depicts **Gradient Descent's** predictive capabilities, having a RMSE of $\approx 0.795$ on the **test dataset**.

### Stochastic Gradient Descent

>I will now use the ```SGDRegessor``` default parameters; a ```GridSearchCV``` will be performed in the next section, using ```train```, ```validate```, and ```test``` sets defined above.
"""



"""### LogisticRegression Model & SVC

> As tasked, I transformed the dataset into a classification dataset whereby the target variable was binned on the basis of its inter-quartile range. That is, for each quartile, a value between 0-3 was assigned to sample of the target variable that lied within that quartile range.\
Like ```SGDRegressor```, I will use default parameters and reserve ```GridSearchCV``` for the next section. Moreover, I will use a ```Pipeline``` for both models.\
Here is a histogram of the counts:
"""

classification.MedHouseVal.hist()


"""> ```LogisticRegression```:"""



""">> Looking at the ```recall``` for both training and testing score, the ```recall``` suggests that it's difficult for LR to make *positive* prediction for data in the 25-50th-percentile; the ```precision``` and ```accuracy``` aren't that great either.

>```SVC``` (i.e., Support Vector Machine classifier):
"""



""">>```SVC``` performed equally as well as the ```LogisticRegression``` model. \
Therfore, since I had to transform the target variable into one that is for classification--essentially imposing a bias on the target variable and obfuscating the associated Median House Value--I will, henceforth, not consider the former and latter classification models as this dataset is for regression. Notwithstanding the performance of the classification models, it is sensible and tractable to use them as predictive models since each sample in the dataset corresponds to a region in the State of California. To elucidate, one could define a 'threshold' using the ```Latitude``` and ```Longitude``` parameters and assign a classification of, say, 'rich' or 'poor' to those that live along the coast of California or inland, respectively.

### RandomForestRegressor & XGBRegressor
Similar to the previous models, I will create a ```Pipeline``` with the desired data-transformer and model with default parameters.
"""



""">> As you can see above, ```RandomForestRegressor``` performs pretty well on the training data, but the apparent difference between the training score and test score suggests over-fitting has taken place.

>```XGBRegressor```:
"""



""">> ```XGBRegressor``` performed slightly better than ```RandomForestRegressor``` but, again, the difference between the training and test score is indicative of over-fitting.\
\
We are now in a position to further investigate how the model assumption (i.e., model features are normally distributed) impacts my models predictive power. In the next section, I will do just that, performing a ```GridSearchCV``` to obtain the best parameters for each model that prevents over-fitting, restores generalizability, and enhances the respective predictive power of the models. I will also implement principal component analysis (or PCA), as suggested, to retain important, pertinent features.

## Regularization Dimensionality Reduction, Performance Measurement, and Model Evaluation

> We first need to perform a GridSearchCV to ascertain the correct hyperparmeters that offer the best performance
"""

parameters = {
    'n_estimators':[100,200,300,400],
    'min_samples_split':[2,3,4,5],
    'min_samples_leaf':[1,2,3,4],
    'ccp_alpha':[0.0001,0.0002,0.0003, 0.0005],
    'max_samples':[1000,1500,2000,2500]
}
param_grid_xgb = {
    'n_estimators': [100, 200, 300],  # Number of gradient boosted trees
    'learning_rate': [0.0001, 0.001, 0.01],  # Step size shrinkage used in update to prevent overfitting
    'min_child_weight': [3, 4 ,5],  # Minimum sum of instance weight (hessian) needed in a child
    'subsample': [0.7, 0.8, 1.0],  # Subsample ratio of the training instances
    'colsample_bytree': [0.7, 0.8, 1.0]  # Subsample ratio of columns when constructing each tree
}
param_sgd = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.0002, 0.0003, 0.0004],
    'max_iter': [100, 300, 500, 700, 1000],
    'tol':[1e-3, 1e-5,1e-7,1e-11],
    'learning_rate':['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0':[1e-2, 1e-4]
}

""">>Performing the GridSearchCV using the parameters of their respective models"""



""">With these hyperparameters, I may now perform cross-validation, using ```sklearn.model_selection.cross_val_score``` on the training data. Using ```cross_val_score``` ```cv=20``` (i.e., 20-Fold), each model's performance is as follows:"""


"""> In order to improve the performance of each model, it may be wise to performe dimensionality reduction. Let's perform PCA

"""

data_new.drop(['Population', 'AveRooms', 'AveBedrms'], axis=1, inplace=True)
new_data.drop(['Population', 'AveRooms', 'AveBedrms'], axis=1, inplace=True)
"""> And, re-splitting"""
def logabf(x):
  return np.log(np.absolute(x))
val = split_scaler(data_new, QuantileTransformer(n_quantiles=4000, output_distribution='normal',random_state=0))
val2 = split_scaler(new_data, QuantileTransformer(n_quantiles=3000, output_distribution='normal',random_state=0))
"""> Now that I've established that I'm not dealing the a linear model, Gradient Descent and Stochastic Gradient Descent will be discarded as model considerations and the normality assumption will be rejected in favor of ```RobustScaler```"""

data_new.columns

RF = RandomForestRegressor(n_estimators=300, min_samples_split=2, min_samples_leaf=2, max_samples=2500, ccp_alpha=0.0001, random_state=0)
RF.fit(val[0],val[2])

xgb = XGBRegressor(n_estimators=300, subsample=0.8, min_child_weight=3, learning_rate=0.01, colsample_bytree=0.8, random_state=0)
xgb.fit(val[0], val[2])

parameters = {
    'n_estimators':[100,200,300,400],
    'min_samples_split':[2,3,4,5],
    'min_samples_leaf':[1,2,3,4],
    'ccp_alpha':[0.0001,0.0002,0.0003, 0.0005],
    'max_samples':[1000,1500,2000,2500]
}
param_grid_xgb = {
    'n_estimators': [100, 200, 300],  # Number of gradient boosted trees
    'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
    'min_child_weight': [1, 3, 5, 7],  # Minimum sum of instance weight (hessian) needed in a child
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Subsample ratio of the training instances
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Subsample ratio of columns when constructing each tree
}

# grid = GridSearchCV(RandomForestRegressor(random_state=0), parameters, n_jobs=-1)
# grid1 = GridSearchCV(XGBRegressor(interaction_constraints=[['MedInc','HouseAge','AveOccup'], ['Latitude', 'Longitude']], learning_rate=0.01, random_state=0), param_grid_xgb, n_jobs=-1)


# RF = RandomForestRegressor(n_estimators=300, min_samples_split=2, min_samples_leaf=1, max_samples=2500, ccp_alpha=0.0001, random_state=0, n_jobs=-1)
# RF.fit(val[0],val[2])

# xgb = XGBRegressor(n_estimators=300, subsample=0.5, min_child_weight=7, interaction_constraints=[['MedInc','HouseAge','AveOccup'], ['Latitude', 'Longitude']],learning_rate=0.01, colsample_bytree=0.8, random_state=0, n_jobs=-1)
# xgb.fit(val[0], val[2])
# print(f'======RandomForestRegressor======')
# print(f'Cross-Validation Score:\n{cross_val_score(RF, val[0], val[2], cv=10).mean()}')

# print(f'\n======XGBRegressor======')
# print(f'Cross-Validation Score:\n{cross_val_score(xgb, val[0], val[2], cv=10).mean()}')

# print(f'Training Score:\n{RF.score(val[0], val[2])}')
# print(f'Testing Score:\n{RF.score(val[1], val[3])}')

# print(f'Training Score:\n{xgb.score(val[0], val[2])}')
# print(f'Testing Score:\n{xgb.score(val[1], val[3])}')
# {'colsample_bytree': 0.8, 'max_depth': 12, 'min_child_weight': 7, 'n_estimators': 300, 'subsample': 0.5}
#{'ccp_alpha': 0.0001, 'max_samples': 2500, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
param = {
  'max_iter':[1200, 1500, 1600, 2000, 3000, 4000],
  'batch_size': [200,300, 400,500, 600, 700, 800 ],
  'learning_rate':['constant','invscaling', 'adaptive'],
  'activation':['relu', 'identity', 'logistic', 'tanh'],
  'solver':['adam', 'sgd', 'lbfgs']
}
mlp_params = {
    'hidden_layer_sizes':[(100,), (200,), (300,), (400,)],
    'activation':['identity', 'logistic', 'tanh', 'relu']
}
mlp = MLPRegressor(solver='adam', learning_rate='constant', max_iter=1200, random_state=0)
mlp.fit(XtrR, YtrR)
mlp_search = GridSearchCV(mlp, mlp_params, n_jobs=-1)
mlp_search.fit(XtrR, YtrR)
print(f'Best Parameters:\n{mlp_search.best_params_}')
# grid = GridSearchCV(MLPRegressor(tol=1e-11, random_state=0), param, n_jobs=-1)
# grid.fit(val2[0],val2[2])
# print(grid.best_params_)
# mlp = MLPRegressor(max_iter=800, batch_size=500, solver='adam',activation='relu',tol=1e-9,learning_rate='constant', random_state=0)
# mlp1 = MLPRegressor(max_iter=800, batch_size=500, solver='adam',activation='relu',tol=1e-9,learning_rate='constant', random_state=0)
# mlp1.fit(val2[0],val2[2])
# mlp.fit(val[0], val[2])
# print(f'Training Score:\n{mlp.score(val[0],val[2])}\n')
# print(f'Cross-Validation Score:\n{cross_val_score(mlp, val[0], val[2], cv=5).mean()}')
# print(f'Test Score:\n{mlp.score(val[1], val[3])}')
# # {'activation': 'relu', 'learning_rate': 'constant', 'solver': 'adam'}

# print(f'Training Score:\n{mlp1.score(val2[0],val2[2])}\n')
# print(f'Cross-Validation Score:\n{cross_val_score(mlp1, val2[0], val2[2], cv=5).mean()}')
# print(f'Test Score:\n{mlp1.score(val2[1], val2[3])}')
# {'activation': 'relu', 'learning_rate': 'constant', 'solver': 'adam'}