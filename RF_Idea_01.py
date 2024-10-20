#%%
import numpy as np
import itertools as it
import functools as ft
import matplotlib.pyplot as plt
import scipy as sy
import scipy.special as ss
import scipy.stats as stat
import scipy.stats.qmc as ssq
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import collections as col
from timeit import timeit


#%%
np.set_printoptions(suppress=True, precision=5)

class Vector_Constructor:
    def __init__(self, f: callable, bounds: list, var: int, n: int):
        self.f = f
        self.b = bounds
        self.var = int(var)
        self.n = n

    def construct(self):
        engine = ssq.LatinHypercube(d=self.var, seed = 0)
        x = ssq.scale(engine.random(n=2*self.n), self.b['lb'], self.b['ub'])
        A = np.zeros((self.n, len(self.b['lb'])+1))
        for i in it.count():
            if i<self.n:
                A[i,:] = [*x[i,:],self.f(x[i,:])]
            else:
                break
        df = pd.DataFrame(A, columns=['a','teq','s0','F'])
        return df

## Test Functions
    
def Ishigami(k):
    x,y,z = k
    a = 7
    b = 0.1
    return np.sin(x) + a*pow(np.sin(y),2) + b*pow(z, 4)*np.sin(x)

def test(k:list):
    x,y,z = k
    return 0.2*x - 5*y + 10*y*z
# Rahmstorf Model
def gmsl_model(a=3.4, Teq=-0.5, S0=0, T_forcing=None, dt=1):
    n_time = len(T_forcing)
    S = np.zeros((n_time,))
    S[0] = S0
    for t in range(1, n_time):
        S[t] = S[t-1] + np.multiply(a,dt*(T_forcing[t-1]-Teq))
    return S


#Define Maximum-Log-Likelihood
def log_likelihood(parameters, forcing, data):
    model = gmsl_model(a=parameters[0], Teq=parameters[1], S0=parameters[2], T_forcing=forcing, dt=1)
    llhood = np.sum(np.log(stat.norm.pdf(model-data["sealevel"], loc=np.zeros(len(model)), scale=data["sigma_sealevel"])))
    return llhood

#NOAA Temperature & Sealevel Data
dfS = pd.read_csv(r"D:\MATH722\MATH722\data_temperature_sealevel.csv")

# normalize temperature and sea level data relative to 1951-1980
T_1951_1980 = dfS.loc[(dfS["year"] >= 1951) & (dfS["year"] <= 1980), "temperature"].mean()
dfS["temperature"] = dfS["temperature"] - T_1951_1980
S_1951_1980 = dfS.loc[(dfS["year"] >= 1951) & (dfS["year"] <= 1980), "sealevel"].mean()
dfS["sealevel"] = dfS["sealevel"] - S_1951_1980

# Set maximum number of rows
pd.set_option('display.max_rows', None)


#Setting Initial Conditions
S0 = dfS['sealevel'][0]

def new_gmsl(k):
    a,teq, s0 = k
    return gmsl_model(a=a, Teq=teq,S0=s0, T_forcing=dfS['temperature'],dt=1)[-1]
# Bounds
#[0,-1,-200][5,2,200]
lb = [-np.pi]*3
ub = [np.pi]*3
# lb = [0,-1,-100]
# ub = [5,2,100]
d = len(lb)
bounds = {'lb':lb, 'ub':ub}
definition = {'modvar': ['a','teq','s0'], 'numvar': 3, 'num_samp': 20000, 'arg_vals':None,'lowb': lb, 'upb': ub,'n_boot': 0}
# Number of Elements

n =50000
n_bins = 70
vc = Vector_Constructor(Ishigami, bounds,3,n)
df  = vc.construct().copy().to_numpy()
##df_n = vc.construct().copy()
##model = Sobol(new_gmsl, definition)


# Model Run


n_estimators = 150
RF = RandomForestRegressor(n_estimators=n_estimators, random_state=0)

# Validation Procedure


val = train_test_split(df[:,:3], df[:,3], test_size=0.3, train_size=0.7, shuffle=False, random_state=0)
val[0] = val[0].astype('float32')

RF.fit(val[0], val[2])





# Permutation Procedure

n_time = 3
X_shuf =val[0].copy()
rng = np.random.default_rng(0)
#%%
# PVI: Total and First-order Measures 
X_new = val[1].copy()
y_pred1 = RF.predict(val[1])

def pvi1(data: np.ndarray[float], y_pred: np.ndarray[float], f: callable) -> np.ndarray[float]:
    '''
    ===First-Order Permutation Variable Importances===

        data: your training dataset
        y_pred: your model output (specifically the prediction of your 'data' arg)
        f: in this case, simply RF.predict
    '''
    rows, columns = data.shape
    V_y = np.var(y_pred)
    u = np.zeros((columns,))
    for i in it.count():
        if i<columns:
            d = data.copy()
            k = d[:,i]
            d = np.roll(d,1, axis=0)
            d[:,i] = k
            u[i] = np.square(y_pred - f(d)).sum()/rows
        else:
            break
    Si = 1 - u/(2*V_y)
    return Si
def pvi_t(data: np.ndarray[float], y_pred: np.ndarray[float], f: callable) -> np.ndarray[float]:
    '''
    ===Total-Order Permutation Variable Importances===

        data: your training dataset
        y_pred: your model output (specifically the prediction of your 'data' arg)
        f: in this case, simply RF.predict
    '''
    rows, columns = data.shape[0],data.shape[1]
    u = np.zeros((columns,))
    V_y = np.var(y_pred)
    for i in it.count():
        if i<columns:
            d = data.copy()
            d[:,i] = np.roll(d[:,i], 1, axis=0)
            u[i] = np.square(y_pred, f(d)).sum()/rows
        else:
            break
    S_t = u/(2*V_y)
    return S_t

# Second-Order Importances



Si = pvi1(X_new, y_pred1, RF.predict)
print(Si)
print(timeit('pvi2(X_new, y_pred1, RF.predict, S_I=Si, n_boots=30, alpha=0.025)', number=100))

#%%


###Distance Correlation
##
##
##ind_X = rng.choice(val[1].shape[0], size=val[1].shape[0], replace=False)
##AS = val[1][ind_X,:].copy()
####BS = df[-ind_X, :3].copy()
##yp = RF.predict(AS)
####p0 = stat.norm.pdf(y_pred, loc=np.mean(y_pred), scale=np.std(y_pred))
##nn = val[1].shape[0]
##A = np.zeros((3,AS.shape[0],AS.shape[0]), dtype=np.double)
##B,B1 = np.zeros_like(A), np.zeros_like(A)
##
####C = np.zeros_like(B)
##a = np.zeros_like(A)
##b, b1 = np.zeros_like(a), np.zeros_like(a)
####c = np.zeros_like(b)
##for i in it.count():
##    if i<3:
##        S = AS.copy()
##        ss = S[:,i].copy()
##        S = rng.permuted(S, axis=0)
##        S[:,i] = ss
##        y = RF.predict(S)
##        for l in it.chain(range(A.shape[1])):
##            for m in it.chain(range(A.shape[1])):
##                a[i,l,m] = norm(S[l]-S[m])
##                b[i,l,m] = abs(yp[l]-y[m])
##                b1[i,l,m] = abs(yp[l]-yp[m])
##    else:
##        break
##        
##
##        
##
##a_hatl = np.array([[np.sum(a[i,l,:])/(nn-2) for l in range(A.shape[1])] for i in it.chain(range(3))])
##a_hatm = np.array([[np.sum(a[i,:,m])/(nn-2) for m in range(A.shape[1])]for i in it.chain(range(3))])
##a_hat = np.sum(a)/((nn-1)*(nn-2))
##
##b_hatl = np.array([[np.sum(b[i,l,:])/(nn-2) for l in it.chain(range(A.shape[1]))]for i in it.chain(range(3))])
##b1_hatl = np.array([[np.sum(b1[i,l,:])/(nn-2) for l in it.chain(range(A.shape[1]))] for i in it.chain(range(3))])
##b_hatm = np.array([[np.sum(b[i,:,l])/(nn-2) for l in it.chain(range(A.shape[1]))] for i in it.chain(range(3))])
##b1_hatm = np.array([[np.sum(b1[i,:,m])/(nn-2) for m in it.chain(range(A.shape[1]))]for i in it.chain(range(3))])
##b_hat = np.sum(b)/((nn-1)*(nn-2))
##b1_hat = np.sum(b1)/((nn-1)*(nn-2))
##
##for i in it.count():
##    if i< 3:
##        for l in it.chain(range(A.shape[1])):
##            for m in it.chain(range(A.shape[1])):
##                A[i,l,m] += (a[i,l,m] - a_hatl[i,l] - a_hatm[i,m] +a_hat)
##                B[i, l,m] += (b[i,l,m] - b_hatl[i,l] - b_hatm[i,m] + b_hat)
##                B1[i,l,m] += (b1[i,l,m] - b1_hatl[i,l] - b1_hatm[i,m] + b1_hat)
##    else:
##        break
##
##indB = np.indices(B.shape)
##i,l,m = (indB[1]!=indB[2]).nonzero()
##dV2BB = np.array([np.sum(B.dot(B)[i,l,m])/(nn*(nn-3)) for i in range(3)])


##            C[i,l,m] += (c[i,l,m] - c_hatl[i,l] - c_hatm[i,m] +c_hat)
##for i in it.chain(range(3)):
##    S = BS.copy()
##    for l in it.chain(range(A.shape[1])):
##        for m in it.chain(range(A.shape[1])):
##            c[i,l,m] = norm(S[l] - S[m])
##
##
###for i in it.chain(range(3))]
### The above F-loop is for perm-variables

##c_hatl = np.array([[np.sum(c[i,l,:])/(nn-2) for l in range(A.shape[1])]for i in it.chain(range(3))])
##c_hatm = np.array([[np.sum(c[i,:,m])/(nn-2) for m in range(A.shape[1])]for i in it.chain(range(3))])

####a_hat = np.mean(a)
##c_hat = np.sum(c)/((nn-1)*(nn-2))
####c_hat = np.mean(c)

####b_hatl = np.array([np.mean(b[l,:]) for l in range(A.shape[1])])

##b_hat = np.mean(b)


        

#Inner Products
####ii,jj, kk = (indices[1]!=indices[2]).nonzero()
##inner_AB =(np.einsum('ijk,ijk->i',np.triu(A, k=1),np.triu(B, k=1))+np.einsum('ijk,ijk->i',np.tril(A,k=-1),np.tril(B,k=-1)))/(nn*(nn-3))
##inner_BC = (np.einsum('ijk,ijk->i',np.triu(B, k=1),np.triu(C, k=1))+np.einsum('ijk,ijk->i',np.tril(B,k=-1),np.tril(C,k=-1)))/(nn*(nn-3))
##inner_BB = (np.einsum('ijk,ijk->i',np.triu(B, k=1),np.triu(B, k=1))+np.einsum('ijk,ijk->i',np.tril(B,k=-1),np.tril(B,k=-1)))/(nn*(nn-3))
##inner_AC = (np.einsum('ijk,ijk->i',np.triu(A, k=1),np.triu(C, k=1))+np.einsum('ijk,ijk->i',np.tril(A,k=-1),np.tril(C,k=-1)))/(nn*(nn-3))
##inner_AA = (np.einsum('ijk,ijk->i',np.triu(A, k=1),np.triu(A, k=1))+np.einsum('ijk,ijk->i',np.tril(A,k=-1),np.tril(A,k=-1)))/(nn*(nn-3))
##inner_CC = (np.einsum('ijk,ijk->i',np.triu(C, k=1),np.triu(C, k=1))+np.einsum('ijk,ijk->i',np.tril(C,k=-1),np.tril(C,k=-1)))/(nn*(nn-3))
##
### DCOR
##RXSY = (inner_AB)/np.sqrt(inner_AA*inner_BB)
##RYX_S = (inner_BC)/np.sqrt(np.dot(inner_AB,inner_BC))
##RXSS_ = (inner_AC)/np.sqrt(np.dot(inner_AA,inner_CC))
##
##SF = RXSY
##SXF = (SF - RXSS_*RYX_S)/(np.sqrt(1-RXSS_*RXSS_*RXSS_*RXSS_))
##                    b[i,l,m] += np.abs(b1[l]-b1[m])
##                    b_hatl = (np.einsum('lm->l',b[i,:,:])/(nn-2))
##                    b_hatm = (np.einsum('lm->m',b[i,:,:])/(nn-2))
##                    b_hat = np.einsum('lm->', b[i,:,:])/((nn-1)*(nn-2))
##                    B[i,l,m]+= (b[i,l,m] - b_hatl - b_hatm + b_hat)
##                    
##                    c_cop = c1[:,i].copy()
##                    c1 = rng.permuted(c1, axis=0)
##                    c1[:,i] = c_cop
##                    c[i,l,m] += np.abs(c1[l,-1*i] - S_new[m,-1*i])
##                    c_hatl = (np.einsum('lm->l',c[i,:,:])/(nn-2))
##                    c_hatm = (np.einsum('lm->m',c[i,:,:])/(nn-2))
##                    c_hat = np.einsum('lm->', c[i,:,:])/((nn-1)*(nn-2))
##                    C[i,l,m]+= (c[i,l,m] - c_hatl - c_hatm + c_hat)
                



##
##
###===============Model Covariance Matrix===============
##
##X = X_new.copy()
##x_hat = np.array([np.mean(X_new[:,i]) for i in range(3)])
##y_hat = np.mean(val[2])
##nx_i, n_feats = X_new.shape
##cov = np.zeros((3,3))
##for i in range(n_feats):
##    for j in range(n_feats):
##        for k in range(nx_i):
##            cov[i,j] +=(X[k,j] - x_hat[j])*(X[k,i] - x_hat[i])
##
##        cov[i,j] /=nx_i-1
##
##
##
##
###================RF's Feature Importances===============
##
def gcd(y: int)->int:
    gcf = 0
    for i in it.count(2):
        if y%i!=0:
            continue
        else:
            gcf = i
            break
    return gcf

#Original parameter data: i.e. the whole dataset
# fv= col.defaultdict(list)

# for (i,tree) in enumerate(RF.estimators_):
#     n_nodes = tree.tree_.node_count
#     children_left = tree.tree_.children_left
#     children_right = tree.tree_.children_right
#     feature = tree.tree_.feature
#     threshold = tree.tree_.threshold
#     impurity = tree.tree_.impurity
#     for j in it.chain(range(1)):
#         node_indicator, *_ = tree.tree_.decision_path(val[0].copy())
#         leaf_id = tree.tree_.apply(val[0].copy())
#         node_index = node_indicator.indices[node_indicator.indptr[j]:node_indicator.indptr[j+1]]
#         decisions = col.deque()
#         for node_id in node_index:
#             if leaf_id[j] == node_id:
#                 continue

#             if val[0][j, feature[node_id]] <= threshold[node_id]:
#                 threshold_sign = '<='

#             else:
#                 threshold_sign = '>'

#             decisions.append(feature[node_id])
#     fv[i].extend(decisions)
#Split Original Parameter data into two separate arrays: A and B
indx = rng.choice(len(val[0]), size=len(val[0]), replace=False)
A = val[0][indx,:]
B = val[0][-indx,:]
def decpath(data, estimator) -> list:
    sep_dat = data
    dec = col.defaultdict(list)
    for (i,tree) in enumerate(estimator):
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        impurity = tree.tree_.impurity
        for j in it.chain(range(1)):
            node_indicator, *_ = tree.tree_.decision_path(sep_dat.copy())
            leaf_id = tree.tree_.apply(sep_dat.copy())
            node_index = node_indicator.indices[node_indicator.indptr[j]:node_indicator.indptr[j+1]]
            decisions = col.deque()
            for node_id in node_index:
                if leaf_id[j] == node_id:
                    continue

                if sep_dat[j, feature[node_id]] <= threshold[node_id]:
                    threshold_sign = '<='

                else:
                    threshold_sign = '>'

                decisions.append(feature[node_id])

        dec[i].extend(decisions)
    return dec


def decision_paths(k, estimators)->list:
    n_estimators = len(estimators)
    F =col.defaultdict(list)

    for i in it.chain(range(k.shape[1])):
        X = k.copy()
        x = X[:,i].copy()
        X = rng.permuted(X, axis=0)
        X[:,i] = x
        for (j, tree) in enumerate(estimators):
            feature = tree.tree_.feature
            n_nodes = tree.tree_.node_count
            decisions = col.deque() 
            for m in it.chain(range(1)):
                node_indicator, *_ = tree.tree_.decision_path(X.copy())
                leaf_id = tree.tree_.apply(X.copy())
                node_index = node_indicator.indices[node_indicator.indptr[m]:node_indicator.indptr[m+1]]

                for node_id in node_index:
                    if leaf_id[m] == node_id:
                        continue
                    decisions.append(feature[node_id])
            F[i].append(decisions)

    return F
def Jaccard1(A: np.ndarray, B: np.ndarray, n_estimators: int) -> np.ndarray:
    A = A
    B = B
    J = np.zeros(n_estimators)
    for i in it.chain(range(n_estimators)):
        intersect = 0 
        if len(A[i])<len(B[i]):
            for k in it.chain(range(len(A[i]))):
                if A[i][k] == B[i][k]:
                    intersect +=1
                else: 
                    pass
        elif len(A[i])>len(B[i]):
            for k in it.chain(range(len(B[i]))):
                if A[i][k] == B[i][k]:
                    intersect +=1
                else:
                    pass
        elif len(A[i])==len(B[i]):
            for k in it.chain(range(len(A[i]))):
                if A[i][k] == B[i][k]:
                    intersect +=1
                else:
                    pass
        union = len(A[i]) +len(B[i]) - intersect
        if union == 0:
            pass
        else:
            J[i] = intersect/union
    return J
            


# Jaccard for First-Order
def Jaccard(F: list, original: list, n_estimators: int, n_columns: int) -> np.ndarray[float]:
    permed_F = F
    origin  = original 
    J = np.zeros((n_columns, n_estimators))
    for i in it.count():
        if len(permed_F[i]) == 0:
            break
        else:
            for j in it.chain(range(n_estimators)):
                intersect = 0
                if len(permed_F[i][j])< len(origin[j]):
                    for k in it.chain(range(len(permed_F[i][j]))):
                        if permed_F[i][j][k] == origin[j][k]:
                            intersect+=1
                        else:
                            pass
                elif len(permed_F[i][j])>len(origin[j]):
                    for k in it.chain(range(len(origin[j]))):
                        if permed_F[i][j][k] == origin[j][k]:
                            intersect += 1
                        else:
                            pass
                elif len(permed_F[i][j]) == len(origin[j]):
                    for k in it.chain(range(len(origin[j]))):
                        if permed_F[i][j][k] == origin[j][k]:
                            intersect +=1

                        else:
                            pass
                union = len(permed_F[i][j]) +len(origin[j]) - intersect
                J[i,j] = intersect/union
        
    return J

def Jaccard2(F: list, original: list, n_estimators: int, n_columns: int) -> np.ndarray:
    permed_F = F
    origin  = original 
    J = np.zeros((n_columns**2, n_estimators))
    for h,elem in enumerate(it.product(range(n_columns), repeat=2)):
        i,j = elem
        K = permed_F[i][j]
        for k in it.chain(range(n_estimators)):
            intersect = 0
            if len(K[k])<len(origin[k]):
                for l in it.chain(range(len(K[k]))):
                    if K[k][l] == origin[k][l]:
                        intersect+=1
                    else:
                        pass
            elif len(K[k])>len(origin[k]):
                for l in it.chain(range(len(origin[k]))):
                    if K[k][l] == origin[k][l]:
                        intersect+=1
                    else: 
                        pass
            elif len(K[k]) == len(origin[k]):
                for l in it.chain(range(len(K[k]))):
                    if K[k][l] == origin[k][l]:
                        intersect+=1
                    else:
                        pass
            union = len(K[k]) + len(origin[k]) - intersect
            J[h,k] = intersect/union
    return J

def levenshtein_distance(s1: list, s2: list)->int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1] 


#Second-Order Jaccard 
def dec_path2(data, estimators)->list:
    n_estimators = len(estimators)
    F = col.defaultdict(lambda: col.defaultdict(list))
    for (i,j) in it.product(range(data.shape[1]), repeat=2):
        X = data.copy()
        x = X[:,[i,j]].copy()
        X = rng.permuted(X, axis=0)
        X[:,[i,j]] = x
        for (k, tree) in enumerate(estimators):
            feature = tree.tree_.feature
            n_nodes = tree.tree_.node_count
            decisions = col.deque() 
            for m in it.chain(range(1)):
                node_indicator, *_ = tree.tree_.decision_path(X.copy())
                leaf_id = tree.tree_.apply(X.copy())
                node_index = node_indicator.indices[node_indicator.indptr[m]:node_indicator.indptr[m+1]]

                for node_id in node_index:
                    if leaf_id[m] == node_id:
                        continue
                    decisions.append(feature[node_id])
            F[i][j].append(decisions)
    return F





        
#The following code snippet is from Severin Pappadeux: Give credit where credit is due
# def query_neighbors(tree, x, k):
#     return tree.query(x, k = k + 1)[0][:, k]

# def build_tree(points):
#     if points.shape[1] >= 20:
#         return BallTree(points, metric="chebyshev")
#     return KDTree(points, metric="chebyshev")

# def kldiv(x, xp, k=3, base=2) -> float:
#     """KL Divergence between p and q for x~p(x), xp~q(x)
#     x, xp should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
#     if x is a one-dimensional scalar and we have four samples
#     """
#     assert k < min(len(x), len(xp)), "Set k smaller than num. samples - 1"
#     assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
#     x, xp = np.asarray(x), np.asarray(xp)
#     x, xp = x.reshape(x.shape[0], -1), xp.reshape(xp.shape[0], -1)
#     d = len(x[0])
#     n = len(x)
#     m = len(xp)
#     const = np.log(m) - np.log(n - 1)
#     tree = build_tree(x)
#     treep = build_tree(xp)
#     nn = query_neighbors(tree, x, k)
#     nnp = query_neighbors(treep, x, k - 1)
#     kld = (const + d * (np.log(nnp).mean() - np.log(nn).mean())) / np.log(base)
#     return kld
#First-Order Jaccard
# F = []
# A1 = []

# if __name__== '__main__':
#     p1, p2= mp.Pool(), mp.Pool()
#     F.append(p1.starmap(decision_paths,[(A, RF.estimators_)]))
#     A1.append(p2.starmap(decpath,[(A, RF.estimators_)]))
#     p1.close()
#     p2.close()
#     p1.join()
#     p2.join()

    # process = []
    # p1 = Pool()
    # result1 = p1.starmap_async(decision_paths, [(A, RF.estimators_)])
    # p1.close()
    # p1.join()
# p4 = mp.Process(Jaccard1, (A1, B1, n_estimators))
    # with cf.ProcessPoolExecutor() as executor:
    #     f1 = executor.submit(decision_paths, *(A,RF.estimators_))
    #     f2 = executor.submit(decpath,*(A,RF.estimators_))
    #     print(f1.result())
    #     print(f2.result())
        # process.append(f1)
        # process.append(f2)
        # for processes in cf.as_completed(process):
        #     print(processes.result())
    # print(f2.result())


F = decision_paths(A,RF.estimators_)
A1, B1 = decpath(A, RF.estimators_), decpath(B, RF.estimators_) #Obtain Dec__paths for sample A and 
J1 = Jaccard1(A1,B1, n_estimators) # Compute Jaccard Between
J = Jaccard(F,A1, n_estimators, 3)
x0,x1,x2 = J[0],J[1],J[2]

#Non-Parametric Approach: Using Kernel Density Estimation
kernel0 = stat.gaussian_kde(x0)
kernel1 = stat.gaussian_kde(x1)
kernel2 = stat.gaussian_kde(x2)
kernelc = stat.gaussian_kde(J1)

##Find range for each feature 
x0_min, x0_max = x0.min(), x0.max()
x1_min, x1_max = x1.min(), x1.max()
x2_min, x2_max = x2.min(), x2.max()
j1_min, j1_max = J1.min(), J1.max()

# Feature Positions
X0 = np.mgrid[x0_min:x0_max:100j]
X1 = np.mgrid[x1_min:x1_max:100j]
X2 = np.mgrid[x2_min:x2_max:100j]
j1_pos = np.mgrid[j1_min:j1_max:100j]

# Compute Kernel Densities
kden_0 = kernel0(X0)[:,np.newaxis]
kden_1 = kernel1(X1)[:,np.newaxis]
kden_2 = kernel2(X2)[:, np.newaxis]
kden_c = kernelc(j1_pos)[:, np.newaxis]

#Second-Order Jaccard
F1 = dec_path2(A,RF.estimators_ )
J2 = Jaccard2(F1, A1, n_estimators, 3)
Kldiv = np.array([[stat.entropy(J1, J2[0]), stat.entropy(J1, J2[1]), stat.entropy(J1, J2[3])],
                  [stat.entropy(J1, J2[3]), stat.entropy(J1, J2[4]), stat.entropy(J1, J2[5])],
                  [stat.entropy(J1, J2[6]), stat.entropy(J1, J2[7]), stat.entropy(J1, J2[8])]

])
def gaussian_kernel(u):
    return 1/(np.sqrt(2*np.pi))*np.exp(-0.5*pow(u,2))

def kde(x, data, bandwidth):
    n = len(data)
    estimate = 0
    for xi in data:
        estimate += gaussian_kernel((x-xi)/bandwidth)
    return estimate/ (n*bandwidth)
bandwidth = 0.1
bins = np.histogram_bin_edges(J1, bins=9)
def gauss_pdf(x, mu, sig):
    exp = np.exp(-pow(x-mu, 2)/(2*pow(sig,2)))
    return 1/(np.sqrt(2*np.pi)*sig)*exp

def simpsons_rule(func, a, b, n, *args):
    # Ensure n is even
    if n % 2:
        raise ValueError("n must be even!")
    
    h = (b - a) / n
    integral = func(a, *args) + func(b, *args)
    
    for i in range(1, n):
        multiplier = 4 if i % 2 == 1 else 2
        integral += multiplier * func(a + i * h, *args)
    
    return (h / 3) * integral
probs0 = []
probs = col.defaultdict(list)
for bin0 in it.pairwise(bins):
    a,b = bin0
    func = ft.partial(kde,data=J1, bandwidth =bandwidth)
    probs0.append(simpsons_rule(func,a,b,1000))
for i in it.chain(range(9)):
    for bin0 in it.pairwise(bins):

        a,b = bin0
        func = ft.partial(kde,data=J2[i], bandwidth =bandwidth)
        probs[i].append(simpsons_rule(func,a,b,1000))

#Matrix of Relative Entropies 
REL_ENTR = np.array([[stat.entropy(probs0, probs[0]), stat.entropy(probs0, probs[1]), stat.entropy(probs0, probs[2])],
                     [stat.entropy(probs0, probs[3]), stat.entropy(probs0, probs[4]), stat.entropy(probs0, probs[5])],
                     [stat.entropy(probs0, probs[6]), stat.entropy(probs0, probs[7]), stat.entropy(probs0, probs[8])]

                        ])
print(f'The Relative Entropy for 100,000 samples:\n {REL_ENTR}')
print(f'The Kullback-Liebler Divergence for 100,000 samples: \n {Kldiv}')
# kernel00 = stat.gaussian_kde(J2[0])
# kernel01 = stat.gaussian_kde(J2[1])
# kernel02 = stat.gaussian_kde(J2[2])
# kernel10 = stat.gaussian_kde(J2[3])
# kernel11 = stat.gaussian_kde(J2[4])
# kernel12 = stat.gaussian_kde(J2[5])
# kernel20 = stat.gaussian_kde(J2[6])
# kernel21 = stat.gaussian_kde(J2[7])
# kernel22 = stat.gaussian_kde(J2[8])


# #Kernel Ranges
# x00_min, x00_max = J2[0].min(), J2[0].max()
# x01_min, x01_max = J2[1].min(), J2[1].max()
# x02_min, x02_max = J2[2].min(), J2[2].max()
# x10_min, x10_max = J2[3].min(), J2[3].max()
# x11_min, x11_max = J2[4].min(), J2[4].max()
# x12_min, x12_max = J2[5].min(), J2[5].max()
# x20_min, x20_max = J2[6].min(), J2[6].max()
# x21_min, x21_max = J2[7].min(), J2[7].max()
# x22_min, x22_max = J2[8].min(), J2[8].max()

# X00 = np.mgrid[x00_min:x00_max:100j]
# X01 = np.mgrid[x01_min:x01_max:100j]
# X02 = np.mgrid[x02_min:x02_max:100j]
# X10 = np.mgrid[x10_min:x10_max:100j]
# X11 = np.mgrid[x11_min:x11_max:100j]
# X12 = np.mgrid[x12_min:x12_max:100j]
# X20 = np.mgrid[x20_min:x20_max:100j]
# X21 = np.mgrid[x21_min:x21_max:100j]
# X22 = np.mgrid[x22_min:x22_max:100j]

# ##Feature Positions
# kden00 = kernel00(X00)[:,np.newaxis]
# kden01 = kernel01(X01)[:, np.newaxis]
# kden02 = kernel02(X02)[:, np.newaxis]
# kden10 = kernel10(X10)[:,np.newaxis]
# kden11 = kernel11(X11)[:,np.newaxis]
# kden12 = kernel12(X12)[:,np.newaxis]
# kden20 = kernel20(X20)[:,np.newaxis]
# kden21 = kernel21(X21)[:, np.newaxis]
# kden22 = kernel22(X22)[:, np.newaxis]

# def kldiv(x: list, y: list):

# Kl_mat = np.array(
#     [[ss.kl_div(kden_c, kden00).mean(), ss.kl_div(kden_c, kden01).mean(), ss.kl_div(kden_c, kden02).mean()],
#      [ss.kl_div(kden_c, kden10).mean(), ss.kl_div(kden_c, kden11).mean(), ss.kl_div(kden_c, kden12).mean()],
#      [ss.kl_div(kden_c, kden20).mean(), ss.kl_div(kden_c, kden21).mean(), ss.kl_div(kden_c, kden22).mean()]]
    
# ).reshape((3,3))



# data = {'alpha': x0,'Teq': x1, 'S0': x2, 'Control': J1}
# sns.displot(data=data, kind='kde', fill=True)
# plt.show()

#Probability Estimation Using Histograms
# bin_edges0 = np.histogram_bin_edges(x0, bins='auto')
# bin_edges1 = np.histogram_bin_edges(x1, bins='auto')
# bin_edges2 = np.histogram_bin_edges(x2, bins='auto')
# bin_edgesc = np.histogram_bin_edges(J1, bins='auto')
# hist0, bins0 = np.histogram(x0, bins=15, density=True)
# hist1, bins1 = np.histogram(x1, bins=15, density=True)
# hist2, bins2 = np.histogram(x2, bins=15, density=True)
# histc, binsc = np.histogram(J1, bins=15, density=True)
# P = histc*np.diff(binsc)
# Q0 = hist0*np.diff(bins0)
# Q1 = hist1*np.diff(bins1)
# Q2 = hist2*np.diff(bins2)

##important = np.zeros((3,))
##important2 = np.zeros((3,3))
##important3 = np.zeros((3,))

##for tree in RF.estimators_[:10]:
##    start = 0
##    n_nodes = tree.tree_.node_count
##    children_left = rng.permuted(tree.tree_.children_left)
##    children_right = rng.permuted(tree.tree_.children_right)
##    feature = tree.tree_.feature
##    threshold = tree.tree_.threshold
##    impurity = tree.tree_.impurity
##    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
##    is_leaves = np.zeros(shape=n_nodes, dtype=np.int64)
##    node_id = list(np.arange(n_nodes))
##    while len(node_id)>0:
##        nodeid= node_id.pop()
##        is_split_node = children_left[nodeid] != children_right[nodeid]
##        if is_split_node:
##            weight_node_samps = tree.tree_.weighted_n_node_samples
##            w0 = weight_node_samps[nodeid]
##            w1 = weight_node_samps[children_left[nodeid]]
##            w2 = weight_node_samps[children_right[nodeid]]
##            f = feature[nodeid]
##            cl = children_left[nodeid]
##            cr = children_right[nodeid]
##            fcl = feature[cl]
##            fcr = feature[cr]
##            mimp = np.mean(impurity)
##
##            important[feature[nodeid]] += (w0*impurity[nodeid] -
##                                            w1*impurity[cl] -
##                                            w2*impurity[cr])
##            important2[fcl,fcr] += (w0*impurity[nodeid] -
##                                    w1*impurity[cl]/max(impurity) -
##                                    w2*impurity[cr]/max(impurity))/2
##       
##            important3[f] += (impurity[nodeid]/max(impurity))/2
##=====================================================
            
# Partial Dependence: The conditional expectation




#Corrected ICE Values
def ice(mat: list, f: callable, s: int):
    n, m = mat.shape
    cond_xi = np.zeros((n,n))
    data = np.array([], dtype=np.float64).reshape(0,m)
    for i in it.count():
        if i<n:
            x_c = np.array([mat[i,:]]*n)
            x_c[:,s] = mat[:,s]
            data = np.vstack((data, x_c), dtype=np.float64)
            fi_s = np.zeros((n,))
            for (j,k) in enumerate(it.chain(x_c)):
                fi_s[j] = f(k)
            cond_xi[:,i]=fi_s
        else:
            break
            
    return (cond_xi, data)


def ice1(mat: list, f: callable):
    n, m = mat.shape
    samps = np.zeros((m, n, m+1), dtype=np.float64)
    for i in it.count():
        if i<m:
            d = mat.copy()
            dv = mat[:,i].copy()
            d = rng.permuted(d.copy(), axis=0)
            d[:,i] = dv
            samps[i,:,:m] = d
            for k in it.chain(range(n)):
                samps[i,k,m] = f(d[k])
        else:
            break
    samps = samps.reshape((m*n, m+1))
    return samps

# Group Model Validation & Preparation
##val1 = train_test_split(df[:,[0,1]], df[:,3], test_size=0.3, train_size=0.7, shuffle=False, random_state=0)
##val2 = train_test_split(df[:,[0,2]], df[:,3], test_size=0.3, train_size=0.7, shuffle=False, random_state=0)
##val3 = train_test_split(df[:,[1,2]], df[:,3], test_size=0.3, train_size=0.7, shuffle=False, random_state=0)
##mod1 = RandomForestRegressor(n_estimators=n_estimators,random_state=0)
##mod2 = RandomForestRegressor(n_estimators=n_estimators,random_state=0)
##mod3 = RandomForestRegressor(n_estimators=n_estimators,random_state=0)
##
##
### Group Model Fit
##mod1.fit(val1[0],val1[2])
##mod2.fit(val2[0], val2[2])
##mod3.fit(val3[0],val3[2])
##
### Model Group Feature Importances No.0
##x01 = mod1.feature_importances_
##x02 = mod2.feature_importances_
##x12 = mod3.feature_importances_
##
### Model Group Feature Importances No.1: Insert 0 for missing feature
##x01 = np.insert(x01,2,0)
##x02 = np.insert(x02, 1, 0)
##x12 = np.insert(x12, 0, 0)
##tot = x01+x02+x12
##tot1 = tot/3
##
### Group Importances
##x0 = sorted(np.array([x01[0], x02[0]]), reverse=True)
##x1 = sorted(np.array([x01[1], x12[1]]), reverse=True)
##x2 = sorted(np.array([x02[2], x12[2]]), reverse=True)
###... and modified GIs
##
##f0 = np.insert(x0, 2, 0)
##f1 = np.insert(x1, 2, 0)
##f2 = np.insert(x2, 2, 0)
##
### The New Measures No.0
##S10 = np.array([[x1[i] - x0[j] for j in range(2)] for i in range(2)])
##S20 = np.array([[x2[i] - x0[j] for j in range(2)] for i in range(2)])
##S21 = np.array([[x2[i] - x1[j] for j in range(2)] for i in range(2)])
##
### Original FI: Measures No.1
##
##SS10 = np.array([[f1[i] - f0[j] for j in range(3)] for i in range(3)])
##SS20 = np.array([[f2[i] - f0[j] for j in range(3)] for i in range(3)])
##SS21 = np.array([[f2[i] - f1[j] for j in range(3)] for i in range(3)])
##
###Permuted Measures
##SS0 = S10-S20+S21
##SS1 = S20-S21+S10
##SS2 = S21-S10+S20 +1



# Resampling
##samps = ice1(df[:,:3], new_gmsl)

##cond, data = ice(df[:,:3], new_gmsl, 0)


##valk = train_test_split(samps[:,:3],samps[:,3], train_size=0.7, test_size=0.3, random_state=0, shuffle=False)
##mod = RandomForestRegressor(n_estimators=n_estimators,random_state=0)
##mod.fit(valk[0], valk[2])
##v_new = np.array([max(mod.estimators_[i].tree_.impurity) for i in range(100)])
##mv = np.mean(v_new)
##max_imp = np.zeros((3,))
##important = {'0':[],'1':[],'2':[]}
##important1 = np.zeros((3,), dtype=np.float64)
##for tree in mod1.estimators_:
##    n_nodes = tree.tree_.node_count
##    children_left = tree.tree_.children_left
##    children_right = tree.tree_.children_right
##    feature = tree.tree_.feature
##    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
##    is_leaves = np.zeros(shape=n_nodes, dtype=np.int64)
##    stack = [(0,0)]
##    while len(stack)>0:
##        node_id, depth = stack.pop()
##        node_depth[node_id] = depth
##        cl = children_left[node_id]
##        cr = children_right[node_id]
##        is_split_node = cl != cr
##        if is_split_node:
##            stack.append((cl,depth+1))
##            stack.append((cr, depth+1))
##            weight_node_samps = tree.tree_.weighted_n_node_samples
##            w0 = weight_node_samps[node_id]
##            w1 = weight_node_samps[cl]
##            w2 = weight_node_samps[cr]
##            f = feature[node_id]
##            fcl = feature[cl]
##            fcr = feature[cr]
##            impurity = tree.tree_.impurity
##
##            important[feature[node_id]] += (w0*impurity[node_id] -
##                                            w1*impurity[cl] -
##                                            w2*impurity[cr])
##
##for tree in mod3.estimators_:
##    n_nodes = tree.tree_.node_count
##    children_left = tree.tree_.children_left
##    children_right = tree.tree_.children_right
##    feature = tree.tree_.feature
##    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
##    is_leaves = np.zeros(shape=n_nodes, dtype=np.int64)
##    stack = [(0,0)]
##    while len(stack)>0:
##        node_id, depth = stack.pop()
##        node_depth[node_id] = depth
##        cl = children_left[node_id]
##        cr = children_right[node_id]
##        is_split_node = cl != cr
##        if is_split_node:
##            stack.append((cl,depth+1))
##            stack.append((cr, depth+1))
##            weight_node_samps = tree.tree_.weighted_n_node_samples
##            w0 = weight_node_samps[node_id]
##            w1 = weight_node_samps[cl]
##            w2 = weight_node_samps[cr]
##            f = feature[node_id]
##            fcl = feature[cl]
##            fcr = feature[cr]
##            impurity = tree.tree_.impurity
##            if f ==0:
##                important[feature[node_id]+1] += (w0*impurity[node_id] -
##                                            w1*impurity[cl] -
##                                            w2*impurity[cr])
##            elif f==1:
##                important[feature[node_id]+1] += (w0*impurity[node_id] -
##                                            w1*impurity[cl] -
##                                            w2*impurity[cr])
##            else:
##                pass
##
##for tree in mod2.estimators_:
##    n_nodes = tree.tree_.node_count
##    children_left = tree.tree_.children_left
##    children_right = tree.tree_.children_right
##    feature = tree.tree_.feature
##    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
##    is_leaves = np.zeros(shape=n_nodes, dtype=np.int64)
##    stack = [(0,0)]
##    while len(stack)>0:
##        node_id, depth = stack.pop()
##        node_depth[node_id] = depth
##        cl = children_left[node_id]
##        cr = children_right[node_id]
##        is_split_node = cl != cr
##        if is_split_node:
##            stack.append((cl,depth+1))
##            stack.append((cr, depth+1))
##            weight_node_samps = tree.tree_.weighted_n_node_samples
##            w0 = weight_node_samps[node_id]
##            w1 = weight_node_samps[cl]
##            w2 = weight_node_samps[cr]
##            f = feature[node_id]
##            fcl = feature[cl]
##            fcr = feature[cr]
##            impurity = tree.tree_.impurity
##            if f ==0:
##                important[feature[node_id]] += (w0*impurity[node_id] -
##                                            w1*impurity[cl] -
##                                            w2*impurity[cr])
##            elif f==1:
##                important[feature[node_id]] += (w0*impurity[node_id] -
##                                            w1*impurity[cl] -
##                                            w2*impurity[cr])
##            else:
##                pass               
##


            
##ind_plots = rng.choice(val[1].shape[0], size=6, replace=False)
##fig, ax = plt.subplots(1,1, figsize=(10,10))
##
##ax.scatter(np.array([val[1][:,1]]*6).T, cond_xi1[:,ind_plots],10,marker='_', color='dimgrey', label=r'$\hat{f}_{\mid X_{\sim 2}}$')
##
##
##ax.set_title(r'Model-Output Conditioned on $T_{eq}$')
##ax.set_ylabel(r'$\hat{f}_{\mid X_{\sim 2}}$')
##ax.set_xlabel(r'$T_{eq}$')
##plt.legend()
##plt.show()
def pd(mat: list, f: callable):
    gg = np.zeros_like(mat)
    n, m = mat.shape
    samps = np.zeros((m, n, m))
    for i in it.chain(range(m)):
        for k in it.chain(range(n)):
            d = mat.copy()
            dv = d[k,i]
            d = rng.permuted(d, axis=0)
            d[k,i] = dv
            samps[i,k, :] = d[-1]
            gg[k,i]+=np.mean(f(d))
    return (gg, samps)

