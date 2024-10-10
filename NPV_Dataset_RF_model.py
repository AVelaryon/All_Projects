import numpy as np
# from sklearn import tree
import itertools as it
from sklearn.metrics import mean_squared_error as MSE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
import scipy.stats.qmc as ssq
from sklearn.model_selection import train_test_split, GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import pandas as pd 
import collections as col
import matplotlib as mpl
# from random import choices



pd.set_option('display.max_columns', 10000)
pd.set_option('display.precision',5)
pd.set_option('display.max_rows', 10000)

#NPV Dataset (rows are different gulf coast segments, columns are different ensemble members)
NPV = pd.read_excel(r"C:\Users\dwigh\Downloads\NPVOptimal_Gulf_Sneasy_245.xlsx")
# print(NPV)
dNPV = NPV.sum(axis=0)
# print(dNPV)
dNPV.drop(index='Unnamed: 0', inplace=True)
dNPV.rename('NPV_Dataset', inplace=True)
# rand = pd.DataFrame(np.random.randn(10000, 1),index=dNPV.index, columns=['Signal_VAR'])
# print(dNPV)
# print(f'Minimum:{dNPV.min()}\n')
# print(f'Maximum:{dNPV.max()}')
# dNPV = dNPV.apply(normalize)
#Feature Dataset 
# THe removed features,'sd_temp','sd_ocean_heat','sd_glaciers','sd_greenland','sd_gmsl','sigma_whitenoise_co2','rho_temperature','rho_ocean_heat','rho_glaciers','alpha0_CO2',
#                  'CO2_0','N2O_0','thermal_s0','greenland_v0','glaciers_v0','glaciers_s0','antarctic_s0','Q10','CO2_fertilization','CO2_diffusivity','greenland_a',
#                  'greenland_alpha','anto_beta','antarctic_gamma','antarctic_alpha','glaciers_n','antarctic_kappa','antarctic_nu','antarctic_flow0','antarctic_c','antarctic_bed_height0',
                #  'antarctic_slope','wvel_s','vslel_s','vslmult_s','sd_antarctic','rho_antarctic','temperature_0', 'anto_alpha', 'antarctic_runoff_height0','rho_gmsl','greenland_b','antarctic_mu','glaciers_beta0']
# print('heat_diffusivity and thermal_alpha was added. Antarctic precip0, rho_greenland, wvpdl_s and maybe a few others should be removed. Moreover,before you do anything perform BayesCV again. ')
# It should be noted that the model\'s output denotes money. The question: does the dataset's climate features have a relationship with Human response (i.e., moneys alloted in response to sea-level rise.)? 
#  Although we have a model that has nonlinear features, its relationship to the model's output might--the model itself-- is linear. Random Forest may have interpret this fact. A simple proof of this would be loading the
# features and appending a y-intercept vector and, then, perform gradient descent.
FEAT  = pd.read_excel(r"C:\Users\dwigh\Downloads\features_Sneasy_245.xlsx")
FEAT.drop(columns=['Unnamed: 0'], inplace=True)
FEAT.set_index(dNPV.index, drop=False, inplace=True)
#Combining Sets
NPV_FEAT = FEAT.join(dNPV, how='left')
# NPV_FEAT = NPV_FEAT.join(dNPV, how='left')
# print(NPV_FEAT)

npv_columns = FEAT.columns

# Features/ Output
X, Y = NPV_FEAT.iloc[:,:-1], NPV_FEAT.iloc[:, -1]
X = X[['sd_antarctic', 'rho_greenland', 'rho_gmsl', 'temperature_0',
       'ocean_heat_0', 'Q10', 'heat_diffusivity', 'rf_scale_aerosol',
       'climate_sensitivity', 'thermal_alpha', 'greenland_b',
       'greenland_alpha', 'greenland_beta', 'anto_alpha', 'antarctic_mu',
       'antarctic_precip0', 'antarctic_lambda', 'antarctic_temp_threshold',
       'dvbm_s', 'movefactor_s', 'wvpdl_s']]
#===Desired Features X[['sd_antarctic', 'rho_greenland', 'rho_gmsl', 'temperature_0',
    #    'ocean_heat_0', 'Q10', 'heat_diffusivity', 'rf_scale_aerosol',
    #    'climate_sensitivity', 'thermal_alpha', 'greenland_b',
    #    'greenland_alpha', 'greenland_beta', 'anto_alpha', 'antarctic_mu',
    #    'antarctic_precip0', 'antarctic_runoff_height0', 'antarctic_lambda',
    #    'antarctic_temp_threshold', 'dvbm_s', 'movefactor_s', 'wvpdl_s']]
#Performing Split
val = train_test_split(X, Y, train_size=0.8,random_state=0)
val[0] = val[0].astype('float32')

# val2 = train_test_split(gami[:, :-1], gami[:, -1], train_size=0.8, random_state=0)
# val2[0] = val2[0].astype('float32')
# print(X.iloc[:,  [4, 10, 12, 16, 17, 23, 26, 27, 28, 29, 31, 33, 36, 40, 42, 49, 50,51, 52, 56]].columns)
# stand_val = train_test_split(standard, Y, train_size=0.8, random_state=0)
# robust_val = train_test_split(robust, Y, train_size=0.8, random_state=0)
# quantile_val = train_test_split(quantile, Y, train_size=0.8, random_state=0)

# The NPV Model
# RF = RandomForestRegressor(n_estimators=284, max_features=16,max_depth=20, n_jobs=10,random_state=0)
#  The Rahmstorf Model
RF1 = RandomForestRegressor(n_estimators=284,max_features=16,max_depth=20, random_state=0, n_jobs=5)
RF1.fit(val[0], val[2])
#  The Ishigami Model
# RF2 = RandomForestRegressor(n_estimators=500, max_features=2, max_depth=33, random_state=0, n_jobs=20)
# RF2.fit(val2[0],val2[2])

#Data Inspection: KDE plots
# for column in npv_columns:
#     NPV_FEAT.plot(y=column,kind='kde', )
# plt.show()


#Model Fitting 
# RF.fit(val[0], val[2])
# print(RF.feature_importances_)
# print(f'Train Pearson\'s R^2 :\t {RF.score(val[0], val[2])}\n')
# print(f'Test Pearson\'s R^2 :\t {RF.score(val[1], val[3])}\n')
# print(f'Model Train MSE:\t{np.square(val[2]-RF.predict(val[0])).mean()}\n')
# print(f'Model Test MSE: \t{np.square(val[3]-RF.predict(val[1])).mean()}')
# Best Model Parameters:{'max_depth': 32, 'max_features': 50, 'min_samples_split': 2, 'n_estimators': 400}





# grid = GridSearchCV(RF, param_grid=parameter6,scoring='neg_mean_squared_error', cv=4,n_jobs=15)
# grid.fit(val[0], val[2])
# print(f'Best Model Parameters:{grid.best_params_}\n')
# print(f'Best Val MAD:{grid.best_score_}')

# bae = BayesSearchCV(estimator=RF, search_spaces=parameter3,scoring='r2', n_iter=50,n_points=5, cv=4, n_jobs=20, random_state=0)
# bae.fit(val[0], val[2])
# res = pd.DataFrame(bae.cv_results_)
# print(f'Best Model Parameters:{bae.best_params_}\n')
# print(f'Best Val MAD:{bae.best_score_}')
# res.to_excel(r"C:\Users\dwigh\OneDrive\Desktop\NPV_Test\NPV_R_1.xlsx")# Change to NPV18

# res = pd.read_excel(r"C:\Users\dwigh\OneDrive\Desktop\NPVHyperparameterTuning\NPV_MSE_15.xlsx",
#                      usecols=['param_max_depth', 'param_max_features','param_n_estimators', 'mean_test_score', 'rank_test_score', 'std_test_score' ])
# res.set_index('rank_test_score',drop=True, inplace=True)
# res.sort_index(axis=0, inplace=True)
# print(res.iloc[:,:11].to_latex())
# fig, ax = plt.subplots(figsize=(9,9))

# scores = res['mean_test_score']
# grid1 = res['param_max_depth']
# grid2 = res['param_max_features']
# grid3 = res['param_n_estimators']
# scores = np.array(scores).reshape((len(grid3),len(grid2)))
# for indx, val in enumerate(grid3.to_numpy()):
#     ax.plot(grid2, scores[indx,:], '-o')
# plt.show()
# #  03/11/24: Last Test
# Best Model Parameters:{'max_depth': 31, 'max_features': 40, 'min_samples_split': 2, 'n_estimators': 400}
# Best Val MAD:-9.935248573990325
# Test Score: 0.9099107906440629

# 03/16/24
# max_depth=31, max_features=40, n_estimators=500
# MAD:-9.92707

# 03/16/24
# max_depth=29, max_features=45, n_estimators=550
# MAD: -9.916356
# n_estimators=410, max_depth=32, max_features=50
# Test MSE: 156.80013794700267
# 03/18/24
# n_estimators=425, max_features=39
# Val MSE=169.173677
# np.set_printoptions(precision=3)
rng = np.random.default_rng(seed=0)
def bootstrapp(y_pred, data,n_boots):
    rows = data.shape[0]
    error = (y_pred - RF1.predict(data)).to_numpy()
    S_CI = np.zeros((n_boots,))
    for i in range(n_boots):
        indx = rng.choice(rows, size=rows, replace=True)
        S_CI[i] = np.mean(np.square(error[indx]))
    return S_CI

def pvi1(data: np.ndarray[float], y_pred: np.ndarray[float], f: callable,*, n_boots: int, alpha: float, column_set: dict) -> np.ndarray[float]:
    rows, columns = data.shape
    npv_columns = column_set
    V_y = np.var(y_pred)
    u = dict()
    # S_CI = dict()
    # CI = dict()
    for i in range(columns):
        k = data[:,i]
        d = np.roll(data,rows//2,axis=0)
        d[:,i] = k
        u[npv_columns[i]] = 1 - np.square(y_pred - f(d)).mean()/(2*V_y)
    #     S_CI[npv_columns[i]] = 1 - bootstrapp(y_pred,d, n_boots)/(2*V_y)
    #     p0,p1 = np.quantile(S_CI[npv_columns[i]], [alpha,1-alpha])
    #     CI[npv_columns[i]] = [p0,p1,p1-p0]
    # CI = pd.DataFrame(CI.values(), index = CI.keys(), columns=['5th','95th','Quantile Difference'])
    # CI.index.name = 'Interactions'
    # CI.columns.name = 'Confidence Interval Difference'
    return u

def mod_pvi1(data: np.ndarray[float], y_pred: np.ndarray[float], f: callable,*, n_boots: int, alpha: float, column_set: dict) -> np.ndarray[float]:
    rows, columns = data.shape
    npv_columns = column_set
    V_y = np.var(y_pred)
    u = dict()
    # S_CI = dict()
    # CI = dict()
    for (var, lst) in npv_columns.items():
        k = data[lst]
        d = np.roll(data,rows//2,axis=0)
        d[:, data.columns.get_indexer(lst)] = k
        d = pd.DataFrame(d, columns=data.columns)
        u[var] = 1 - np.square(y_pred - f(d)).mean()/(2*V_y)
    #     S_CI[npv_columns[i]] = 1 - bootstrapp(y_pred,d, n_boots)/(2*V_y)
    #     p0,p1 = np.quantile(S_CI[npv_columns[i]], [alpha,1-alpha])
    #     CI[npv_columns[i]] = [p0,p1,p1-p0]
    # CI = pd.DataFrame(CI.values(), index = CI.keys(), columns=['5th','95th','Quantile Difference'])
    # CI.index.name = 'Interactions'
    # CI.columns.name = 'Confidence Interval Difference'
    return u

def pvi_t(data: np.ndarray[float], y_pred: np.ndarray[float], f: callable,*,n_boots: int, alpha: float) -> np.ndarray[float]:
    '''
    ===Total-Order Permutation Variable Importances===

        data: your training dataset
        y_pred: your model output (specifically the prediction of your 'data' arg)
        f: in this case, simply RF.predict
    '''
    rows, columns = data.shape
    npv_columns = data.columns
    u = np.zeros((columns,))
    V_y = np.var(y_pred)
    S_CI = dict()
    CI = dict()
    for i, col in enumerate(npv_columns):
        d = data.copy()
        d[col] = np.roll(d[col],rows//2, axis=0)
        u[i] = np.square(y_pred - f(d)).mean()/(2*V_y)
    #     S_CI[npv_columns[i]] = bootstrapp(y_pred,d, n_boots)/(2*V_y)
    #     p0,p1 = np.quantile(S_CI[npv_columns[i]], [alpha,1-alpha])
    #     CI[npv_columns[i]] = [p0,p1,p1-p0]
    # CI = pd.DataFrame(CI.values(), index = CI.keys(), columns=['5th','95th','Quantile Difference'])
    # CI.index.name = 'Interactions'
    # CI.columns.name = 'Confidence Interval Difference'
    S_t = pd.DataFrame(u[:, np.newaxis], index=npv_columns, columns=[r'$P_{\tau}$'])
    return S_t

def mod_pvi_t(data: np.ndarray[float], y_pred: np.ndarray[float], f: callable,*, n_boots: int, alpha: float, column_set: dict) -> np.ndarray[float]:
    rows, columns = data.shape
    npv_columns = column_set
    V_y = np.var(y_pred)
    u = dict()
    # S_CI = dict()
    # CI = dict()
    for (var, lst) in npv_columns.items():
        d = data.copy()
        d[lst] = np.roll(d[lst], rows//2, axis=0)
        d = pd.DataFrame(d, columns=data.columns)
        u[var] = np.square(y_pred - f(d)).mean()/(2*V_y)
    #     S_CI[npv_columns[i]] = 1 - bootstrapp(y_pred,d, n_boots)/(2*V_y)
    #     p0,p1 = np.quantile(S_CI[npv_columns[i]], [alpha,1-alpha])
    #     CI[npv_columns[i]] = [p0,p1,p1-p0]
    # CI = pd.DataFrame(CI.values(), index = CI.keys(), columns=['5th','95th','Quantile Difference'])
    # CI.index.name = 'Interactions'
    # CI.columns.name = 'Confidence Interval Difference'
    return u
def pvi2(data: np.ndarray[float], y_pred: np.ndarray[float], f: callable,*,S_I: np.ndarray[float] ,n_boots: int, alpha: float, columns_set: dict) -> tuple[np.ndarray[float], list]:
    '''
    ===Second-Order Permutation Variable Importances (Full Matrix)===

        data: your training dataset
        y_pred: your model output (specifically the y_true value)
        f: in this case, simply RF.predict

        S_I: First-Order Importances
        n_boots: number of bootstrap samples
        alpha: parameter for confidence interval [alpha, 1-alpha]
        S_CI: sampling distribution 
        CI: Confidence Intervals
    '''
    rows, columns = data.shape
    npv_columns = columns_set
    n = n_boots
    CI = dict()
    S = dict()
    V_y = np.var(y_pred)
    S_CI = dict()
    # new_index = [item for item in it.combinations(npv_columns,2)]
    # for i,j in it.combinations(range(columns),2):
    for i,j in it.product(range(columns), range(columns)):
        k = data[:,[i,j]]
        d = np.roll(data,rows//2, axis=0)
        d[:, [i,j]] = k
        S[(npv_columns[i],npv_columns[j])] = 1 - np.mean(np.square(y_pred - f(d)))/(2*V_y) -S_I[npv_columns[i]] - S_I[npv_columns[j]]

    #     S_CI[(npv_columns[i],npv_columns[j])] = 1 - bootstrapp(y_pred,d, n_boots)/(2*V_y) -S_I[i] - S_I[j]
    #     p0,p1= np.quantile(S_CI[(npv_columns[i],npv_columns[j])], [alpha,1-alpha])
    #     CI[(npv_columns[i],npv_columns[j])] = [p0,p1,p1-p0]
    # CI = pd.DataFrame(CI.values(), index = CI.keys(), columns=['5th','95th','Quantile Difference'])
    # CI.index.name = 'Interactions'
    # CI.columns.name = 'Confidence Interval Difference'
    return S

def mod_pvi2(data: np.ndarray[float], y_pred: np.ndarray[float], f: callable,*,S_I: np.ndarray[float] ,n_boots: int, alpha: float, columns_set: dict) -> tuple[np.ndarray[float], list]:
    '''
    ===Second-Order Permutation Variable Importances (Full Matrix)===

        data: your training dataset
        y_pred: your model output (specifically the y_true value)
        f: in this case, simply RF.predict

        S_I: First-Order Importances
        n_boots: number of bootstrap samples
        alpha: parameter for confidence interval [alpha, 1-alpha]
        S_CI: sampling distribution 
        CI: Confidence Intervals
    '''
    rows, columns = data.shape
    npv_columns = columns_set
    n = n_boots
    CI = dict()
    S = dict()
    V_y = np.var(y_pred)
    S_CI = dict()
    # new_index = [item for item in it.combinations(npv_columns,2)]
    # for i,j in it.combinations(range(columns),2):
    for (var1,lst1),(var2, lst2) in it.combinations(npv_columns.items(), r=2):
        k = data[lst1+lst2]
        d = np.roll(data,rows//2, axis=0)
        d[:, data.columns.get_indexer(lst1+lst2)] = k
        d = pd.DataFrame(d, columns=data.columns)
        S[(var1, var2)] = 1 - np.mean(np.square(y_pred - f(d)))/(2*V_y) -S_I[var1] - S_I[var2]


        S_CI[(var1,var2)] = 1 - bootstrapp(y_pred,d, n_boots)/(2*V_y) -S_I[var1] - S_I[var2]
        p0,p1= np.quantile(S_CI[(var1,var2)], [alpha,1-alpha])
        CI[(var1,var2)] = [p0,p1,p1-p0]
    CI = pd.DataFrame(CI.values(), index = CI.keys(), columns=['5th','95th','Quantile Difference'])
    CI.index.name = 'Interactions'
    CI.columns.name = 'Confidence Interval Difference'
    return S,CI
#========================================================================================================================#
#============================THIS SECTION CONCERNS SOBOL-PVI FOR ISHIGAMI AND RAHMSTORF MODELS===========================#
#================================================MAY HAVE TO IMPORT SOBOL================================================#
# definition = {'modvar': [r'$\alpha$',r'$T_{eq}$',r'$S_{\circ}$'], 'numvar': 3, 'num_samp': 60000, 'arg_vals':None,'lowb': lb0, 'upb': ub0,'n_boot': 1000}
# Sob_Rahm = Sobol(new_gmsl, definition)
# SRF = Sob_Rahm.fom().to_dict()['$S_{i}$']
# SRS = Sob_Rahm.som().to_dict()['$S_{ik}$']
# SOB_label = SRF.keys() | SRS.keys()
# print(SRS[SRS.values()>0])
# SRS_labels = SRS.index
# rahm_columnset= ['alpha', 'Teq', 'S0']
# ishi_columnset = ['x1','x2','x3']
# Si = pvi1(val1[0],val1[2], RF1.predict, n_boots=1000, alpha=0.05, column_set=rahm_columnset)

# Sik = pvi2(val1[0], val1[2], RF1.predict, n_boots=1000,S_I=Si, alpha=0.05, columns_set=rahm_columnset)
# print(Sik)
# definition = {'modvar': [r'$\alpha$',r'$T_{eq}$',r'$S_{\circ}$'], 'numvar': 3, 'num_samp': 60000, 'arg_vals':None,'lowb': lb0, 'upb': ub0,'n_boot': 1000}
# labels = [r'$\alpha$', r'$T_{eq}$', r'$S_{\circ}$', r'($\alpha$, $T_{eq}$)']
# labels = [r'$x_{1}$', r'$x_{2}$',r'($x_{1}$,$x_{2}$)', r'($x_{1}$, $x_{3}$)', r'($x_{2}$, $x_{3}$)']


# Variables = ['sd_temp', 'sd_ocean_heat', 'sd_glaciers', 'sd_greenland','sd_antarctic',
#               'sd_gmsl', 'sigma_whitenoise_co2', 'rho_temperature','rho_ocean_heat', 
#               'rho_glaciers', 'rho_greenland', 'rho_antarctic','rho_gmsl', 'alpha0_CO2']
# Climate = ['CO2_0', 'N2O_0', 'temperature_0','ocean_heat_0','Q10', 'CO2_fertilization',
#            'CO2_diffusivity', 'heat_diffusivity', 'rf_scale_aerosol','climate_sensitivity']
# Glaciers = ['glaciers_v0','glaciers_s0','glaciers_beta0', 'glaciers_n']
# Thermals = ['thermal_s0', 'thermal_alpha']
# Cost = ['dvbm_s', 'movefactor_s', 'vslel_s','vslmult_s', 'wvel_s', 'wvpdl_s']
# Greenland = ['greenland_a', 'greenland_b','greenland_alpha', 'greenland_beta']
# Antarctic = [ 'antarctic_gamma', 'antarctic_alpha','antarctic_mu', 'antarctic_nu',
#               'antarctic_precip0', 'antarctic_kappa','antarctic_flow0', 'antarctic_runoff_height0',
#                 'antarctic_c','antarctic_bed_height0', 'antarctic_slope', 'antarctic_lambda','antarctic_temp_threshold',
#                 'antarctic_s0','anto_alpha', 'anto_beta']
# Modified Categories
Variables = ['sd_antarctic', 
               'rho_greenland','rho_gmsl']
Climate = ['temperature_0','ocean_heat_0','Q10', 'heat_diffusivity', 'rf_scale_aerosol','climate_sensitivity']
# Glaciers = ['glaciers_v0','glaciers_s0','glaciers_beta0', 'glaciers_n']
Thermals = ['thermal_alpha']
Cost = ['dvbm_s', 'movefactor_s', 'wvpdl_s']
Greenland = [ 'greenland_b','greenland_alpha', 'greenland_beta']
Antarctic = ['antarctic_mu',
              'antarctic_precip0', 'antarctic_lambda','antarctic_temp_threshold','anto_alpha',]
npv_column_set = dict()
npv_column_set['Variables'] = Variables
npv_column_set['Climate'] = Climate
# npv_column_set['Glaciers'] = Glaciers
npv_column_set['Thermals'] = Thermals
npv_column_set['Cost'] = Cost
npv_column_set['Greenland'] = Greenland
npv_column_set['Antarctic'] = Antarctic


PT = mod_pvi_t(val[0], val[2], RF1.predict, n_boots=None, alpha=None, column_set=npv_column_set)
PT = pd.DataFrame(PT.values(), index=PT.keys(), columns=[r'$P_{\tau}$'])
print(PT.to_latex(float_format='{:.4f}'.format))
# print(PT[PT[r'$P_{\tau}$']>=0.0068].index)
PI = mod_pvi1(val[0], val[2], RF1.predict, n_boots=None, alpha=None, column_set=npv_column_set)
PI = pd.DataFrame(PI.values(), index=PI.keys(), columns=[r'$P_{i}$'])
print(PI.to_latex(float_format='{:.4f}'.format))
print(PT)
PIK, CIK =  mod_pvi2(val[0],val[2], RF1.predict, S_I=PI.to_dict()[r'$P_{i}$'], n_boots=1000, alpha=0.05, columns_set=npv_column_set)

# new_PIK = {index: quant  for index,quant in PIK.items()}
new_PIK = pd.DataFrame(PIK.values(), index=PIK.keys(), columns=[r'$P_{ik}$'])

PIK_confide = new_PIK.join(CIK, how='left')

print(PIK_confide.to_latex(float_format='{:.4f}'.format))
# print(PIK_confide.to_latex(float_format='{:.4f}'.format))
# print(f'FIrst-Order PVi:\t {PI}\n\n')
# PIK = mod_pvi2(val[0], val[2], RF1.predict, S_I=PI, n_boots=None, alpha=None, columns_set=npv_column_set)
# print(f'Second-Order PVi:\t {PIK}')
# Si, CI = pvi1(val[0], val[2], RF.predict, n_boots=1000, alpha=0.05)
# Si.sort_values(by=Si.columns[0],axis=0, ascending=False, inplace=True)
# print(f'First-Order:\t{Si.to_latex(float_format='{:.4f}'.format)}\n')
# Si_confide = Si.join(CI, how='left')
# print(Si_confide.to_latex(float_format='{:.4f}'.format))
# S_t, CI_t = pvi_t(val[0], val[2], RF.predict, n_boots=1000, alpha=0.05)
# S_t.sort_values(by=S_t.columns[0],axis=0, ascending=False, inplace=True)
# print(S_t.to_latex(float_format='{:.4f}'.format))
# St_confide = S_t.join(CI_t, how='left')
# print(f'Total-Order:\t{St_confide.to_latex(float_format='{:.4f}'.format)}\n')
# print(f'Total-Order:\t{St_confide.columns}\n')

# Sik, CIS = pvi2(val[0], val[2], RF.predict,S_I=Si,  n_boots=1000, alpha=0.05)
# Sik_confide = Sik.join(CIS, how='left')
# print(Sik.to_latex(float_format='{:.3f}'.format))
# # print(f'Total-Order Indice:\t {(S_t.to_numpy()>=0.0068).nonzero()}')
# indx = (S_t.to_numpy()>=0.0068).nonzero()
# print(f'Columns:\t {S_t.index[indx[0]]}\n')
# print(f'{S_t.iloc[indx[0]]}')
# print(f'{S_t.iloc[indx[0]].to_latex()}')


# ['sd_antarctic', 'rho_greenland', 'rho_gmsl', 'temperature_0',
#        'ocean_heat_0', 'Q10', 'heat_diffusivity', 'rf_scale_aerosol',
#        'climate_sensitivity', 'thermal_alpha', 'greenland_b',
#        'greenland_alpha', 'greenland_beta', 'anto_alpha', 'antarctic_mu',
#        'antarctic_precip0', 'antarctic_runoff_height0', 'antarctic_lambda',
#        'antarctic_temp_threshold', 'dvbm_s', 'movefactor_s', 'wvpdl_s']
# Sik,S_CI = pvi2(val[0], val[2], RF.predict, S_I=Si,n_boots=1000, alpha=0.05)
# S_t_confide = S_t.join(CI_t, how='left')
# print(S_t_confide.to_latex(float_format='{:.3f}'.format))
# print(Si.sum(axis=0), S_t.sum(axis=0), sep='\n')

# Si_variables = Si.loc[Variables].sum(axis=0)
# Si_Climate = Si.loc[Climate].sum(axis=0)
# Si_Glaciers = Si.loc[Glaciers].sum(axis=0)
# Si_Thermals = Si.loc[Thermals].sum(axis=0)
# Si_cost = Si.loc[Cost].sum(axis=0)
# Si_Greenland = Si.loc[Greenland].sum(axis=0)
# Si_Antarctic = Si.loc[Greenland].sum(axis=0)
# new_Si = [np.median(Si.loc[Variables].to_numpy()),np.median(Si.loc[Climate].to_numpy()),np.median(Si.loc[Glaciers].to_numpy()),np.median(Si.loc[Thermals].to_numpy()),np.median(Si.loc[Cost].to_numpy()),np.median(Si.loc[Greenland].to_numpy()),np.median(Si.loc[Antarctic].to_numpy())]
# new_St = [np.median(S_t.loc[Variables].to_numpy()),np.median(S_t.loc[Climate].to_numpy()),np.median(S_t.loc[Glaciers].to_numpy()),np.median(S_t.loc[Thermals].to_numpy()),np.median(S_t.loc[Cost].to_numpy()),np.median(S_t.loc[Greenland].to_numpy()),np.median(S_t.loc[Antarctic].to_numpy())]
# new_Si = pd.DataFrame(np.asarray(new_Si)[:, np.newaxis], index=['Variables','Climate','Glaciers','Thermals','Cost','Greenland','Antarctic'])
# new_St = pd.DataFrame(np.asarray(new_St)[:, np.newaxis], index=['Variables','Climate','Glaciers','Thermals','Cost','Greenland','Antarctic'])
# # new_Si = pd.DataFrame(np.asarray([Si.loc[item].sum(axis=0) for item in [Variables, Climate, Glaciers, Thermals, Cost, Greenland, Antarctic]])[:,np.newaxis],index=['Variables','Climate','Glaciers','Thermals','Cost','Greenland','Antarctic'])
# print(new_Si)
# print(new_Si.to_latex(float_format='{:.4f}'.format))
# print(new_St)
# print(new_St.to_latex(float_format='{:.4f}'.format))
# Ski = dict()
# new_S_CI = dict()
# new_CI = dict()

# for ((varl, varr),(varlb, varrb)) in zipped:
#     l = varl+varr
#     Ski[(varlb, varrb)] = np.median([Sik[item] for item in it.combinations(l,2)])
#     new_S_CI[(varlb, varrb)] = np.median(np.asarray([S_CI[item] for item in it.combinations(l,2)]), axis=0)
#     # print(new_S_CI[(varlb, varrb)])
#     p0,p1 = np.quantile(new_S_CI[(varlb, varrb)], [0.05, 0.95])
#     new_CI[(varlb, varrb)] = [p0,p1,p1-p0]

# Ski = pd.DataFrame(Ski.values(), index=Ski.keys(), columns=['Categorized Second-Order PVi'])
# new_CI = pd.DataFrame(new_CI.values(), index=new_CI.keys(), columns=['5th','95th','Quantile Difference'])
# print(new_Si)
# print(new_Si.to_latex(float_format='{:.4f}'.format))
# print(Ski)
# print(Ski.to_latex(float_format='{:.4f}'.format))
# print(new_CI)
# print(new_CI.to_latex(float_format='{:.4f}'.format))
# Sijk = pvi3(val[0].to_numpy(), val[2].to_numpy(), RF.predict,Si=Si, Sik=Sik)
# print(Sijk)
# print(Sik.to_latex())
# print(CI.to_latex())





# def pvdp(data: np.ndarray,y_true: np.ndarray, feature: list, n_trajectories: int)-> tuple[np.ndarray]:
#     n, m = data.shape
#     output = np.zeros((n_trajectories, n))
#     Vi_samp = np.zeros_like(output)
#     # X = data
#     for i in range(n_trajectories):
#         X = data.copy() 
#         x = X[:, feature]
#         X = rng.permuted(X, axis=0)
#         X[:, feature] = x
#         output[i,:] = RF.predict(X)
#         Vi_samp[i, :] = np.square(y_true - output[i,:])
#     return output,np.mean(output, axis=0), 1-np.mean(Vi_samp, axis=0)/(2*np.var(y_true)), 1-np.std(Vi_samp, axis=0)/(2*np.var(y_true))

# np.set_printoptions(precision=3, threshold=np.inf)
# Si = pvi1(val[0], val[2],RF.predict)
# S_t = pvi_t(val[0], val[2], RF.predict)
# print(f'First-Order PVI:\t {Si}\n')
# print(f'Total-Order PVI:\t {Si.sum(axis=1)}\n')
# # # # print(Si.index)
# # # labels = [r'$x_{1}$', r'$x_{2}$', r'$x_{3}$', r'($x_{1}$, $x_{3}$)']

# Sik= pvi2(val[0],val[2], RF.predict, S_I=Si, n_boots=1000, alpha=0.05)
# print(f'Second-Order PVI:\t {Sik}\n')
# # indice = (np.triu(Sik, k=1).ravel() > 0 ).nonzero()
# importances  = PI
# importances1 = PT
# importances['Variables'] = Si.loc[Variables].sum(axis=0) 
# importances['Climate'] = Si.loc[Climate].sum(axis=0)
# importances['Glaciers'] = Si.loc[Glaciers].sum(axis=0)
# importances['Thermal'] = Si.loc[Thermals].sum(axis=0)
# importances['Cost'] = Si.loc[Cost].sum(axis=0)
# importances['Greenland'] = Si.loc[Greenland].sum(axis=0)
# importances['Antartic Ice Sheet'] = Si.loc[Antarctic].sum(axis=0)

# # importances1['dvdm_s'] = Si.loc['dvbm_s']
# # importances1['movefactor_s'] = Si.loc['movefactor_s']
# # importances1['vslel_s'] = Si.loc['vslel_s']
# # importances1['vslmult_s'] = Si.loc['vslmult_s']
# # importances1['wvel_s'] = Si.loc['wvel_s']
# # importances1['wvpdl_s'] = Si.loc['wvpdl_s']


props = dict()
kw = dict()
props['textprops'] = {'fontsize':14}
props['startangle'] = -270
props['wedgeprops'] = {'width': 0.2}
kw['arrowprops'] = dict(arrowstyle='-')
kw['zorder'] = 0
kw['va'] = 'center'
colors = ['#176BA0','#1AC9E6', '#1DE48D','#6B007B','#B51CE4','#E044A7', '#744EC2']
props['colors'] = colors
# print(importances.values())

#  startangle=-250
# values = np.asarray(list(importances.values())).ravel()
# values1 = np.asarray(list(importances1.values())).ravel()
# labels= importances.keys()
# labels1 = importances1.keys()
# # # , '#744EC2'
# colors = ['#176BA0','#1AC9E6', '#1DE48D','#6B007B','#B51CE4','#E044A7', '#744EC2']
def donut_chart(values: np.ndarray, values1: np.ndarray, labels: list,labels1: list, props: dict, kw: dict):
    fig, ax= plt.subplots(1,2, figsize=(9,8))
    wedges0, text0 = ax[0].pie(values,wedgeprops=props['wedgeprops'],colors=props['colors'], startangle= props['startangle'], textprops = props['textprops'])
    for i, p in enumerate(wedges0):
        ang = (p.theta2-p.theta1)/2 + p.theta1
        x,y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        horizontalalignment = {-1:'right', 1:'left'}[int(np.sign(x))]
        connectionstyle = f'angle, angleA=0, angleB={ang}'
        kw['arrowprops'].update({'connectionstyle':connectionstyle})
        ax[0].annotate(f'{values[i]: .1%}', xy=(x,y), xytext=(1.1*np.sign(x),1.2*y), horizontalalignment=horizontalalignment, **kw)
    ax[0].set_title('BRICK-CIAM: First-Order PVi')
    ax[0].legend(labels=labels,loc='lower left',bbox_to_anchor = (0, -0.3), title='PVi Indices:' )

    wedges1, text1 = ax[1].pie(values1,wedgeprops=props['wedgeprops'],colors=props['colors'], startangle= props['startangle'], textprops = props['textprops'])
    for i, p in enumerate(wedges1):
        ang = (p.theta2-p.theta1)/2 + p.theta1
        x,y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        horizontalalignment = {-1:'right', 1:'left'}[int(np.sign(x))]
        connectionstyle = f'angle, angleA=0, angleB={ang}'
        kw['arrowprops'].update({'connectionstyle':connectionstyle})
        ax[1].annotate(f'{values1[i]: .1%}', xy=(x,y), xytext=(1.1*np.sign(x),1.2*y), horizontalalignment=horizontalalignment, **kw)
    ax[1].set_title('BRICK-CIAM: Total-Order PVi')
    ax[1].legend(labels=labels1,loc='lower left',bbox_to_anchor = (0, -0.3), title='PVi Indices:' )
    plt.show()

# donut_chart(values, values1, labels, labels1, props, kw)
# fig, ax = plt.subplots(figsize=(9,8))
# wedges0, text0 = ax.pie(values.ravel(),wedgeprops=props['wedgeprops'],colors=props['colors'], startangle= props['startangle'], textprops = props['textprops'])
# for i, p in enumerate(wedges0):
#     ang = (p.theta2-p.theta1)/2 + p.theta1
#     x,y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
#     horizontalalignment = {-1:'right', 1:'left'}[int(np.sign(x))]
#     connectionstyle = f'angle, angleA=0, angleB={ang}'
#     kw['arrowprops'].update({'connectionstyle':connectionstyle})
#     ax.annotate(f'{values.ravel()[i]: .1%}', xy=(x,y), xytext=(1.1*np.sign(x),1.2*y), horizontalalignment=horizontalalignment, **kw)
# ax.set_title('BRICK-CIAM: Total-Order PVi')
# ax.legend(labels=labels,loc='upper right',bbox_to_anchor = (1.2,0.9), title='PVi Indices:' )
# donut_chart()
# # # explode = [0.05]*6
# fig, ax = plt.subplots(figsize=(9,8))
# wedges,texts= ax.pie(values.ravel(),wedgeprops=wedgeprops,colors=colors, startangle=-250, textprops = textprops)

# for i, p in enumerate(wedges):
#     ang = (p.theta2-p.theta1)/2 + p.theta1
#     x,y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
#     horizontalalignment = {-1:'right', 1:'left'}[int(np.sign(x))]
#     connectionstyle = f'angle, angleA=0, angleB={ang}'
#     kw['arrowprops'].update({'connectionstyle':connectionstyle})
#     ax.annotate(f'{values.ravel()[i]: .1%}', xy=(x,y), xytext=(1.1*np.sign(x),1.2*y), horizontalalignment=horizontalalignment, **kw)
# ax.set_title('BRICK-CIAM: First-Order PVi')
# ax.legend(labels=labels,loc='upper right',bbox_to_anchor = (1.2,0.9), title='PVi Indices:' )
# plt.show()
# Si = pd.DataFrame(Si, index=npv_columns).to_excel(r"C:\Users\dwigh\OneDrive\Desktop\NPVHyperparameterTuning\NPV_550_SI.xlsx")
# print(S_t.to_latex(float_format='{:.3f}'.format))
# Sik,CI = pvi2(val[0], val[2], RF.predict, S_I=Si,n_boots=1000, alpha=0.05)
# print(f'{Sik.to_latex(float_format='{:.2f}'.format, longtable=True)}')
# print(Sik.to_latex(float_format='{:.2f}'.format), CI.to_latex(float_format='{:.2f}'.format),sep='\n')





# feature selection:['sd_temp', 'sd_greenland', 'rho_greenland',
    #    'rho_gmsl', 'alpha0_CO2',
    #    'glaciers_s0', 'thermal_alpha', 'greenland_a', 'greenland_b',
    #    'greenland_alpha', 'greenland_beta', 'glaciers_beta0', 'glaciers_n',
    #    'anto_alpha', 'anto_beta', 'antarctic_gamma', 'antarctic_alpha']



# Feature Column Names: ['sd_temp', 'sd_ocean_heat', 'sd_glaciers', 'sd_greenland',
    #    'sd_antarctic', 'sd_gmsl', 'sigma_whitenoise_co2', 'rho_temperature',
    #    'rho_ocean_heat', 'rho_glaciers', 'rho_greenland', 'rho_antarctic',
    #    'rho_gmsl', 'alpha0_CO2', 'CO2_0', 'N2O_0', 'temperature_0',
    #    'ocean_heat_0', 'thermal_s0', 'greenland_v0', 'glaciers_v0',
    #    'glaciers_s0', 'antarctic_s0', 'Q10', 'CO2_fertilization',
    #    'CO2_diffusivity', 'heat_diffusivity', 'rf_scale_aerosol',
    #    'climate_sensitivity', 'thermal_alpha', 'greenland_a', 'greenland_b',
    #    'greenland_alpha', 'greenland_beta', 'glaciers_beta0', 'glaciers_n',
    #    'anto_alpha', 'anto_beta', 'antarctic_gamma', 'antarctic_alpha',
    #    'antarctic_mu', 'antarctic_nu', 'antarctic_precip0', 'antarctic_kappa',
    #    'antarctic_flow0', 'antarctic_runoff_height0', 'antarctic_c',
    #    'antarctic_bed_height0', 'antarctic_slope', 'antarctic_lambda',
    #    'antarctic_temp_threshold', 'dvbm_s', 'movefactor_s', 'vslel_s',
    #    'vslmult_s', 'wvel_s', 'wvpdl_s']



  
