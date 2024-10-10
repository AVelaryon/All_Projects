import numpy as np
import pandas as pd
import scipy.stats.qmc as ssq
import itertools as it
import matplotlib.pyplot as plt



class Sobol:
    '''
    An attempt at computing Sobol SA measures. The user must supply, during instantiation,
    a callable (func) and a dictionary, comprising of the following keyword arguments:
    {'modvar': list, 'numvar': int, 'num_samp': int, 'arg_vals', 'lowb': list, 'upb': list,
    'model_type': str,'n_boot': int}.
    ***The use of model_type requires more development: an error arises for 'Differential'
    models**
    
    '''
    def __init__(self, model: callable, definition: dict):
        self.model = model
        self.defn = definition

    @staticmethod 
    def bootstrapp(ma: list, m_BA: list, mean: float,n_boots: int) -> np.ndarray[float]:
        rows = m_BA.shape[0]
        mult = ma*m_BA
        VI = np.zeros((n_boots,))
        V0 = np.zeros((n_boots,))
        for i in range(n_boots):
            indx = np.random.choice(rows, size=rows, replace=True)
            VI[i] = np.mean(mult[indx]) - pow(mean,2)
            V0[i] = np.mean(np.square(ma)[indx]) - pow(mean,2)
        return VI/V0
    
    @staticmethod
    def param_sampling(dims: int, nsamp: int) -> list:
        '''
        Parameter Sampling from Latin Hypercube, where (d) denotes dimensions and
        (n) the number of sample points. The sampling generates a random sample between [0,1)
        '''
        engine = ssq.LatinHypercube(d = dims, seed = 0)
        params = engine.random(n = 2*nsamp)
        return params
    
    @classmethod
    def parameter_scaling(cls, lowb: list, upb: list, nsamp: int) -> list:
        '''
        Rescaling the parameter samples using dictionary defined (lowb) and (upb) bounds
        of parameter space.
        '''
        lb, ub = lowb, upb
##        for i in it.chain(range(len(lowb))):
##            prm_scaled[:,i] = prm_scaled[:,i]*(ub[i] - lb[i]) + lb[i]
        prm_scaled = ssq.scale(cls.param_sampling(len(lowb), nsamp), lb, ub)
        return prm_scaled
    
    @classmethod
    def subdiv(cls, lowb: list, upb: list,nsamp: int) -> tuple:
        '''
        Splits sample into two sets: A & B. Generate a random sample of  indices within range (n)*(d),
        where (n) denotes num-of-sample points and (d) denotes num-of-vars, of size=(n) without replacement.
        '''
        numsp = nsamp
        scaled_samp = cls.parameter_scaling(lowb, upb, nsamp)
        rng = np.random.default_rng(0)
        ind = rng.choice(2*numsp, size=(numsp,), replace=False)
        pA = scaled_samp[ind,:]
        pB = scaled_samp[-ind,:]
        return pA, pB
    
    @classmethod
    def mrun(cls,funct: callable, lowb: list, upb: list, nsamp: int) -> dict:
        '''
        Run the supplied model with the scaled parameter samples and return the statistics of model
        run.
        '''
     
        sA, sB = cls.subdiv(lowb, upb, nsamp)
        mA = funct(sA)
        mB = funct(sB)
        meanA = np.mean(mA)
        meanB = np.mean(mB)
        varA = np.var(mA)
        varB = np.var(mB)
        
            
        mrn = {'meanA': meanA, 'meanB': meanB, 'varA': varA, 'varB': varB, 'mA': mA, 'mB': mB}
        return mrn
    
    def fom(self):
        '''
        First-Order Sensitivity Measures 
        '''
        lb ,ub, nsamp,modvar = self.defn['lowb'], self.defn['upb'], self.defn['num_samp'], self.defn['modvar']
        sA , sB = self.subdiv(lb, ub, nsamp)
        # print(f'Shape of sA:{sA.shape}')
        info = self.mrun(self.model, lb, ub, nsamp)
        info_def = self.defn
        meana = info['meanA']
        ma = info['mA']
        d = len(lb)
        Vi = np.zeros((d,), dtype=np.double)
        for i in range(d):
            p_BA = sB.copy()
            p_BA[:,i] = sA[:,i]
            m_BA = self.model(p_BA) 
            Vi[i] = np.mean(ma*m_BA, dtype=np.double) - pow(meana,2)

        Si = pd.DataFrame(Vi/info['varA'],index=modvar, columns=[r'$S_{i}$'])
        return Si
    
    def som(self, S):
        lb, ub, nsamp= self.defn['lowb'], self.defn['upb'], self.defn['num_samp']
        sA, sB = self.subdiv(lb, ub, nsamp)
        info = self.mrun(self.model, lb, ub, nsamp)
        S = S
        d = len(lb)
        meana = info['meanA']
        vara = info['varA']
        modvar = self.defn['modvar']
        ifodefn = self.defn
        ma = info['mA']
        Sik = dict()
        for i, k in it.combinations(range(d),2):
            AB = sB.copy()
            AB[:,[i,k]] = sA[:,[i,k]]
            m_AB = self.model(AB)
            Sik[(modvar[i],modvar[k])] = (np.mean(ma*m_AB) - pow(meana,2))/vara - S[i] -S[k]
            
        Sik = pd.DataFrame(Sik.values(), index = Sik.keys(), columns=[r'$S_{ik}$'])
        
        return Sik

    def totm(self):
        lb, ub, nsamp= self.defn['lowb'], self.defn['upb'], self.defn['num_samp']
        sA, sB = self.subdiv(lb, ub, nsamp)
        info = self.mrun(self.model, lb, ub, nsamp)
        idef = self.defn
        meana = info['meanA']
        d = len(lb)
        V_i = np.zeros((d,))
        for i in range(d):
            p_AB = sA.copy()
            p_AB[:,i] = sB[:,i]
            m_AB = np.asarray([self.model(p_AB[j,:]) for j in range(nsamp)])
            V_i[i] = np.mean(info['mA']*m_AB) - pow(meana,2)

        T = 1-(V_i/info['varA'])
        return pd.DataFrame(T, index=idef['modvar'], columns=['S_T'])

    def confidef(self):
        lb, ub, nsamp,modvar= self.defn['lowb'], self.defn['upb'], self.defn['num_samp'], self.defn['modvar']
        d = len(lb)
        n = nsamp
        sA, sB = self.subdiv(lb, ub, nsamp)
        info = self.mrun(self.model, lb, ub, nsamp)
        Vi = np.zeros((d,), dtype=np.double)
        ma = info['mA']
        meana = info['meanA']
        n_boot = self.defn['n_boot']
        CI_S = dict()
        for i in range(d):
            p_BA = sB.copy()
            p_BA[:,i]= sA[:,i]
            m_BA = self.model(p_BA)
            Vi[i] = np.mean(ma*m_BA, dtype=np.double) - pow(meana,2)/info['varA']
            S_CI = self.bootstrapp(ma, m_BA,meana, n_boot)
            p0,p1 = np.quantile(S_CI, [0.05,0.95])
            CI_S[modvar[i]] = [p0,p1,p1-p0]
        CI = pd.DataFrame(CI_S.values(), index=CI_S.keys(), columns=['5th','95th','Quantile Difference'])
        Si = pd.DataFrame(Vi,index=modvar, columns=[r'$S_{i}$'])
        # print(Si, CI, sep='\n\n')
        return Si, CI

    def confides(self, SI):
        lb, ub, nsamp = self.defn['lowb'], self.defn['upb'], self.defn['num_samp']
        d = len(lb)
        sA, sB = self.subdiv(lb, ub, nsamp)
        info = self.mrun(self.model, lb, ub, nsamp)
        S = SI
        ma = info['mA']
        vara = info['varA']
        meana = info['meanA']
        n_boots = self.defn['n_boot']
        modvar = self.defn['modvar']
        CI_S = dict()
        Sik = dict()
        for i, k in it.combinations(range(d),2):
            BA = sB.copy()
            BA[:,[i,k]] = sA[:,[i,k]]
            MBA = self.model(BA)
            Sik[(modvar[i],modvar[k])] = (np.mean(ma*MBA) - pow(meana,2))/vara - S[i] -S[k]
            S_CI = self.bootstrapp(ma, MBA,meana, n_boots)
            p0,p1 = np.quantile(S_CI, [0.05, 0.95])
            CI_S[(modvar[i],modvar[k])] = [p0,p1,p1-p0]

        CI = pd.DataFrame(CI_S.values(), index = CI_S.keys(), columns=['5th','95th','Quantile Difference'])
        CI.index.name = 'Interactions'
        CI.columns.name = 'Confidence Interval Difference'
        return Sik, CI
