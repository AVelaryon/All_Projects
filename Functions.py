import numpy as np
import scipy as sy
from numpy.linalg import *
import matplotlib.pyplot as plt
from itertools import*
import os
import re
class ODESolver:
    """
    **Runge-Kutta 4th Order**
    
    Class takes three arguments: function definition, conditions, and number of equations 
    """
    def  __init__(self, f, conditions, numeqn):
       #Define function
        self.f = f
        self.numeqn = int(numeqn)
        if self.numeqn == 1:
            self.conditions = float(conditions)
        else:
            self.conditions = np.asarray(conditions)
    
    def solve(self, time):
        self.time = np.asarray(time)
        n = len(self.time)
        self.sol = np.zeros((n, self.numeqn))
        self.sol[0,:] = self.conditions
        #Integrate
        sol = np.array([])
        y, f, t = self.sol, self.f,self.time
        for i in range(n-1):
            dt = t[i+1]-t[i]
            dt1 = dt/2
            k1 = np.multiply(dt , f(y[i,:],t[i]))
            k2 = np.multiply(dt, f(y[i,:] + 0.5 * k1, t[i]+dt1))
            k3 = np.multiply(dt, f(y[i,:] + 0.5 * k2, t[i]+dt1))
            k4 = np.multiply(dt, f(y[i,:] + k3, t[i]+dt))
            y[i+1,:] = y[i,:] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        return y

class SimpRule:
    """
    **Simpson's (1/3) Rule Numerical Integration**
    Instantiation takes three arguments: function definition, bound (list of 2 elements), and error bound.
    This class doesn't perform multivariable integration. 
    
    Furthermore, the bounds of integration must be in the form of a list (preferably in increasing order but won't affect anything, if it isn't).
    The calculation terminates when the approximate relative error (error:= abs((int[n]- int[n-1])/int[n]), viz. the integral with [n]-segments minus the integral with [n-1]-segments)
    
    """
    def __init__(self, f, bounds, error):
        self.f = f
        self.b = bounds
        if len(self.b) != 2:
            print('Your input has too many elements for a bound. There should only be 2.')
        else:
            pass
        self.err = error
        
    def solveinteg(self):
        #Step Size
        n = 2
        old = 0
        ii = 1
        errl = 1
        
        print(f'n \t\t  error \t\t integral')
        
        while errl >self.err:
            #Step Size 
            h = (max(self.b)-min(self.b))/n
            sum1, sum2 = 0,0
            for i in range(1,n,2):
                sum1 += 4*self.f(min(self.b)+ i*h)
            for i in range(2, n-1,2):
                sum2 += 2*self.f(min(self.b) + i*h)
                
            integral = (h/3)*(self.f(min(self.b)) + sum1 + sum2 + self.f(max(self.b)))
            errl = np.abs((integral -old)/integral) 
            print(f'{ii} \t\t {errl:.6f} \t\t {integral:.6f}')
            old = integral
            n  += 2
            ii += 1

class EulerMthd:
    ''' Euler's Method.  The clas takes a function, a stepsize, number of eqns, and number of iterations.
    '''
    def __init__(self, f, h,initial, numeqn,  numiter):
        self.f = f
        self.h = h 
        self.initial = initial
        self.neqn = int(numeqn)
        self.n = int(numiter)
        if self.neqn == 1:
            pass 
        elif self.neqn>1:
            if self.neqn == len(self.initial):
                    pass
            else: 
                    raise ValueError('Number of equations doesn\'t equal the number of initial conditions')
            
    def solve(self, time):
        self.time = np.asarray(time)
        n = len(self.time)
        self.sol = np.zeros((n, self.neqn))
        self.sol[0,:] = np.asarray(self.initial)
        y, f, t = self.sol, self.f, self.time 
        for i in np.arange(1, self.n-1):
            y[i,:]= y[i-1,:]+ np.multiply(self.h, f(y[i-1,:], t[i-1]))
        return y
    
class backwardsEuler:
    def __init__(self, f, h,initial, numeqn,  numiter):
        self.f = f
        self.h = h 
        self.initial = initial
        self.neqn = int(numeqn)
        self.n = int(numiter)
        if self.neqn == 1:
            pass 
        elif self.neqn>1:
            if self.neqn == len(self.initial):
                    pass
            else: 
                    raise ValueError('Number of equations doesn\'t equal the number of initial conditions')          
    def solve(self, time):
        self.time = np.asarray(time)
        n = len(self.time)
        self.sol = np.zeros((n, self.neqn))
        self.sol[0,:] = np.asarray(self.initial)
        f = self.f
        t = self.time
        for i in np.arange(0, n-1):
            g = lambda y: y - np.multiply(self.h, f(y,t[i+1])) - self.sol[i,:]
            roots = sy.optimize.fsolve(g, self.sol[i,:])
            self.sol[i+1,:]= roots
        return self.sol

class Lax_Wendroff:
    """
    To be generalized This class solve parabolic partial differential equations. The class takes a function f, initial condition, bounds, start time, end time, and a mesh--in that order.
    Note: Mesh may not be programmed in... If there is isn't a multiple of the spatial that equals the time step, set it equal to 1. It is assumed (for now) that there are constant
    coeffs
    
    """
    def __init__(self, f, initial, xbounds, tbounds, factor):
        self.f = f
        self.initial = initial 
        self.xbounds = np.linspace(xbounds[0], xbounds[1])
        self.tbounds = np.linspace(tbounds[0],tbounds[1])
        self.factor = factor
    def LWsolve(self):
        lenx = len(self.xbounds)
        lent =len(self.tbounds)
        x = self.xbounds
        t = self.tbounds
        self.sol = np.zeros((lenx, lent))
        self.sol[0,:] = 0
        self.sol[lenx-1,:] = 0
        v = self.sol
        for j in range(2,lenx):
            for n in range(1,lent):
                dx = x[j]-x[j-1]
                x_j = j*dx
                v[j,0] = self.initial(x_j)
                dt = (self.factor)*dx
                t_n  = n*dt
                v[j-1,n] =self.f(v[j-1,n-1]) + np.divide(dt, 2*dx)*(self.f(v[j,n-1] - v[j-2,n-1])) + np.divide(pow(dt,2), 2*pow(dx,2))*(self.f((v[j,n-1] - 2*v[j-1,n-1] + v[j-2,n-1])))
        fig = plt.figure(dpi=600)
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax1 = fig.add_subplot(1,2,2, projection='3d')
        for n in range (lent):
            ax.plot(x,t,v[:,n])
            
        for n in range(lenx):
            ax1.plot(x,t,v[n,:])
        return v

class EulerMthdpde:
    ''' To be generalized...Euler's Method. This method is unique and can't be used to solve odes necessarily. It is assumed that your function has constant coeffs.
    '''
    def __init__(self, f, initial, xbounds, tbounds, factor):
        self.f = f
        self.initial = initial 
        self.xbounds = np.linspace(xbounds[0], xbounds[1])
        self.tbounds = np.linspace(tbounds[0],tbounds[1])
        self.factor = factor
      
    def solve(self):
        lenx = len(self.xbounds)
        lent =len(self.tbounds)
        x = self.xbounds
        t = self.tbounds
        self.sol = np.zeros((lenx, lent))
        self.sol[0,:] = 0
        self.sol[lenx-1,:] = 0
        v, f= self.sol, self.f
        for j in np.arange(2,lenx):
            for n in np.arange(1,lent):
                dx = x[j]-x[j-1]
                x_j = j *dx
                v[j,0] = self.initial(x_j)
                dt = t[n]-t[n-1]
                t_n = self.factor*pow(dx,2)
                v[j-1,n] = f(v[j-1,n-1]) + np.divide(dt, pow(dx,2))*(f(v[j,n-1])-2*f(v[j-1,n-1])+f(v[j-2,n-1]))
                
        fig = plt.figure(dpi=600)
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax1 = fig.add_subplot(1,2,2, projection='3d')
        for n in range (lent):
            ax.plot(x,t,v[:,n])
            
        for n in range(lenx):
            ax1.plot(x,t,v[n,:])
        return v

class gridError:
    def __init__(self, f, steps, initial, numeqn,  numiter):
        self.f = f
        self.h = np.asarray(steps)
        self.initial = initial
        self.neqn = int(numeqn)
        self.n = int(numiter)
        if self.neqn == 1:
            pass 
        elif self.neqn>1:
            if self.neqn == len(self.initial):
                    pass
            else: 
                    raise ValueError('Number of equations doesn\'t equal the number of initial conditions')
    def solve(self, time):
        self.time = np.asarray(time)
        n = len(self.time)
        h =self.h
        error = np.zeros((len(h),n-1,self.neqn))
        self.sol = np.zeros((len(h),n, self.neqn))
        self.sol[0,:] = np.asarray(self.initial)
        y, f, t = self.sol, self.f, self.time 
        for j in np.arange(len(h)):
            for i in np.arange(1,self.n-1):
                y[j,i,:]= y[j,i-1,:]+ np.multiply(self.h[j], f(y[j,i-1,:], t[i-1]))
                error[j,i,:]=np.abs(y[j,i,:]-y[j,i-1,:])
        return error, y

def file_extract(path):
    file_pat = input('What is the pattern?')
    results = []
    lis = ''
    for root, dirs, file in os.walk(path):
        pattern = re.compile('(%s)' %file_pat)
        #files = gb.glob('root' + r'[a-z_]\w+[a-z-]\w*\d{4}-\d{2}-\d{2}\.json')
        for i in np.arange(len(file)):
            isl = file[i].replace(',','')
            isl = isl.strip("\'")
            lis+=isl
        matches = pattern.finditer(lis)
        for match in matches:
            results.append(os.path.abspath(match.group(0)))
    return results

def memoize(func):
    cache = {}
    # @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key]=func(*args, **kwargs)
        return cache[key]
    return wrapper


class cobwebplot:
    """
    This class is the CobwebPlot for discrete time systems. It takes a function (f), number of equations (numeq), start-value (start must be an integer),
    initial condition (xinitial), and a stop-value (stop must be an integer)-- in that order. For numeq>1, an error arises np.asarray should be used.
    """
    def __init__(self, f, numeq,start,xinitial, stop):
        self. xinitial =xinitial
        self.f = f 
        self.numeq = numeq
        self.start = start 
        self.stop = stop
        tp = set(map(type, [self.start, self.stop]))
        if tp.issubset([int,...]) == False:
            raise TypeError(tp)
        try:
            if self.numeq != len(xinitial):
                raise ValueError('Number of equations don\'t equal number of initial values')
        except TypeError:
            pass 
    def cobweb(self):
        xk, xkk = np.empty((2,self.stop+1,2))
        xk[0], xkk[0]= self.xinitial, self.f(xk[0])
        for i in np.arange(self.start+1, self.stop, 2):
            xk[i] = xk[i-1]
            xkk[i]=self.f(xk[i-1])
            xk[i+1] = xkk[i]
            xkk[i+1] = xkk[i]
            print(f'{xk[i]} \t\t\t {xkk[i]}')
        
        if self.numeq > 1:
            fig, axs = plt.subplots(self.numeq)
            for i in range(self.numeq):
                axs[i-1].plot(xk[0,:], xkk[0,:], color='red')
                axs[i].plot(xk[1,:], xkk[1,:], color='orange')
        else:
            t = np.linspace(-5,5,num=1000)
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.scatter(xk, xkk, color='black', marker='.')
            ax1.plot(xk, xkk, color='red')
            ax1.plot(t,t, color='blue')
#             ax1.plot(t, self.f(t), color='seagreen')
            ax1.set_title('Cobweb Plot x_{k+1} vs. x_{k}')  
            ax2.plot(range(len(xk)), xk, color='firebrick')
            ax2.set_title('Time Series x_{k} vs. k')

class Discretel:
    """This programs find solutions for fdifference equations
    """
    def __init__(self, f, numeq, initc, intern):
        self.f = f
        self.numeq = numeq
        self.initc =np.asarray(initc)
        self.intern = intern
        if self.numeq !=len(self.initc):
            raise ValueError('Number of equations don\'t equal the number of initial conditions')
        else:
            pass
    def discretl(self):
        k = input('How many trajectories?')
        xk = np.zeros((int(k),self.intern, self.numeq))
        xk[0,0,:] = self.initc
        for j in range(1,int(k)):
            xk[j,0,:] = np.random.randint(10, size=(1,self.numeq))
            for i in range(1, self.intern):
                xk[j,i,:]=self.f(xk[j,i-1,:])
        return xk

class LU_Decomp:
    def __init__(self, A, B):
        self.A =np.asarray(A)
        self.b = np.asarray(B)
    @staticmethod
    def mult_matrix(M, N):
        """Multiply square matrices of same dimension M and N"""
        '---Credit: quantstart'
        # Converts N into a list of tuples of columns                                                                                                                                                                                                      
        tuple_N = zip(*N)

        # Nested list comprehension to calculate matrix multiplication                                                                                                                                                                                     
        return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]

    @staticmethod
    def pivot_matrix(M):
        m = len(M)
        id_mat = np.eye(m, dtype = np.double)
        for k in iter(range(m)):
            row = max(iter(range(k,m)), key=lambda i: np.abs(M[i,k]))
            if k != row:
                id_mat[[k,row]] = id_mat[[row,k]]
        return id_mat
        
    def plu(self):
        A = self.A
        n,m = A.shape
        di = np.diag_indices(m)
        L = np.identity(m,dtype=np.double)
        U = np.zeros_like(L, dtype=np.double)
        P = self.pivot_matrix(A)
        PA = self.mult_matrix(P,A)
        for j in iter(range(m)):
            L[j,j] = 1.0
            for i in iter(range(j+1)):
                s1 = np.einsum('kj,ik->ij',U,L)
                U[i,j] = (P@A)[i,j] - s1[i,j]
            for i in iter(range(j,m)):
                s2 = np.einsum('kj,ik->ij', U,L)
                L[i,j] = ((P@A)[i,j] - s2[i,j])/U[j,j]
        L[di] =1
        return P,L,U
    
    def solve(self):
        P,L,U = self.plu()
        n,m = L.shape
        b = np.dot(P,self.b)
        y = np.zeros_like(b, dtype = np.double)
        x = y.copy()
        y[0] = b[0]/L[0,0]
        for i in range(1, m):
            y[i] = (b[i] - np.dot(L[i,:i],y[:i]))/L[i,i]
        x[-1] = y[-1]/U[-1,-1]
        for i in range(m-2,-1,-1):
            x [i] = (y[i] -np.dot(U[i,i:],x[i:]))/U[i,i]
        return x
class Spectral:
    '''
    This class return D and D2 matrices.
    The only argument the user must provide is the number of point (n)
    '''
    def __init__(self, n):
        self.n = int(n)

    @staticmethod
    def xn(n):
        return np.array([np.cos(np.pi*i/n) for i in range(n+1)])

    def d_mats(self):
        print('The first returned argument is Chebyshev differential matrix, D, and the second order differential\n Chebyshev matrix, D2')
        n = self.n
        x = self.xn(n).copy()
        D = np.zeros((len(x), len(x)))
        D2 = np.zeros_like(D)
        c = np.zeros((len(x),))
        c[0]=2
        c[len(c)-1]=2
        c[1:len(c)-1]=1
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j :
                    D[i,j] = (pow(-1,i+j)*c[i])/(c[j]*(x[i]-x[j]))
                    D2[i,j] = 2*D[i,j]*(D[i,i] -pow(x[i]-x[j],-1))
                elif j ==0 and i == 0:
                    D[0,0] = (2*pow(n,2)+1)/6
                elif i==len(x)-1 and j==len(x)-1:
                    D[len(x)-1,len(x)-1] = -(2*pow(n,2)+1)/6
                elif j !=0 and i != 0 and j != len(x)-1 and i!=len(x)-1 and i == j:
                    D[i,j] = -x[j]/(2*(1-pow(x[j],2)))
                    for n in range(len(x)):
                        sm =0
                        sm +=-D2[i,n]
                    D2[i,j]=sm
        return D, D2

class Gram_Schmidt:
    '''
    This class computes the orthonormal basis of a matrix. The user must first instantiate the class
    with a matrix, A, and it will return the associated basis of the matrix.
    '''
    def __init__(self, A):
        self.A = np.asarray(A)
    @staticmethod
    def diag_sign(A):
        D = np.diag(np.sign(np.diag(A)))
        return D
    @classmethod
    def adjust_sign(cls, Q,R):
        D = cls.diag_sign(Q)
        Q[:,: ]= Q@D
        R[:,:] = D@R
        return Q,R
    def process(self):
        A = self.A.copy()
        n,m = A.shape
        Q = np.empty((n,n))
        u = np.empty((n,n))
        proj = np.array([])

        u[:,0] = A[:,0]
        Q[:,0] = u[:,0]/norm(u[:,0])

        for i in range(1,n):
            u[:,i] = A[:,i]
            for j in range(i):
                u[:,i]-=(A[:,i]@Q[:,j])*Q[:,j]
                proj = np.append(proj, A[:,i]@Q[:,j])

            Q[:,i] = u[:,i]/norm(u[:,i])

        R = np.zeros((n,m))
        for i in range(n):
            for j in range(i,m):
                R[i,j] = A[:,j]@Q[:,i]
        Q, R =self.adjust_sign(Q,R)
        return  Q,R, proj

##class QR_Householder:
##    '''
##    This class employs the Householder Reflection Transformation to compute Q and R
##    decompositions of a matrix A. The user must supply the matrix A. 
##    '''
##    def __init__(self, A):
##        self.A = A
##        
##    @staticmethod
##    def mult_matrix(M, N):
##        """Multiply square matrices of same dimension M and N"""
##        # Converts N into a list of tuples of columns                                                                     
##        tuple_N = zip(*N)
##
##        # Nested list comprehension to calculate matrix multiplication                                                    
##        return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]
##
##    @staticmethod
##    def trans_matrix(M):
##        """Take the transpose of a matrix."""
##        n = len(M)
##        return [[ M[i][j] for i in range(n)] for j in range(n)]
##
##    @staticmethod
##    def euclid_norm(x):
##        """Return the Euclidean norm of the vector x."""
##        return pow(sum([x_i**2 for x_i in x]), 1/2)
##    
##    @staticmethod
##    def Q_i(Q_min, i, j, k):
##        """Construct the Q_t matrix by left-top padding the matrix Q                                                      
##        with elements from the identity matrix."""
##        if i < k or j < k:
##            return float(i == j)
##        else:
##            return Q_min[i-k][j-k]
##        
##    @staticmethod
##    def cmp(a,b):
##        return (a>b) - (a<b)
##    
##    def householder(self):
##        """Performs a Householder Reflections based QR Decomposition of the                                               
##        matrix A. The function returns Q, an orthogonal matrix and R, an                                                  
##        upper triangular matrix such that A = QR."""
##        A = self.A
##        n = len(A)
##
##        # Set R equal to A, and create Q as a zero matrix of the same size
##        R = A
##        Q = [[0.0] * n for i in iter(range(n))]
##
##        # The Householder procedure
##        for k in range(n-1):  # We don't perform the procedure on a 1x1 matrix, so we reduce the index by 1
##            # Create identity matrix of same size as A                                                                    
##            I = [[float(i == j) for i in iter(range(n))] for j in iter(range(n))]
##
##            # Create the vectors x, e and the scalar alpha
##            # Python does not have a sgn function, so we use cmp instead
##            x = [row[k] for row in R[k:]]
##            e = [row[k] for row in I[k:]]
##            alpha = -1*self.cmp(x[0],0) * self.euclid_norm(x)
##
##            # Using anonymous functions, we create u and v
##            u = list(map(lambda p,q: p + alpha * q, x, e))
##            norm_u = self.euclid_norm(u)
##            v = list(map(lambda p: p/norm_u, u))
##
##            # Create the Q minor matrix
##            Q_min = [ [float(i==j) - 2.0 * v[i] * v[j] for i in iter(range(n-k))] for j in iter(range(n-k)) ]
##
##            # "Pad out" the Q minor matrix with elements from the identity
##            Q_t = [[ Q_i(Q_min,i,j,k) for i in iter(range(n))] for j in iter(range(n))]
##
##            # If this is the first run through, right multiply by A,
##            # else right multiply by Q
##            if k == 0:
##                Q = Q_t
##                R = self.mult_matrix(Q_t,A)
##            else:
##                Q = self.mult_matrix(Q_t,Q)
##                R = self.mult_matrix(Q_t,R)
##
##        # Since Q is defined as the product of transposes of Q_t,
##        # we need to take the transpose upon returning it
##        return self.trans_matrix(Q), R

class Vector_Constructor:
    def __init__(self, f, bounds, n):
        self.f = f
        self.b = np.asarray(bounds)
        self.n = n

    def construct(self):
        b = self.b.copy()
        x = []
        for i in range(b.shape[0]):
            a,c = b[i]
            x.append(np.linspace(a,c, num=self.n))

        x = np.asarray(x)
        A=[]
        for item in product(*x):
            A.append(np.concatenate((list(item),[self.f(item)]),axis=0))
        A = np.asarray(A).T
        
##        xx = (x[:-1,:]
##        var = []
##        for item in islice(cycle(xx), len(xx)):
##            var.a
        return A

