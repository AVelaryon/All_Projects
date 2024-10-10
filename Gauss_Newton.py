import numpy as np
import itertools as it
import matplotlib.pyplot as plt

class optimization:
    '''
    This class is an attempt a creating an optimization function to determine "best-fit"
    coefficients for a linear or nonlinear equation. The first argument that the user must supply is the function (f),
    the set of data points (x), the y-data (y), the jacobian (jac), the initial estimate (p0), the tolerance (tol), and the
    maximum number of iterations (max_iter)--- in that order. NOTE: When you define your jacobian, your returned value should use np.column_stack, otherwise... well
    
    '''
    def __init__(self,f,xpts, ypts, jac, p0, tol, max_iter):
        self.f = f
        self.x = np.asarray(xpts)
        self.y = np.asarray(ypts)
        self.j = jac
        self.init = p0
        self.tol = tol
        self.iter = max_iter
        if isinstance(self.init, (float,int)) ==True:
            self.p0 = float(p0)
        else:
            self.p0 = np.asarray(p0)
        
        if len(self.x) != len(self.y):
            raise ValueError('The length of your x-data doesn\'t equal the length of y-data')
        
        
    def Gauss_Newton(self):
        x , y = self.x, self.y
        itre = self.iter
        init = self.init
        jacob = self.j(x,init)
        res1  = np.zeros((len(x), itre))
        for i in range(itre):
            beta = init
            #res1[:, i] = y - self.f(x,beta) Setting a condtion for residuals needs improvement
            res = y - self.f(x,beta)
            jacob = self.j(x,beta)
            print(f'Iteration:{i} and coeffs {beta} with residual {np.linalg.norm(res)}')
            init  = beta +(np.linalg.inv(jacob.T@jacob)@jacob.T)@res
            if np.linalg.norm(beta - init)<self.tol:
                break
            if np.linalg.norm(beta-init)>1e7:
                print('The error is getting worst, which isn\'t supposed to happen! Try something else...')
                break
        return init
    
class Newton2D:

    def __init__(self, F: np.ndarray[callable], Jac: np.ndarry[callable], tol: float, max_iter: int) -> None:
        self.F = F
        self.J = Jac
        self.tol = tol
        self.iter = max_iter

    def nwmthd2d(self, initial: np.ndarray[float]) -> np.ndarray[float]:
        X = np.zeros((self.F.shape[0], self.iter))
        X[:,0] = initial
        for i in it.chain(range(1,self.iter)):
            X[:, i] = X[:, i-1] - np.linalg.inv(self.J(*X[:,i-1])).dot(self.F(*X[:,i-1]))
            if np.linalg.norm(X[:,i]-X[:,i-1])<self.tol:
                break
            else:
                pass
        
    



# def f(x, init):
#     a,b,c,d = init
#     return a+ b*np.sin(c*x +d)

# def jac(x,init):
#     a,b,c,d = init
#     return np.column_stack([[1]*len(x), np.sin(c*x+d), x*np.cos(c*x +d), np.cos(c*x+d)])
# x = np.linspace(0.1, 10, num=100)
# y = f(x,[3,5,6,9]) 
# op = optimization(f, x, y, jac, [3,5,6.5,9], 1e-6, 600)
# model = op.Gauss_Newton()

# plt.scatter(x, y, color='red', marker='.')
# plt.plot(x, f(x,model))
# plt.show()

def f(v,w):
    b,s,q=1/25, 11111/1e7, 83333/1e6

    return v*pow(b+w,2)-s*(b+2*w)/pow(w,2) -q

def g(v,w):
    b,s,d,n=1/25, 11111/1e7, 64537/50000,-887299/500000
    return (-b/2)*(1+v) + (s/2)*pow(w, -2) - w + (1/2)*(1-v)*w - (1/2)*d*pow(1-v, 1/2)-n
def F(v,w):
    return np.array([f(v,y),g(v,w)])
def Jac(v,w):
    return np.array([[pow(25*w+1,2), 2*(pow(w, 4)*v+4e-2*pow(w,3)*v+1.1111e-3*w+4.4444e-5)/pow(w,3)],
                     [3.22685e-1/pow(1-v,1/2)-(1/2)*(w+4e-2),(-1/2)*(pow(w,3)*(1+v)+1)/pow(w,3)]
                     ])

model2 = Newton2D(F, Jac, 1e-8, 1000)
initial = np.array([1,1/2])
print(model2.nwmthd2d(initial))