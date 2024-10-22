import numpy as np 
import itertools as it

np.set_printoptions(threshold=100)
rng = np.random.default_rng(seed=0)

class Neural_Network:
    def __init__(self,X_train: np.ndarray[float], Y_train: np.ndarray[float], num_layers: list[int], alpha: float) -> None:
        self.Xbias = np.c_[X_train, np.ones(X_train.shape[0])]
        self.Y = Y_train[:, np.newaxis]
        self.dim = X_train.shape
        self.nl = num_layers
        self.alpha = alpha
        self.weights = [rng.random((self.dim[1]+1, num_layers[0]))]

        for i in range(len(num_layers)-1):
            self.weights.append(rng.random((num_layers[i], num_layers[i+1])))

        self.weights.append(rng.random((num_layers[-1],1)))
        print(self.Y.shape)
        
    @staticmethod
    def activation(x):
        return np.maximum(0,x)
    
    @staticmethod
    def deriv_activation(x):
        return x>0
    
    @staticmethod
    def loss(y_true, y_pred):
        return 2*(y_pred.ravel()-y_true.ravel())/y_true.size
    def forward(self):
        self.Z = []
        self.A = [self.Xbias]
        for i in range(len(self.weights)):
            if i == len(self.weights)-1:
                self.Z.append(self.A[-1].dot(self.weights[i]) )
                self.A.append(self.Z[-1])
            else:
                self.Z.append(self.A[-1].dot(self.weights[i]))
                self.A.append(self.activation(self.Z[-1]))
        
        return self.Z, self.A
            
    def backward_propagation(self, Z,A):
        self.Z,self.A = Z,A
        error = -(2/len(self.Y))*(self.Y-self.A[-1])
        dW = dict()
        dW[len(self.weights)-1] = self.A[-2].T.dot(error)
        for i in reversed(range(len(self.weights)-1)):
            error = error@self.weights[i+1].T*self.deriv_activation(self.Z[i]) 
            dW[i] = self.A[i].T.dot(error)
        # for i in range(len(dW)):
        #     dW[i] = np.clip(dW[i], -self.clip, self.clip)
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha*dW[i]
        

    def train(self):
        initial_error = 0 
        for i in it.count():
            Z, A = self.forward()
            self.backward_propagation(Z,A)
            new_error = np.abs(self.Y-A[-1]).mean()
            
            if np.isclose(initial_error,new_error, rtol=1e-20) == True:
                print(f'Training Stopped at Iteration={i}')
                break
            elif i > 0 and new_error > initial_error:
                print(f'Training Stopped at Iteration={i} because of overfitting')
                # break
            else:
                initial_error = new_error
            if i % 500 == 0:
                corr_matrix = np.corrcoef(self.Y.ravel(), A[-1].ravel())
                corr = corr_matrix[0,1]
                r2 = pow(corr,2)
                print(f'Iteration={i}\thas\tMSE={np.square(self.Y-A[-1]).mean()}\tand\t R^2={r2}')
            

        print(f'Trained Prediction:\n{A[-1]}')
        print(f'True Model Output:\n{self.Y}')
        corr_matrix = np.corrcoef(self.Y.ravel(), A[-1].ravel())
        corr = corr_matrix[0,1]
        r2 = pow(corr,2)
        print(f'Train Pearson\'s R^2:\t{r2}\twith\tMSE={np.square(self.Y-A[-1]).mean()}')
    def predict(self, X: np.ndarray[float],y: np.ndarray[float]) -> np.ndarray[float]:
        self.Xbias = np.c_[X,np.ones(X.shape[0])]
        Z,A = self.forward()
        corr_matrix = np.corrcoef(y.ravel(), A[-1].ravel())
        corr = corr_matrix[0,1]
        r2 = pow(corr,2)
        print(f'Test Pearson\'s R^2:\n{r2}')
        return A[-1]


# L1 Regularization:
# Loss(omega, b) = (1/n)\sum_{i=0}^{n}(y_true-f_{omega}(z))^{2} + lambda*\sum_{i=0}^{n}abs{omega}

# L2 Regularization 
# Loss(omega, b) = (1/n)\sum_{i=0}^{n}(y_true-f_{omega}(z))^{2} + lambda*\sum_{i=0}^{n}(omega)^{2}
    
    

