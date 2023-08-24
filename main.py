import numpy as np
import matplotlib.pyplot as plt

# make the neural network based on model that has been made
class Model_neural:
    def __init__(self, train_itteration=1000, learning_rate=0.01):
        self.train_itteration = train_itteration
        self.learning_rate = learning_rate
        
        self.W1 = np.random.rand(3, 2)
        self.b1 = np.random.rand(3, 1)
        
        self.W2 = np.random.rand(2, 3)
        self.b2 = np.random.rand(2, 1)
        
        self.comulative_erros = []
    
    def params(self):
        return self.W1, self.b1, self.W2, self.b2
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def deriv_sigmoid(self, Z):
        return (1- self.sigmoid(Z)) * (self.sigmoid(Z))
    
    def ReLu(self, Z):
        return np.maximum(0, Z)
    
    def deriv_ReLu(self, Z):
        return (Z > 0) * 1
    
    # only works in 2x1 array
    def error(self, pred, true):
        return np.sum(np.square(pred - true))
    
    # can use for the vector and matrix
    def predict(self, X):
        Z1 = np.add(self.W1.dot(X.T), self.b1)
        a1 = self.sigmoid(Z1)
        
        Z2 = np.add(self.W2.dot(a1), self.b2)
        a2 = self.ReLu(Z2)
        
        prediction = a2
        
        return prediction
    
    # it's only can be used by vector
    def gradient_descent(self, x, y):
        # forwrd propegation
        Z1 = np.add(self.W1.dot(x), self.b1)
        a1 = self.sigmoid(Z1)
        
        Z2 = np.add(self.W2.dot(a1), self.b2)
        a2 = self.ReLu(Z2)
        
        # backward propegation
        dC_da2 = 2 * (a2 - y)
        
        dW2 = np.ones((2, 3)) * a1.T * self.deriv_ReLu(Z2) * dC_da2
        db2 = self.deriv_ReLu(Z2) * dC_da2
        
        dC_da1 = self.W2.T.dot((self.deriv_ReLu(Z2) * dC_da2))
                
        dW1 = np.ones((3, 2)) * x.T * self.deriv_sigmoid(Z1) * dC_da1
        db1 = self.deriv_sigmoid(Z1) * dC_da1        
        return dW1, db1, dW2, db2
    
    def update_param(self, dW1, db1, dW2, db2):
        self.W1 -= dW1 * self.learning_rate
        self.W2 -= dW2 * self.learning_rate
        self.b1 -= db1 * self.learning_rate
        self.b2 -= db2 * self.learning_rate
    
    def Train(self, X, Y, y_size, batch=5):
        if batch >= 5:
            # initiate the gradient descent
            dW1, db1, dW2, db2 = 0, 0, 0, 0
            
            self.comulative_erros = []
            
            if y_size >= batch:
                total_number = int(self.train_itteration/batch)
                n = 1
                for itter in range(1, self.train_itteration+1):
                    rd_idx = np.random.randint(0, y_size)
                    x = X[rd_idx].reshape(2, 1)
                    y = Y[rd_idx].reshape(2, 1)
                    
                    dW1_, db1_, dW2_, db2_ = self.gradient_descent(x, y)
                    
                    dW1 += dW1_
                    db1 += db1_
                    dW2 += dW2_
                    db2 += db2_
                    
                    # for every batch dW and db update we are going to update the param
                    # with the avarage dW and dB
                    if itter % batch == 0:
                        dW1 = dW1/batch
                        db1 = db1/batch
                        dW2 = dW2/batch
                        db2 = db2/batch
                        
                        self.update_param(dW1, db1, dW2, db2)
                        error = self.error(pred=self.predict(X), true=Y.T)
                        print(f"{n}/{total_number} error [mse]: {error}")                
                        self.comulative_erros.append(error)
                        n += 1
                        
                        # we turn the dW and db into 0 again because we are 
                        # going to start from the begining again
                        dW1, db1, dW2, db2 = 0, 0, 0, 0
            
            else:
                for itter in range(self.train_itteration):
                    rd_idx = np.random.randint(0, y_size)
                    x = X[rd_idx].reshape(2, 1)
                    y = Y[rd_idx].reshape(2, 1)
                    
                    dW1, db1, dW2, db2 = self.gradient_descent(x, y)
                    
                    self.update_param(dW1, db1, dW2, db2)
                    
                    error = self.error(pred=self.predict(X), true=Y.T)
                    print(f"error [mse]: {error}")
                    self.comulative_erros.append(error)
        
        else:
            print("batch min is 5")
            
            
    
    def save_error_record(self, fname=None, ftype='txt'):
        if fname == None:    
            f = open(f"error_record.{ftype}", 'w')
        else:
            f = open(f"{fname}.{ftype}", 'w')
        
        if ftype == 'txt':
            f.write(self.comulative_erros)
        
        elif ftype == 'csv':
            n = 0
            for i in self.comulative_erros:
                f.write(f"{n},{i}\n")
                n += 1

test_input = np.array([
    [3, 1.5],
    [2, 1],
    [4, 1.5],
    [3, 4],
    [3.5, 0.5],
    [2, 0.5],
    [2.5, 1],
    [1, 1],
    [1, 1.5],
    [2, 4]
])

test_output = np.array([
    [1, 0],
    [1.1, 2.3],
    [0, 1.4],
    [1.5, 0],
    [1.1, 0],
    [1.3, 0],
    [1.1, 0.4],
    [1.1, 0],
    [1.2, 0],
    [0, 1]
])

model = Model_neural(train_itteration=1000)

model.Train(X=test_input, Y=test_output, y_size=10, batch=5)

plt.plot(model.comulative_erros)
plt.savefig("error-plot.png")
plt.show()

model.save_error_record(ftype='csv')