
import numpy as np
from numpy.linalg import inv
import pandas as pnd
import matplotlib.pyplot as plt

alpha=0.01
iterr=1500

class LinearRegresion:
    def compute_cost(self,X,y,theta):
        m=X.shape[0]
        h_theta=np.dot(X,theta)
        squared_error=np.power((h_theta-y),2)
        j_theta=1/(2*m)*np.sum(squared_error)
        return j_theta;    
    
    def gradient_descent(self,X,y,theta,alpha,no_itter):
        m=X.shape[0]
        j_hist=[]
        #made equations on a piece of paper to see what is going on
        for i in range(no_itter):
           h_theta=X.dot(theta) #X * theta:97x2,2x1
          # errors=h_theta-y #97x1
          # new_X=errors.T.dot(X) #97x1.T 97x2
           theta=theta-alpha*(1/m)*((h_theta-y).T.dot(X)).T #we transpose it
           j_hist.append(self.compute_cost(X,y,theta))
        return theta,j_hist,thinking
   
    def plot_data(self,X,y,theta):
        plt.figure(figsize=(8,5))
        plt.scatter(X[:,1],y,c='blue',s=12)
        plt.plot(X[:,1],np.dot(X,theta),c='red')
        plt.show()
        
    def normalization(self,X):#we do normalization if more features differ alot frrom eachother
        x_normalized=(X-np.average(X))/(np.max(X)-np.min(X))
        return x_normalized
    
def normal_equation(X,y):
    return inv(X.T.dot(X)).dot(X.T.dot(y))        
#it is slower with numpy    
#lines=np.loadtxt("ex1data1.txt",delimiter=',',unpack=False)


data=pnd.read_csv('ex1data1.txt',header=None);

X=data.values[:,0].reshape(data.shape[0],1)#making it 96x1 vector
y=data.values[:,1].reshape(data.shape[0],1);
theta=np.zeros((2,1))#we added a column of zeros for theta0

model=LinearRegresion()

X=np.append(np.ones((data.shape[0],1)),X,axis=1);

model.plot_data(X,y,theta)

theta,j_history,thinking=model.gradient_descent(X,y,theta,alpha,iterr)
#print("Computed cost is ",model.compute_cost(X,y,theta));
print(thinking)
model.plot_data(X,y,theta)




