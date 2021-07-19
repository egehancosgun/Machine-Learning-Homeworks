import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Dataset Import & Visualization

X = np.genfromtxt("hw05_data_set.csv",delimiter = ",",skip_header=1)

plt.scatter(X[:,0],X[:,1], c = "Black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([-6,-4,-2,0,2,4,6])
plt.yticks([-6,-4,-2,0,2,4,6])
plt.show()

# Initializing the EM Algorithm

centroids = np.genfromtxt("hw05_initial_centroids.csv",delimiter = ",")
K = centroids.shape[0]
N = X.shape[0]
D = X.shape[1]
initial_clusters = np.zeros((len(X),1))
for i in range(len(X)):
    initial_clusters[i] = np.argmin([np.linalg.norm(X[i]-centroids[j]) for j in range(K)])
    
X_clusters = np.hstack((X,initial_clusters))

#Visualization of Clusters (Black rectangles are the centers of each cluster.)

plt.scatter(X[:,0],X[:,1], c = initial_clusters)
plt.scatter(centroids[:,0],centroids[:,1],marker = "s", c = "black", edgecolors = "face")  
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([-6,-4,-2,0,2,4,6])
plt.yticks([-6,-4,-2,0,2,4,6])
plt.show()

covariances = np.array([np.cov(X[X_clusters[:,2] == i], rowvar = False, bias = True) for i in range(K)])
prior_probs = np.array([len(X[X_clusters[:,2] == i])/X.shape[0] for i in range(K)])

#EM Algorithm

def EM (centroids,covariances,prior_probs,X,iterations):
    
    K = centroids.shape[0]
    N = X.shape[0]
    D = X.shape[1]
    
    #E Step
    
    for m in range(iterations):  
        
        posterior_probs = np.zeros((K,N))
        for i in range(K):
            for j in range(N):
                posterior_probs[i,j] = prior_probs[i] * multivariate_normal(centroids[i], covariances[i]).pdf(X[j])
        posterior_probs = posterior_probs / posterior_probs.sum(0)
        
    #M Step 
           
        proportions = np.zeros((K,1))
        for i in range(K):
            for j in range(N):
                proportions += posterior_probs[i,j]
        proportions = proportions / N
         
        centroids = np.zeros((K,D))
        
        for i in range(K):
            for j in range(N):
                centroids[i] += X[j] * posterior_probs[i,j]
            centroids[i] = centroids[i] / posterior_probs[i, :].sum()
        
        covariances = np.zeros((K,D,D))
        for i in range(K):
            for j in range(N):
                diff = np.reshape(X[j]-centroids[i],(2,1))
                covariances[i] += posterior_probs[i,j] * np.matmul(diff, np.transpose(diff))
            covariances[i] = covariances[i] / posterior_probs[i,:].sum()
        
    return centroids, covariances
        
#EM for 100 iterations

centers , covariances = EM(centroids,covariances,prior_probs,X,100)
print(centers)

new_clusters = np.zeros((len(X),1))
for i in range(len(X)):
    new_clusters[i] = np.argmin([np.linalg.norm(X[i]-centers[j]) for j in range(K)])
    
#Plotting

data_centers = np.array([[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5], [0.0 ,0.0]])
data_covariances = np.array([ [[0.8, -0.6], [-0.6, 0.8]], [[0.8, 0.6], [0.6, 0.8]], [[0.8, -0.6], [-0.6, 0.8]], [[0.8, 0.6], [0.6, 0.8]], [[1.6, 0.0], [0.0, 1.6]] ])

x, y = np.meshgrid(np.linspace(-6,6,1000),np.linspace(-6,6,1000))
xy = np.column_stack([x.flat, y.flat])

for i in range(K):
    z1 = multivariate_normal.pdf(xy, data_centers[i], data_covariances[i]).reshape(x.shape)
    z2 = multivariate_normal.pdf(xy, centers[i], covariances[i]).reshape(x.shape)
    plt.contour(x,y,z2, levels = [0.05], colors = "black")
    plt.contour(x,y,z1, levels = [0.05], colors = "black", linestyles = "dashed")

plt.scatter(X[:,0],X[:,1], c = new_clusters)
plt.scatter(centers[:,0],centers[:,1],marker = "s", c = "black")  
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([-6,-4,-2,0,2,4,6])
plt.yticks([-6,-4,-2,0,2,4,6])
plt.show()
    


