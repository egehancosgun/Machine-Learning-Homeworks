import numpy as np
import matplotlib.pyplot as plt

#Data Loading
data = np.genfromtxt("hw03_data_set.csv",delimiter=",")
eruptions = data[1:,0]
waiting = data[1:,1]

#Data Splits
x_train = eruptions[:150]
x_test = eruptions[150:]
y_train = waiting[:150]
y_test = waiting[150:]

#Parameters
bin_width = 0.37
origin = 1.5

minimum_value = np.min(x_train)
maximum_value = np.max(x_train)
N = x_train.shape[0]

#Borders (From Lab Implementation)
left_borders = np.arange(origin, maximum_value, bin_width)
right_borders = np.arange(origin + bin_width, maximum_value+bin_width, bin_width)

#Regressogram
regressogram = []
for i in range(len(left_borders)):
   flag_list = []
   values = []
   for k in range(N):
       flag_list.append((left_borders[i] < x_train[k]) & (x_train[k] <= right_borders[i]))
       if flag_list[k] == True:
           values.append(y_train[k])
   nominator = np.sum(values)
   denominator = np.sum(flag_list)        
   regressogram.append(nominator/denominator)
         
#Plotting
plt.figure(figsize=(10,4))
plt.plot(x_train,y_train,"b.",markersize=10,alpha=0.3)
plt.plot(x_test,y_test,"r.",markersize=10,alpha=0.8)
plt.legend(["training","test"])
plt.title("h=0.37",fontweight="bold")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram[b], regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram[b], regressogram[b + 1]], "k-")  
plt.show()


#RMSE 
y_predicted = []
for i in range(len(y_test)):
    for j in range(len(left_borders)):
        if (x_test[i] > left_borders[j]) & (x_test[i] <= right_borders[j]):
            y_predicted.append(regressogram[j])
y_predicted = np.array(y_predicted)
RMSE = np.sqrt(np.sum((y_test-y_predicted)**2)/len(y_test))
print(f"Regressogram => RMSE is {RMSE:.4f} when h is {bin_width}")

#Running Mean Smoother 
interval = np.linspace(origin,maximum_value,10001)
running_mean_smoother = []
for i in range(len(interval)):
    flag_list = []
    values = []
    for k in range(N):
        flag_list.append(np.abs((interval[i]-x_train[k])/bin_width)<0.5)
        if flag_list[k] == True:
            values.append(y_train[k])
    nominator = np.sum(values)
    denominator = np.sum(flag_list)
    running_mean_smoother.append(nominator/denominator)  

#Plotting      
plt.figure(figsize=(10,4))
plt.plot(x_train,y_train,"b.",markersize=10,alpha=0.3)
plt.plot(x_test,y_test,"r.",markersize=10,alpha=0.8)
plt.legend(["training","test"])
plt.title("h=0.37",fontweight="bold")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(interval,running_mean_smoother, "k")
plt.plot(interval,running_mean_smoother, "k")  
plt.show()

#RMSE
y_predicted = []
for i in range(len(y_test)):
    for j in range(len(interval)):
        if j+1>=len(interval):
            break
        if (x_test[i] > interval[j]) & (x_test[i] <= interval[j+1]):
            y_predicted.append(running_mean_smoother[j])

y_predicted = np.array(y_predicted)
RMSE_running = np.sqrt(np.sum((y_test-y_predicted)**2)/len(y_test))
print(f"Running Mean Smoother => RMSE is {RMSE_running:.4f} when h is {bin_width}")

#Kernel Smoother
interval_2 = np.linspace(origin,maximum_value,10001)
kernel_smoother = []
for i in range(len(interval_2)):
    nominator_list = []
    denominator_list = []
    for k in range(N):
        a = (interval_2[i]-x_train[k])/bin_width
        denominator_list.append((1/np.sqrt(2*np.pi)*(np.exp(-1*a**2/2))))
        nominator_list.append(denominator_list[k]*y_train[k])
        
    nominator = np.sum(nominator_list)
    denominator = np.sum(denominator_list)
    kernel_smoother.append(nominator/denominator) 
    
#Plotting      
plt.figure(figsize=(10,4))
plt.plot(x_train,y_train,"b.",markersize=10,alpha=0.3)
plt.plot(x_test,y_test,"r.",markersize=10,alpha=0.8)
plt.legend(["training","test"])
plt.title("h=0.37",fontweight="bold")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(interval_2,kernel_smoother, "k")
plt.plot(interval_2,kernel_smoother, "k")  
plt.show()

#RMSE
y_predicted = []
for i in range(len(y_test)):
    for j in range(len(interval_2)):
        if j+1>=len(interval_2):
            break
        if (x_test[i] > interval_2[j]) & (x_test[i] <= interval_2[j+1]):
            y_predicted.append(kernel_smoother[j])   
y_predicted = np.array(y_predicted)
RMSE_kernel = np.sqrt(np.sum((y_test-y_predicted)**2)/len(y_test))
print(f"Kernel Smoother => RMSE is {RMSE_kernel:.4f} when h is {bin_width}")    





