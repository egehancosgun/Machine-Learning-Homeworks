import numpy as np
import matplotlib.pyplot as plt

#Data Loading
data = np.genfromtxt("hw04_data_set.csv",delimiter=",")
eruptions = data[1:,0]
waiting = data[1:,1]

#Data Splits
x_train = eruptions[:150]
x_test = eruptions[150:]
y_train = waiting[:150]
y_test = waiting[150:]

def tree(y,P):
    #Initialization(from lab)
    node_indices = {}
    is_terminal = {}
    need_split = {}
    node_means = {}
    node_splits = {}
    # put all training instances into the root node(from lab)
    node_indices[1] = np.array(range(len(x_train)))
    is_terminal[1] = False
    need_split[1] = True
    
    while True: 
        split_nodes = [key for key, value in need_split.items() if value == True] #from lab
        if len(split_nodes) == 0:
            break
        for split_node in split_nodes:
            data_indices = node_indices[split_node] #from lab
            need_split[split_node] = False #from lab
            node_mean = np.mean((y_train[data_indices]))
            if len((x_train[data_indices])) <= P or len(np.unique(y_train[data_indices])) == 1: #pruning checked
                is_terminal[split_node] = True
                node_means[split_node] = node_mean
            else:
                is_terminal[split_node] = False
    
                unique_values = np.sort(np.unique(x_train[data_indices])) #from lab
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2 #from lab
                split_scores = np.repeat(0.0, len(split_positions)) #from lab
    
                for s in range(len(split_positions)):
                    left_indices = data_indices[x_train[data_indices] <= split_positions[s]] #from lab
                    right_indices = data_indices[x_train[data_indices] > split_positions[s]] #from lab
                    error = 0
                    if len(left_indices) > 0:
                        error += np.sum((y_train[left_indices] - np.mean(y_train[left_indices]))**2) #sum squared error
                    if len(right_indices) > 0:
                        error+= np.sum((y_train[right_indices] - np.mean(y_train[right_indices]))**2) #sum squared error
                    split_scores[s] = error / len(data_indices)           
    
                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split
    
                # create left node using the selected split (from lab implementation)
                left_indices = data_indices[x_train[data_indices] < best_split]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True
    
                # create right node using the selected split (from lab implementation)
                right_indices = data_indices[x_train[data_indices] >= best_split]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
    #Prediction            
    i=1
    while is_terminal[i] == False:
        if y <= node_splits[i]:
            i = 2*i
        else:
            i = 2*i + 1
    return node_means[i] 
          
interval = np.linspace(np.min(eruptions), np.max(eruptions),1000)
y_predicted = [tree(interval[i],25) for i in range(len(interval))]

#Plotting
plt.figure(figsize=(10,4))
plt.title("P=25",fontweight="bold")
plt.plot(x_train,y_train,"b.",markersize=10,alpha=0.3)
plt.plot(x_test,y_test,"r.",markersize=10,alpha=0.8)
plt.plot(interval, y_predicted, "k")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training","test"])
plt.show()

#RMSE
y_test_predicted = [tree(x_test[i],25) for i in range(len(x_test))]
RMSE = np.sqrt(np.mean((y_test - y_test_predicted)**2))
print(f"RMSE is {RMSE:.4f} when P is 25")

#Changing pruning parameter
pruning_parameters = [5*i for i in range(1,11)]
RMSE_list = []
predictions = {}
for p in pruning_parameters:
    predictions[p] = [tree(x_test[i],p) for i in range(len(x_test))]
    
for key,value in predictions.items():
    RMSE_list.append(np.sqrt(np.mean((y_test - value)**2)))

#Plotting
plt.figure(figsize=(10,4))
plt.scatter(pruning_parameters,RMSE_list ,color="k")
plt.plot(pruning_parameters,RMSE_list, color="k")
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.show()
    
    