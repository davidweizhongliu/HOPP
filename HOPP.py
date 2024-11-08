import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv

distance_min = 2.2 # this is the minimal distance between clusters
def HKmodel(data,x,y,z,distance_min):  #data It should be a three-dimensional list that has been read, and xyz is the list corresponding to each coordinate
    # np.linalg.norm(x) #The default is to find the second norm of the matrix x
    a = data
    x = x[:,np.newaxis] #（1，1900）Row matrix becomes column matrix
    y = y[:,np.newaxis]
    z = z[:,np.newaxis]
    distance_min = distance_min
    updata = True
    while updata :
        narry = []
        for i in range(len(a)):
            # c = [a[i] - a[j] for j in range(len(a))]
            N = [] #Corresponding to row i of A matrix
            cardinal_number = 0
            for j in range(len(a)):
                c = a[i] - a[j]
                norm = np.linalg.norm(c)
                if norm <= distance_min :
                    N.append(1)
                    cardinal_number = cardinal_number + 1
                else:
                    N.append(0)
            N_new = []
            for Ai in N : #Processing of the i-th row of A
                if Ai != 0 :
                    N_new.append(1/cardinal_number)
                else:
                    N_new.append(0)
            narry.append(N_new) #Store the list information of the ith row of A
        A_array = np.mat(narry)
        x_new = A_array * x
        y_new = A_array * y
        z_new = A_array * z
        x_new = np.array(x_new)
        y_new = np.array(y_new)
        z_new = np.array(z_new)
        data_updata = np.hstack((x_new, y_new, z_new))
        if np.linalg.norm(data_updata - a) != 0 :
        # if (data_updata - a ).any() != 0 : #
            a = data_updata
        else:
            updata = False

    #Output convergence matrix
    convergence_array = data_updata
    print(convergence_array)
    # Start to determine the number of clusters that converge, that is, the value of k, and at the same time find out the single type, delete it, and form new data。
    # num = []
    # for i in range(len(convergence_array)) :
    #     for j in range(len(convergence_array)) :
    #         if convergence_array[i] == convergence_array[j] :
    convergence_list = convergence_array.tolist()
    statistics = dict()
    for i in range(len(convergence_list)):
        convergence_coordinate = str(convergence_list[i])
        if convergence_coordinate not in statistics :
            statistics[convergence_coordinate] = 1
        else:
            coordinate_num = statistics[convergence_coordinate]
            coordinate_num += 1
            statistics[convergence_coordinate] = coordinate_num

    print(statistics)
    # Determine the value of k that has nothing to do with the whole
    k = 0
    Irrelevant_value = []
    ir_num = 0
    for key,value in statistics.items() :
        if value > 1 :
            k = k + 1
        if value == 1 :
            print('There is a separate convergence value:' + str(key))
            Irrelevant_value.append(key)
            ir_num = ir_num + 1
    Irrelevant_value = np.array(Irrelevant_value)
    print('Processing is complete, a total of' + str(k) + 'Class, irrelevant values are' + str(ir_num) + 'number。')
    # Process the old data, delete the irrelevant values contained, and form a new data set for Kmeans
    data_new = []
    for i in range(len(data)):
        m = 0
        for j in range(len(Irrelevant_value)) :
            datai = data[i]
            Irrelevant_value_j = np.array(eval(Irrelevant_value[j]))
            # if (data[i] - Irrelevant_value_j).all() == 0 : #Here we need to use the norm to judge, if there is any less than 0 in a.any(), it will be counted as 0
            if np.linalg.norm(data[i] - Irrelevant_value_j) == 0:
                m = m + 1
        if m == 0 :
            data_new.append(data[i])
    print('Data filtering is complete')
    return k , data_new ,ir_num

def TSP_slove(center):
    all_weight = []
    for i in range(len(center)) :
        point_weight = []
        for j in range(len(center)):
            ij_weight = np.linalg.norm(center[i] - center[j])
            point_weight.append(ij_weight)
        all_weight.append(point_weight)
    # print(all_weight)

    sumpath_min = 10000000
    for start_point in range(len(center)) :
        point_num = len(center)
        sumpath = 0
        s = []
        s.append(start_point)
        i = 1
        j = 0
        while True:
            k = 0
            Detemp = 10000000
            while True:
                flag = 0
                if k in s:
                    flag = 1
                if (flag == 0) and (all_weight[k][s[i - 1]] < Detemp):
                    j = k;
                    Detemp = all_weight[k][s[i - 1]];
                k += 1
                if k >= point_num:
                    break;
            s.append(j)
            i += 1;
            sumpath += Detemp
            if i >= point_num:
                break;
        # sumpath += all_weight[start_point][j] #This is the distance back to the starting point
        if sumpath < sumpath_min :
            sumpath_min = sumpath
            s_min = s

    s = s_min
    print("result is：")
    print(sumpath_min)
    for m in range(point_num):
        print("%s-> " % (s[m]), end='')
    return sumpath_min,s

def TSP_slove_julei(center):
    # Calculate the distance matrix
    all_weight = []
    for i in range(len(center)) :
        point_weight = []
        for j in range(len(center)):
            ij_weight = np.linalg.norm(center[i] - center[j])
            point_weight.append(ij_weight)
        all_weight.append(point_weight)
    # print(all_weight)

    i = 1
    n = len(center)
    j = 0
    sumpath = 0
    s = []
    s.append(0)     #Start from the first point

    while True:
        k = 0
        Detemp = 10000000
        while True:
            flag = 0
            if k in s:
                flag = 1
            if (flag == 0) and (all_weight[k][s[i - 1]] < Detemp):
                j = k;
                Detemp = all_weight[k][s[i - 1]];
            k += 1
            if k >= n:
                break;
        s.append(j)
        i += 1;
        sumpath += Detemp
        if i >= n:
            break;
    sumpath += all_weight[0][j]
    print("result：")
    print(sumpath)
    for m in range(n):
        print("%s-> " % (s[m]), end='')
    return sumpath,s

dataA=pd.read_csv(r'C:/David/Robotics/HOPP/data/Separated_apples.csv')
dataX = dataA['x']
x = np.array(dataX)
dataY = dataA['y']
y = np.array(dataY)
dataZ = dataA['z']
z = np.array(dataZ)
DATA2 = np.vstack((x,y,z)).T #Three sets of one-dimensional data are combined into three-dimensional data
print(DATA2)

k,data_new,ir_num = HKmodel(DATA2,x,y,z,distance_min)
print('k is:' + str(k))

# Start kmeans processing
julei = KMeans(n_clusters=k) #Perform kmeans clustering
# julei.fit(data_new) #To cluster the clustered data Modify the point information here, many of the following data_new need to be changed to DATA2
julei.fit(DATA2) #Cluster the clustered data
print('Kmeans Processing complete')
label = julei.labels_ #Obtain cluster labels
print(label)
center = julei.cluster_centers_ #Obtain cluster centers
print(center)

all_sumpath = 0
# Central point processing TSP
sumpath,s = TSP_slove(center)
once_center_sumpath = sumpath
#all_sumpath = all_sumpath + sumpath
# Center point path information storage
center_path_file = 'C:/David/Robotics/HOPP/Results/center_' + str(distance_min) + '_' + str(k) + '_path.csv'
center_path = []
for center_num in s :
    center_and_label = center[center_num].tolist()
    center_and_label.append(center_num)
    center_path.append(center_and_label)
with open(center_path_file, 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile)
    writer.writerow(['x','y','z','label'])
    for center_row in center_path:
        writer.writerow(center_row)
print('The center point path has been saved')

# Random point path planning and the number of times through different clusters
random_sumpath,random_s = TSP_slove(DATA2)

# The following part is mainly a summary of the center point
start_label = label[0]
change_num = 0
random_center_path = []         # A summary of the random center point and its label
center_point_nolabel = []
center_point = center[start_label].tolist()                 # Single center point and its label
center_point_nolabel.append(center[start_label].tolist())     # Only the xyz of the center point is easy to calculate
center_point.append(start_label)
random_center_path.append(center_point)
for random_route in random_s :
    next_label = label[random_route]
    if start_label != next_label :
        center_point = center[next_label].tolist()
        center_point_nolabel.append(center[next_label].tolist())
        center_point.append(next_label)
        random_center_path.append(center_point)
        change_num += 1
    start_label = label[random_route]
print('\nThe random path has passed a total of'+str(change_num)+'Sub-different clusters')
# Random path information storage
random_path_file = 'C:/David/Robotics/HOPP/Results/random_apple_' + str(distance_min) + '_' + str(k) + '_' + str(change_num) + '_path.csv'
random_center_path_file = 'C:/David/Robotics/HOPP/Results/random_center_' + str(distance_min) + '_' + str(k) + '_' + str(change_num) + '_path.csv'
# The following is a summary of the total path of random points
random_path = []
for random_num in random_s :
    random_and_label = DATA2[random_num].tolist()
    random_and_label.append(label[random_num])
    random_path.append(random_and_label)
with open(random_path_file, 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile)
    writer.writerow(['x','y','z','label'])
    for random_row in random_path:
        writer.writerow(random_row)
print('Random point path has been saved')
with open(random_center_path_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y', 'z', 'label'])
    for random_center_row in random_center_path:
        writer.writerow(random_center_row)
print('The center path passed by the random point has been saved')

random_center_sumpath = 0
for center_i in range(len(center_point_nolabel)) :
    if (center_i + 1 ) == len(center_point_nolabel):
        break
    once_length = np.linalg.norm(np.array(center_point_nolabel[center_i]) - np.array(center_point_nolabel[center_i + 1]))
    random_center_sumpath = random_center_sumpath + once_length

# Path planning for each cluster
all_new_road = []
for center_label in s :
    print('doing'+ str(center_label) + 'Apple clustering')
    point_road = []
    need_julei_point = []
    need_julei_point.append(center[center_label])
    for apple in range(len(DATA2)) :
        apple_label = label[apple]
        if label[apple] == center_label :
            need_julei_point.append(DATA2[apple])
    print(str(center_label)+'clustering data is classified, ready for path planning')
    julei_path,julei_s = TSP_slove_julei(need_julei_point)
    for sequence in julei_s :
        point_and_label = need_julei_point[sequence].tolist()
        point_and_label.append(center_label)
        all_new_road.append(point_and_label)
    point_and_label = center[center_label].tolist()
    point_and_label.append(center_label)
    all_new_road.append(point_and_label)
    all_sumpath = all_sumpath + julei_path
print('All path planning has been completed')

# Save all paths and clustering results
all_path_file = 'C:/David/Robotics/HOPP/Results/all_apple_' + str(distance_min) + '_' + str(k) + '_path.csv'
with open(all_path_file, 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile)
    writer.writerow(['x','y','z','label'])
    for row in all_new_road:
        writer.writerow(row)
print('Path saved')

# Clustering and information preservation
results_file = 'C:/David/Robotics/HOPP/Results/all_result.csv'
results = [distance_min,k,ir_num,all_sumpath,once_center_sumpath,random_sumpath,random_center_sumpath,change_num]
with open(results_file, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(results)

# Visualization of results
plt.figure() # Get the picture
ax1 = plt.axes(projection='3d')
ax1.set_xlim(min(x), max(x)) # X axis, horizontal to the right
ax1.set_ylim(min(y), max(y)) # Y axis, left direction and X, Z axis are perpendicular to each other
ax1.set_zlim(min(z), max(z)) # Vertical is the Z axis
color1 = ['r', 'g', 'b', 'c', 'y', 'm', 'darkgrey','#006400','#8B0000','#9400D3','#FF00FF','#E15759', '#4E79A7', '#76B7B2', '#F28E2B','#ADD8E6','#00FF00','#BA55D3','#FFE4B5','#FAA460','#FF69B4','#FFFACD','#00FA9A','#FFE4E1']

# # Visualization of Apple's location
# data_num = 0
# for i in data_new:
#   label_num = label[data_num]
#   ax1.scatter(i[0], i[1], i[2], c=color1[label_num],s=3,marker='o', linewidths=3) # Draw points with scatter function
#   data_num += 1

# Visualization of cluster center points (this can be directly integrated into the visualization of the path)
for i in center:
    # center_point = center[i]
    ax1.scatter(i[0],i[1],i[2], c = '#000000' ,s=7 ,marker='D', linewidths=4)

# Path planning visualization
line_x = []
line_y = []
line_z = []
for i in range(len(center)):
    sequence = s[i]
    line_x.append((center[sequence])[0])
    line_y.append((center[sequence])[1])
    line_z.append((center[sequence])[2])
ax1.plot(line_x,line_y,line_z,color='#000000',linewidth=1)

plt.show()



