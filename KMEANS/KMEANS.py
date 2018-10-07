
# 数据集链接：https://archive.ics.uci.edu/ml/datasets/iris

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

feature_number = 4

def inputData(path):
    first_flag = 1
    f_open = open(path)

    myinput = []

    input_x = np.zeros(feature_number)

    for row in f_open:

        row = row.replace("\t","").replace("\n","")

        row = row.split(",")

        if row[feature_number] == 'Iris-setosa':
            row[feature_number] = 0
        elif row[feature_number] == 'Iris-versicolor':
            row[feature_number] = 1
        elif row[feature_number] == 'Iris-virginica':
            row[feature_number] = 2

        if first_flag == 1:
            myinput = row
            first_flag = 0
        else :
            myinput = np.vstack((myinput,row))

    # print(myinput)

    input_x = np.array([[float(row[i]) for i in range(0,feature_number)] for row in myinput])
    input_y = np.array([[int(row[feature_number])] for row in myinput])

    return input_x,input_y

# 2范式距离
def calDistant(a,b):
    return np.sqrt(np.sum(np.power(a-b,2)))

# 虽然这是一个分类的数据集，不过我们可以使用聚类的方式，忽视掉最后的分类，看我们的聚类是否将他们聚成了一个类了
def kmeans(k,train_x,max_epoch):

    # 随机初始化k个聚类中心
    center = np.zeros((k,feature_number))
    cluster = np.zeros((train_x.shape[0],1))
    cluster_num = np.zeros(k)

    for i in range(k):

        center[i] = train_x[int(np.random.rand()*train_x.shape[0])] # 由于样例中的给的数据的范围都是0-10范围内的



    for epoch in range(max_epoch):
        change = 0

        for i in range(train_x.shape[0]):
            min_dis = 10000
            temp_cluster = -1

            # 分别计算样例距离各个聚类中心的距离
            for j in range(k):
                dis = calDistant(center[j],train_x[i])
                if dis < min_dis:
                    min_dis = dis
                    temp_cluster = j

            if change == 0 and temp_cluster == cluster[i][0]:
                change = 1
            cluster[i] = temp_cluster


        # 此时得到了对应的新的聚类,更新聚类中心
        for i in range(train_x.shape[0]):
            center[int(cluster[i])] = center[int(cluster[i])] + train_x[i]

        for j in range(k):
            cluster_num[j] = np.sum(cluster == j)
            center[j] = center[j] / cluster_num[j]

        if change == 0:# 所有的点都没有变化了
            break


    return cluster

# 由于这是一个4个属性的数据集，是无法可视化的，由肉眼观察得出1类和2类第二个属性隔得不远
def plotOrigin(train_x):
    plt.figure()

    ax = plt.subplot(111,projection='3d')

    x = train_x[:,0:1]
    y = train_x[:,2:3]
    z = train_x[:,3:4]

    ax.scatter(x,y,z)

    plt.show()

    pass


def plotColor(train_x,cluster):
    plt.figure()

    first_flag1 = first_flag2 = first_flag3 = 1

    x1, y1, z1 = np.zeros((1,1)),np.zeros((1,1)),np.zeros((1,1))
    x2, y2, z2 = np.zeros((1,1)),np.zeros((1,1)),np.zeros((1,1))
    x3, y3, z3 = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))
    ax = plt.subplot(111, projection='3d')



    for i in range(train_x.shape[0]):
        if cluster[i] == 0:
            if first_flag1 == 1:
                x1 = train_x[i][0]
                y1 = train_x[i][2]
                z1 = train_x[i][3]
                first_flag1 = 0
            x1 = np.vstack((x1, train_x[i][0]))
            y1 = np.vstack((y1, train_x[i][2]))
            z1 = np.vstack((z1, train_x[i][3]))
        elif cluster[i] == 1:
            if first_flag2 == 1:
                x2 = train_x[i][0]
                y2 = train_x[i][2]
                z2 = train_x[i][3]
                first_flag2 = 0
            x2 = np.vstack((x2, train_x[i][0]))
            y2 = np.vstack((y2, train_x[i][2]))
            z2 = np.vstack((z2, train_x[i][3]))
        elif cluster[i] == 2:
            if first_flag3 == 1:
                x3 = train_x[i][0]
                y3 = train_x[i][2]
                z3 = train_x[i][3]
                first_flag3 = 0
            x3 = np.vstack((x3, train_x[i][0]))
            y3 = np.vstack((y3, train_x[i][2]))
            z3 = np.vstack((z3, train_x[i][3]))

    ax.scatter(x1, y1, z1, color='y')
    ax.scatter(x2, y2, z2, color='r')
    ax.scatter(x3, y3, z3, color='b')

    plt.show()

if __name__ == '__main__':

    train_x,train_y =  inputData("data\\train.txt")

    cluster = kmeans(3,train_x,100)

    # print(cluster)
    plotOrigin(train_x)
    plotColor(train_x, cluster)
    plotColor(train_x, train_y)
    pass