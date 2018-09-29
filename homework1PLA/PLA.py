import numpy as np
import time
import matplotlib.pyplot as plt

feature_num = 5
epoch = 10
learning_rate = 1


# train_x 是 m*feature_num 的矩阵, train_y是 m*1 的矩阵
def inputData(path):

    f_open = open(path)

    input_array = np.array([feature_num + 1])

    first_flag = 1

    for row in f_open:
        row = row.replace("\t"," ").replace("\n","")
        row = row.split(" ")

        if first_flag == 1:
            first_flag = 0
            input_array = row
        else :
            input_array = np.vstack((input_array,row))


    my_train_x = np.array([[float(row[i]) for i in range(feature_num-1)] for row in input_array])
    my_train_y = np.array([[float(row[4])]for row in input_array])

    my_train_x = np.hstack(((np.ones((len(my_train_y),1))),my_train_x))

    # print(my_train_x)
    # print(my_train_y)


    return my_train_x,my_train_y

def sign(x):
    x = x.tolist()

    if x[0][0] > 0:
        return 1
    else:
        return -1

def PLA(train_x,train_y):
    weight = np.zeros((feature_num,1)) # w是feature_num*1的向量
    updates_time = np.zeros(len(train_y))
    token = 0

    # 一轮就是遍历完了所有的数据
    for i in range(epoch):
        for j in range(token,len(train_x)):
            temp = train_x[j]
            temp = temp.reshape(1,feature_num)
            if sign(np.matmul(temp,weight) * train_y[j][0]) == 1:
                continue
            elif sign(np.matmul(temp,weight) * train_y[j][0]) == -1:
                token = j
                updates_time[j] = updates_time[j] + 1
                weight = weight + learning_rate * train_y[j][0] * temp.T

        if j == len(train_x)-1 and token == 0:#此时说明没有错误发生了
            print("final epoch = " + str(i))
            break
        elif j == len(train_x)-1 and token != 0:#这个循环里有错误发生，只是遍历完了数据集
            token = 0


    return weight,updates_time

# 每次随机选取错误点
def PLArand(train_x,train_y):
    m = len(train_y)
    weight = np.zeros((feature_num, 1))  # w是feature_num*1的向量
    updates_time = np.zeros(len(train_y))

    np.random.rand()

    temp = np.hstack((train_x, train_y))

    np.random.shuffle(temp)

    train_x = temp[:,0:feature_num]
    train_y = temp[:,feature_num:feature_num+1]

    # train_x = temp[:][0:feature_num]
    # train_y = temp[:][feature_num:feature_num + 1]

    token = 0

    # 一轮就是遍历完了所有的数据
    for i in range(epoch):
        for j in range(token, len(train_x)):
            temp = train_x[j]
            temp = temp.reshape(1, feature_num)

            if sign(np.matmul(temp, weight) * train_y[j][0]) == 1:
                continue
            elif sign(np.matmul(temp, weight) * train_y[j][0]) == -1:
                token = j
                updates_time[j] = updates_time[j] + 1
                weight = weight + learning_rate * train_y[j][0] * temp.T

        if j == len(train_x) - 1 and token == 0:  # 此时说明没有错误发生了
            # print("final epoch = " + str(i))
            break
        elif j == len(train_x) - 1 and token != 0:  # 这个循环里有错误发生，只是遍历完了数据集
            token = 0

    return weight, updates_time

    pass

def getCorrectionRate(train_x,train_y,weight):

    m = len(train_x)

    prediction = np.sign(np.matmul(train_x,weight))
    equal_array = prediction == train_y

    print("rate of accuracy : " + str(np.sum(equal_array)/m))
    # print(equal_array)



    return prediction
    pass



if __name__ == '__main__':
    train_x,train_y = inputData("data\\hw1_15_train.txt")

    updates_num = []

    for i in range(2000):
        weight,updates_time = PLArand(train_x,train_y)
        updates_num.append(np.sum(updates_time))
        print(weight)

    # prediction = getCorrectionRate(train_x,train_y,weight)

    print(weight)
    #
    print(updates_num)

    print("average number of updates: " + str(np.average(updates_num)))


    plt.figure()

    plt.xlabel("number of updates")
    plt.ylabel("frequency")
    plt.hist(updates_num,normed=True,bins=15,color="steelblue",edgecolor="k")

    plt.show()

    # print("index of the most number of updates:" + str(np.where(updates_time == np.max(updates_time))))


