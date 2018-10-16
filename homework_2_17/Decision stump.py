import numpy as np
import matplotlib.pyplot as plt
from numpy import *

max_epoch = 5000
feauter_num = 9

# 生成在[-1,1]平均分布的data
def generateData():

    x = linspace(-1,1,10)
    y = []
    y_noise = []

    for i in range(10):
        if x[i] < 0:
            y.append(-1)
            rand = np.random.rand()
            if rand > 0.2:
                y_noise.append(-1)
            else:
                y_noise.append(1)
        else:
            y.append(1)
            rand = np.random.rand()
            if rand > 0.2:
                y_noise.append(1)
            else:
                y_noise.append(-1)

    return x,y,y_noise

def inputData(path):
    f = open(path)

    input_data = np.array([0])
    first_flag = 1

    for row in f:
        row = row.replace("\t", "").replace("\n", "")
        row = row.split(" ")
        if first_flag:
            input_data = row
            first_flag = 0
        else:
            input_data = np.vstack((input_data,row))

    x = input_data[:,1:feauter_num + 1]
    x = x.astype(np.float32)
    y = input_data[:,feauter_num + 1].reshape(x.shape[0],1)
    y = y.astype(np.int32)

    return x,y

def h(theta,s,x):
    return s*sign(x-theta)

def decisionStump(x,y,N):

    delim = (max(x)-min(x))/(N-1)

    min_err = 1
    min_theta = 0
    min_s = 1

    theta = -1 + delim / 2
    count_err = 0

    for i in range(N-1):
        s = 1
        count_err = 0
        for j in range(x.size):
            pred = h(theta,s,x[j])
            if pred != y[j]:
                count_err = count_err + 1

        if count_err/N < min_err:
            min_theta = theta
            min_err = count_err/N
            min_s = s

        s = -1
        count_err = 0
        for j in range(x.size):
            pred = h(theta, s, x[j])
            if pred != y[j]:
                count_err = count_err + 1

        if count_err / N < min_err:
            min_theta = theta
            min_err = count_err / N
            min_s = s

        theta = theta + delim

    return min_theta,min_s,min_err

def plotData(x,y,theta,s):
    pos = y == 1
    neg = y == -1

    plt.scatter(x[pos], y[pos], color="r")
    plt.scatter(x[neg], y[neg], color="b")

    plt.plot(x, h(theta, s, x))

    plt.show()

def computeErrout(theta,s):
    return 0.5 + 0.3 * s * (abs(theta)-1)

# 2元分类问题
def classification():
    err_in = []
    err_out = []

    for epoch in range(max_epoch):
        x, y, y_noise = generateData()
        theta, s, err = decisionStump(x, y_noise, x.size)
        print(x)
        err_in.append(err)
        err_out.append(computeErrout(theta, s))

    plt.hist(err_in, 40)
    plt.show()
    plt.hist(err_out, 40)
    plt.show()

    print("average of err_in = " + str(average(err_in)))
    print("average of err_out = " + str(average(err_out)))

# 多远分类问题,这样做会不会有点欠佳，因为很多数据分类是由几个特征同时决定的，选取一个维度作为分类依据不太对吧
def multiClassification():
    x_train,y_train = inputData("data\\train.txt")
    x_test,y_test = inputData("data\\test.txt")

    theta = []
    s = []
    errin = []


    for j in range(feauter_num):
        x_div = x_train[:,j]

        theta_temp,s_temp,errin_temp = decisionStump(x_div,y_train,x_div.shape[0])

        theta.append(theta_temp)
        s.append(s_temp)
        errin.append(errin_temp)

    print(errin)

    errin = np.array(errin)
    theta = np.array(theta)
    s = np.array(s)

    min_index = errin == min(errin)


    random_number = np.random.rand()

    min_errin = errin[min_index][int(random_number * np.sum(min_index))]
    min_theta = theta[min_index][int(random_number * np.sum(min_index))]
    min_s = s[min_index][int(random_number * np.sum(min_index))]
    min_index = np.argwhere(min_index == True).reshape(1)[int(random_number*np.sum(min_index))]

    print("theta_min = " + str(min_theta))
    print("errin_min = " + str(min_errin))

    pred = h(min_theta,min_s,x_test[:,min_index]).reshape(x_test.shape[0],1)
    correct = pred == y_test
    errout = 1 - np.sum(correct)/x_test.shape[0]

    print("errout = " + str(errout))





    # return theta[min_index][int(np.random.rand() * np.sum(min_index))],\
    #        s[min_index][int(np.random.rand() * np.sum(min_index))], \
    #        errin[min_index][int(np.random.rand() * np.sum(min_index))]

    pass

if __name__ == '__main__':
    multiClassification()

    pass
    # y_noise = np.array(y_noise)
    #
    # plotData(x, y_noise, theta, s)

