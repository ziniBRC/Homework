import numpy as np
import matplotlib.pyplot as plt

feature_num = 4
w = np.zeros((feature_num + 1,1))
learning_rate = 1.6
max_epoch = 300

# train_x 是 m*feature_num 的矩阵, train_y是 m*1 的矩阵
def inputData(path):

    f_open = open(path)

    input_array = np.array([feature_num + 1])

    first_flag = 1

    for row in f_open:
        row = row.replace("\t"," ").replace("\n","")
        row = row.split(" ")

        if row[feature_num] == '-1':
            row[feature_num] = '0'

        if first_flag == 1:
            first_flag = 0
            input_array = row
        else :
            input_array = np.vstack((input_array,row))


    my_train_x = np.array([[float(row[i]) for i in range(feature_num)] for row in input_array])
    my_train_y = np.array([[float(row[4])]for row in input_array])

    my_train_x = np.hstack(((np.ones((len(my_train_y),1))),my_train_x))


    return my_train_x,my_train_y

# x is constant or array
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(train_x,train_y,w):
    m = train_y.shape[0]

    loss_function = -1/m *np.sum(np.multiply(train_y,np.log2(sigmoid(np.matmul(train_x,w))))+np.multiply(1-train_y,np.log2(1-sigmoid(np.matmul(train_x,w)))))

    return loss_function

def gradientDescendWithoutReg(train_x,train_y,w):
    m = train_x.shape[0]

    w = (w.T - learning_rate * 1/m * np.sum(np.multiply(sigmoid(np.matmul(train_x,w)) - train_y,train_x),0)).T

    return w

def getAccuracy(train_x,train_y,w):


    pred = sigmoid(np.matmul(train_x,w))

    # print(pred)

    pred = np.int64(pred>0.5)

    accuracy = np.sum(pred == train_y)/train_x.shape[0]

    return accuracy

if __name__ == '__main__':
    train_x,train_y = inputData("data\\train.txt")

    for epoch in range(max_epoch):
        w = gradientDescendWithoutReg(train_x,train_y,w)
        print("loss : " + str(loss(train_x,train_y,w)))

    print("accuracy:" + str(getAccuracy(train_x,train_y,w)))

    pass