import numpy as np
import matplotlib.pyplot as plt

feature_num = 5
learning_rate = 1


def sign(x):

    while type(x) != np.float64:
        x = x[0]

    if x > 0:
        return 1
    else :
        return -1


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

def PocketAlgorithm(train_x,train_y):
    j = 0
    m = len(train_y)
    weight= np.zeros((feature_num,1)) # feature*1 矩阵
    weight_best = np.zeros((feature_num,1)) # feature*1 矩阵
    train_acc_best = 0

    num_of_update_need = 100
    update_time = np.zeros(m) # 每一个数据是否没选中更新
    token = 0


    np.random.rand()
    temp = np.hstack((train_x,train_y))

    np.random.shuffle(temp)

    train_x = temp[:,0:feature_num]
    train_y = temp[:,feature_num:feature_num+1]


    while True:
        for j in range(token,m):
            temp = train_x[j]
            temp = temp.reshape(1, feature_num)

            if sign(np.matmul(temp,weight) * train_y[j][0]) == 1:#如果wx和y同号的话,预测正确
                continue
            else :
                num_of_update_need = num_of_update_need - 1
                token = j
                update_time[j] = update_time[j] + 1
                weight = weight +  learning_rate * train_y[j][0] * temp.T

                train_acc = getCorrectionRate(train_x,train_y,weight)
                if train_acc > train_acc_best:
                    weight_best = weight
                    train_acc_best = train_acc

                if num_of_update_need == 0:
                    return weight_best

        if j == m - 1 and token == 0:  # 此时说明没有错误发生了
            # print("final epoch = " + str(i))
            break
        elif j == m - 1 and token != 0:  # 这个循环里有错误发生，只是遍历完了数据集
            token = 0


    return weight_best


def getCorrectionRate(train_x,train_y,weight):

    m = len(train_x)

    prediction = np.sign(np.matmul(train_x,weight))
    equal_array = prediction == train_y

    # print("rate of accuracy : " + str(np.sum(equal_array)/m))
    # print(equal_array)


    return np.sum(equal_array)/m



if __name__ == '__main__':

    train_x,train_y = inputData("data\\hw1_18_train.txt")
    test_x,test_y = inputData("data\\hw1_18_test.txt")
    error_rate = []


    for i in range(2000):
        weight = PocketAlgorithm(train_x,train_y)

        train_error = 1 - getCorrectionRate(train_x,train_y,weight)
        test_error = 1 - getCorrectionRate(test_x,test_y,weight)
        # print("error rate of train:" + str(train_error))
        #
        # print("error rate of test:" + str(test_error))

        print(weight)
        error_rate.append(test_error)

    print(error_rate)
    print("average error rate of test :" + str(np.average(error_rate)))

    plt.figure()

    plt.xlabel("error rate of test")
    plt.ylabel("frequency")
    plt.hist(error_rate, normed= False , edgecolor="k")

    plt.show()


    pass


# 18 需要你每次更新50次w，这样做2000次，在测试集上计算相关的准确度是多少，化成直方图的形式