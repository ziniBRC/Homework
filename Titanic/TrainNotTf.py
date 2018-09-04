import numpy as np
import matplotlib as plt
import csv

# 是否被获救

mylambda = 1
lr = 0.03

# 设置每一层数值的维度
layer1_dim = 7
layer2_dim = 30
layer3_dim = 100
output_dim = 1

# 首先设置每一层的表示形式

a2 = np.zeros(layer2_dim)
a3 = np.zeros(layer3_dim)
pred = np.zeros(output_dim)
y = np.zeros(output_dim) # y的维度是m*2

# 打算设置2个隐藏层 一个30层 一个100层 ,所以这里存在有3个weight的属性,在0-1的范围内随机初始化
# 由于存在一个bias所以这里weight1:13*30  weight2:31*100 weight3:101*2
weight1 = np.random.randn(layer1_dim + 1,layer2_dim)
weight2 = np.random.randn(layer2_dim + 1,layer3_dim)
weight3 = np.random.randn(layer3_dim + 1,output_dim)

# 在计算导数的时候我们需要先计算delta的值才行

delta2 = np.zeros(layer2_dim)
delta3 = np.zeros(layer3_dim)
delta4 = np.zeros(output_dim)

# 关于每个参数的偏导数

Derivative1 = np.zeros((layer1_dim + 1,layer2_dim))
Derivative2 = np.zeros((layer2_dim + 1,layer3_dim))
Derivative3 = np.zeros((layer3_dim + 1,output_dim))


whole_weight = [weight1, weight2, weight3]

def getCsvData(path):
    myinput = np.empty(layer1_dim)
    csv_reader = csv.reader(open(path))
    first_flag = 1

    for row in csv_reader:
        if first_flag == 1:
            myinput = row
            first_flag = 0
        else :
            myinput = np.vstack((myinput,row)) # 将csv中每一行读入到x的下一行中
    return myinput

# input一共有12个属性，但是我们不是每个都会分析，提取我们要分析的属性并将其数值化
def abstractFeature(origin_input):
    # PassengerId
    # survival	Survival	0 = No, 1 = Yes                                 output
    # pclass	    Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd               1
    # Name
    # sex     	Sex  0 = male, 1 = female                                   1
    # Age     	Age in years                                                1
    # sibsp	# of siblings / spouses aboard the Titanic                      1
    # parch	# of parents / children aboard the Titanic                      1
    # ticket	    Ticket number
    # fare	    Passenger fare                                              1
    # cabin	    Cabin number    这个属性有好多人并没有，暂时不分析这个属性
    # embarked	Port of Embarkation	C = Cherbourg 1, Q = Queenstown 2, S = Southampton 3 1
    col_selected = [2,4,5,6,7,9,11]
    first_flag = 2


    # 排除有问题的数据，将字符串数据转化成数值形式
    for row in origin_input :
        if row[4] == 'female':
            row[4] = 1
        elif row[4] == 'male' :
            row[4] = 0

        if row[11] == 'C':
            row[11] = 1
        elif row[11] == 'Q':
            row[11] = 2
        elif row[11] == 'S':
            row[11] = 3
        else:
            row[11] = 0

        if row[5] == '':
            row[5] = -1

    # 递推式构造x，y
    y = np.array([[float(origin_input[i][1])] for i in range(1,origin_input.shape[0])])

    x = np.array([[float(origin_input[i][j]) for j in col_selected] for i in range(1,origin_input.shape[0]) ]) # shape:891*7

    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         x[i][j] = float(x[i][j])


    # x.astype(float)

    return x,y
    pass

def sigmoid (para):
    return 1 / (1 + np.exp(-para))

def hypothesis(weight,a):
    # 就以a1作为例子 a1:m * 12 weight1:13 * 30  a2:m * 30
    m = a.shape[0]
    temp = np.ones((m,1)) # 1*1
    temp = np.hstack((temp,a))
    return sigmoid(np.matmul(temp,weight)) # 使用矩阵的正常乘法运算

def lossFunc(x_set,y_set):
    # 首先计算的是h theta(x)
    m = x_set.shape[0]

    a2 = hypothesis(weight1,x_set) # m*30
    a3 = hypothesis(weight2,a2) # m*100
    pred = hypothesis(weight3,a3) # m*2

    # 这里应该是m*2的矩阵所有元素加起来
    loss_standard = - 1/m * np.sum((np.multiply(y_set,np.log(pred)) + np.multiply(np.ones(y_set.shape) - y_set, np.log(np.ones(y_set.shape) - pred)))) # 这里的乘法是对应元素逐个相乘

    loss_regulation = mylambda/(2*m) * (np.sum(np.multiply(weight1,weight1)) + np.sum(np.multiply(weight2,weight2)) + np.sum(np.multiply(weight3,weight3)))# 所有weight的平方和

    return loss_standard + loss_regulation

# 反向传播算法  其结果是得到weight值的导数
def BP(x_train):
    m = x_train.shape[0]

    capital_delta1 = np.zeros((layer1_dim + 1,layer2_dim))  # 用于累计每次传播的时候更改的weight的值的大小
    capital_delta2 = np.zeros((layer2_dim + 1,layer3_dim))
    capital_delta3 = np.zeros((layer3_dim + 1,output_dim))

    for i in range(x_train.shape[0]) :
        row = np.resize(x_train[i],(1,layer1_dim))

        # 正向传播计算a
        a2 = hypothesis(weight1, row)  # 1*30
        a3 = hypothesis(weight2, a2)  # 1*100
        pred = hypothesis(weight3, a3)  # 1*2

        # 计算delta
        delta4 = pred - y_train[i]  # 1*1
        delta3 = np.multiply(np.matmul(weight3[1:layer3_dim + 1], delta4.T).T, np.multiply(a3, np.ones(a3.shape) - a3)) # 100*1.T .* 100*1
        delta2 = np.multiply(np.matmul(weight2[1:layer2_dim + 1], delta3.T).T, np.multiply(a2, np.ones(a2.shape) - a2))

        capital_delta1 = capital_delta1 + np.matmul(np.hstack((np.ones((1,1)),row)).T,delta2)
        capital_delta2 = capital_delta2 + np.matmul(np.hstack((np.ones((1,1)),a2)).T,delta3)
        capital_delta3 = capital_delta3 + np.matmul(np.hstack((np.ones((1,1)),a3)).T,delta4)

    Derivative1 = 1 / m * capital_delta1 + mylambda / m * weight1
    Derivative1 = Derivative1[0] - mylambda / m * weight1[0] # 正则项的导数不用减
    Derivative2 = 1 / m * capital_delta2 + mylambda / m * weight2
    Derivative2 = Derivative2[0] - mylambda / m * weight2[0]  # 正则项的导数不用减
    Derivative3 = 1 / m * capital_delta3 + mylambda / m * weight3
    Derivative3 = Derivative3[0] - mylambda / m * weight3[0]  # 正则项的导数不用减

    return Derivative1,Derivative2,Derivative3





if __name__ == '__main__':

    original_input = getCsvData("data\\train.csv")

    x,y = abstractFeature(original_input)

    x_test = x[400:550]  # 选取x的400-550作为测试用
    x_cv = x[550:x.shape[0]]  # 剩下的预留下作为验证集，暂时貌似用不到
    x_train = x[0:400] # 选取train数据集中400个作为训练用的训练集

    y_test = y[400:550]
    y_cv = y[550:y.shape[0]]
    y_train = y[0:400]


    # 预备迭代20次
    for i in range(200):
        Derivative1,Derivative2,Derivative3 = BP(x_train) # 计算这次迭代的参数的导数

        # 更新每个权值的大小,自己在这里画个图，这里应该是减号
        weight1 = weight1 - lr * Derivative1
        weight2 = weight2 - lr * Derivative2
        weight3 = weight3 - lr * Derivative3

        # 打印loss函数的大小
        print("loss = " + str(lossFunc(x_train,y_train)))

    # 正向传播计算a
    a2 = hypothesis(weight1, x_test)  # 1*30
    a3 = hypothesis(weight2, a2)  # 1*100
    pred_test = hypothesis(weight3, a3)

    for row in pred_test:
        if row[0] >= 0.5:
            row[0] = 1
        else:
            row[0] = 0

    error = np.abs(pred_test - y_test)
    error_rate = np.sum(error)/pred_test.shape[0]
    correct_rate = 1 - error_rate
    print("rate of prediction = " + str(correct_rate))




# 日志 2018/9/3 已经完成了BP算法的编写，之后我还需要使用梯度下降算法去重赋值weight的值，更改之后把数据放在程序上去跑，看效果怎么样
# 日志 2018/9/4 遇到了一个很烦人的问题，就是我们读入的属性值很多都是字符串，而在计算过程中我们采用的是数值，如何将字符串转化为数值是个问题
#               自己手写的遇到了很大的问题，第一是无法确认自己写的算法是否是正确的，不过看loss一直在下降可以猜出来应该是正确的，但
#               最好写一个能检查的函数，比如利用Gradient Check来检查自己求导是否是正确的。