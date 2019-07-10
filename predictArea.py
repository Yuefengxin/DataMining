#%% 导入库、准备工作
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import random

# importing sklearn libraries
from sklearn import neural_network, linear_model, preprocessing, svm, tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import warnings

# supressing the warning on the usage of Linear Regression model
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

#%% 读取数据
forest_fires = pd.read_csv('forestfires.csv')
forest_fires.head()

#%% 数据分析
print("Number of instances in dataset = {}".format(forest_fires.shape[0]))
print("Total number of columns = {}".format(forest_fires.columns.shape[0]))
print("Column wise count of null values:-")
print(forest_fires.isnull().sum())
#%% 月份和日期转换
forest_fires.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
forest_fires.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)
forest_fires.head()
#%% 每一列的数据统计分析
forest_fires.describe()
#%% 数据及的相关性分析
forest_fires.corr()
#%% 相关性点图
temp_cols=["X","Y","month","FFMC","DMC","DC","ISI","temp","RH","wind","rain","area"]
area_scatter = pd.plotting.scatter_matrix(forest_fires[temp_cols], diagonal="kwds", figsize=(16, 16))
#%% 每一列数据的直方图
histograms = forest_fires.hist(figsize=(16, 16), bins=20)
#%% 火灾面积直方图
plt.xlabel("Burned Area", fontsize="x-large")
plt.ylabel("No. of instances", fontsize="x-large")
forest_fires["area"].hist(figsize=(16, 8), bins=100)
#%% month and DC and RH and temp的相关性
from itertools import combinations
from scipy.stats import pearsonr
for pair in combinations(forest_fires.columns, 2):
    col_1, col_2 = pair
    # Calculate the coefficient and p-value
    corr_coef, p_val = pearsonr(forest_fires[col_1], forest_fires[col_2])
    # Check for high correlation
    if corr_coef >0.86 or corr_coef <-0.52:
        # Print details for pairs with high correlation
        print("Column pair : {}, {}".format(*pair))
        print("Correlation coefficient : {}".format(corr_coef))
        print("p-value : {}".format(p_val))
#%% 提取特征
x_values = list(forest_fires['X'])
y_values = list(forest_fires['Y'])

loc_values = []

for index in range(0, len(x_values)):
    temp_value = []

    temp_value.append(x_values[index])
    temp_value.append(y_values[index])
    loc_values.append(temp_value)
#%%
month_values = list(forest_fires['month'])
day_values = list(forest_fires['day'])

ffmc_values = list(forest_fires['FFMC'])
dmc_values = list(forest_fires['DMC'])
dc_values = list(forest_fires['DC'])
isi_values = list(forest_fires['ISI'])

temp_values = list(forest_fires['temp'])
rh_values = list(forest_fires['RH'])
wind_values = list(forest_fires['wind'])
rain_values = list(forest_fires['rain'])

area_values = list(forest_fires['area'])
#%%
attribute_list = []

for index in range(0, len(x_values)):
    temp_list = []
    
    temp_list.append(x_values[index])
    temp_list.append(y_values[index])
    
    temp_list.append(month_values[index])
    temp_list.append(day_values[index])

    temp_list.append(ffmc_values[index])
    temp_list.append(dmc_values[index])
    temp_list.append(dc_values[index])
    temp_list.append(isi_values[index])

    temp_list.append(temp_values[index])
    temp_list.append(rh_values[index])
    temp_list.append(wind_values[index])
    temp_list.append(rain_values[index])
    
    attribute_list.append(temp_list)
#%% 计算数据集中位置点的函数
def count_points(x_points, y_points, scaling_factor):
    count_array = []
    
    for index in range(0, len(x_points)):
        temp_value = [x_points[index], y_points[index]]
        count = 0
        
        for value in loc_values:
            if(temp_value == value):
                count = count + 1
        count_array.append(count * scaling_factor )

    return count_array
#%% 地点的散点图
plt.figure(figsize=(8, 6))    
    
ax = plt.subplot()    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)
    
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left() 
    
plt.title("Fire location plot", fontsize = 22)
plt.scatter(x_values, y_values, s = count_points(x_values, y_values, 25), alpha = 0.3)
plt.show()
#%% 绘制直方图函数
def histogram_plot(dataset, title):
    plt.figure(figsize=(8, 6))    
    
    ax = plt.subplot()    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left() 
    
    plt.title(title, fontsize = 22)
    plt.hist(dataset, edgecolor='black', linewidth=1.2)
#%% 烧伤面积> 0的数据集百分比
total_count = 0
positive_data_count = 0

for value in area_values:
    if(value > 0):
        positive_data_count = positive_data_count + 1
    total_count = total_count + 1

print("The number of data records with 'burned area' > 0 are " + str(positive_data_count) + " and the total number of records are " + str(total_count) + ".")
print("The percentage value is " + str(positive_data_count/total_count * 100) + ".")
#%% 面积对数和直方图
log_area_values = list(np.log(np.array(area_values) + 1))
histogram_plot(log_area_values, title = "Burned area distribution")
#%% 拆分测试集和训练集，设置初始参数
train_x, test_x, train_y, test_y = train_test_split(attribute_list, log_area_values, test_size=0.3, random_state = 9)
mse_values = []
variance_score = []
#%% 线性回归模型
linear_regression = linear_model.LinearRegression()

linear_regression.fit(train_x, train_y)
predicted_y = linear_regression.predict(test_x)

print('Coefficients: \n', linear_regression.coef_)

print("\nMean squared error: ", mean_squared_error(test_y, predicted_y))
print('Variance score: %.2f' % r2_score(test_y, predicted_y))

mse_values.append(mean_squared_error(test_y, predicted_y)**0.5)
variance_score.append(r2_score(test_y, predicted_y))
#%% 决策树模型
decision_tree = tree.DecisionTreeRegressor(presort = True)

decision_tree.fit(train_x, train_y)
predicted_y = decision_tree.predict(test_x)

print("Mean squared error: ", mean_squared_error(test_y, predicted_y))
print('Variance score: %.2f' % r2_score(test_y, predicted_y))

mse_values.append(mean_squared_error(test_y, predicted_y))
variance_score.append(r2_score(test_y, predicted_y))

#%% SVM模型
svm_model = svm.SVR()

svm_model.fit(train_x, train_y)
predicted_y = svm_model.predict(test_x)

print("Mean squared error: ", mean_squared_error(test_y, predicted_y))
print('Variance score: %.2f' % r2_score(test_y, predicted_y))

# 使用网格搜索法，选择SVM回归中的最佳C值、epsilon值和gamma值
epsilon = np.arange(0.1, 1.5, 0.2)
C= np.arange(0.1, 1, 0.2)
gamma = np.arange(0.001, 0.01, 0.002)
parameters = {'epsilon':epsilon, 'C':C, 'gamma':gamma}
grid_svr = GridSearchCV(estimator = svm.SVR(), param_grid = parameters,
            scoring = 'neg_mean_squared_error', cv = 5, verbose = 1, n_jobs = 2)
# 模型在训练数据集上的拟合
grid_svr.fit(train_x, train_y)
# 返回交叉验证后的最佳参数值
print(grid_svr.best_params_, grid_svr.best_score_)
# 模型在测试集上的预测
pred_grid_svr = grid_svr.predict(test_x)
# 计算模型在测试集上的MSE值
print("Mean squared error: ", mean_squared_error(test_y, pred_grid_svr))
print('Variance score: %.2f' % r2_score(test_y, pred_grid_svr))

mse_values.append(mean_squared_error(test_y, pred_grid_svr))
variance_score.append(r2_score(test_y, pred_grid_svr))
#%% 随机森林模型
random_forest = RandomForestRegressor()
random_forest.fit(train_x, train_y)
predicted_y = random_forest.predict(test_x)

print("Mean squared error: ", mean_squared_error(test_y, predicted_y))
print('Variance score: %.2f' % r2_score(test_y, predicted_y))

mse_values.append(mean_squared_error(test_y, predicted_y))
variance_score.append(r2_score(test_y, predicted_y))

#%% Lasso模型
lasso_model = linear_model.Lasso()
lasso_model.fit(train_x, train_y)
predicted_y = lasso_model.predict(test_x)

print("Mean squared error: ", mean_squared_error(test_y, predicted_y))
print('Variance score: %.2f' % r2_score(test_y, predicted_y))

mse_values.append(mean_squared_error(test_y, predicted_y))
variance_score.append(r2_score(test_y, predicted_y))

#%% 生成图形函数
def generate_plot(title, ticks, dataset, color_number):
    colors = ["slateblue", "mediumseagreen", "tomato"]
    plt.figure(figsize=(8, 6))
    
    ax = plt.subplot()    
    ax.spines["top"].set_visible(False)   
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left() 
    
    plt.xticks(np.arange(len(ticks)), ticks, fontsize=10, rotation=30)
    plt.title(title, fontsize = 22)
    plt.bar(ticks, dataset, linewidth=1.2, color=colors[color_number])
#%% MSE
ticks = ["Linear Regression", "Decision Tree","SVM", "Random Forest","Lasso"]
generate_plot("Plot of MSE values", ticks, mse_values, 0)
#%% 方差
generate_plot("Plot of Variance scores", ticks, variance_score, 1)




#%% 分类算法
# 目标值转为二进制
binary_area_values = []
count = 0

for value in log_area_values:
    if(value == 0):
        binary_area_values.append(0)
    else:
        binary_area_values.append(1)
#%% 初始化参数
accuracy_values = []
# 拆分数据集
train_x, test_x, train_y, test_y = train_test_split(attribute_list, binary_area_values, test_size=0.15, random_state = 4)
#%% SGD模型
sgd = linear_model.SGDClassifier()
sgd.fit(train_x, train_y)
predicted_y = sgd.predict(test_x)

print("The predicted values are:", predicted_y)
print("The accuracy score is " + str(accuracy_score(test_y, predicted_y) * 100) + "%.")

accuracy_values.append(accuracy_score(test_y, predicted_y) * 100)
#%% 决策树模型
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(train_x, train_y)
predicted_y = decision_tree.predict(test_x)

print("The predicted values are:", predicted_y)
print("The accuracy score is " + str(accuracy_score(test_y, predicted_y) * 100) + "%.")

accuracy_values.append(accuracy_score(test_y, predicted_y) * 100)
#%% 贝叶斯模型
naive_bayes = GaussianNB()
naive_bayes.fit(train_x, train_y)
predicted_y = naive_bayes.predict(test_x)

print("The predicted values are:", predicted_y)
print("The accuracy score is " + str(accuracy_score(test_y, predicted_y) * 100) + "%.")

accuracy_values.append(accuracy_score(test_y, predicted_y) * 100)
#%% SVM模型
svm_model = svm.SVC(kernel='linear')
svm_model.fit(train_x, train_y)
predicted_y = svm_model.predict(test_x)

print("The predicted values are:", predicted_y)
print("The accuracy score is " + str(accuracy_score(test_y, predicted_y) * 100) + "%.")

accuracy_values.append(accuracy_score(test_y, predicted_y) * 100)
#%% 随机森林模型
random_forest = RandomForestClassifier()
random_forest.fit(train_x, train_y)
predicted_y = random_forest.predict(test_x)

print("The predicted values are:", predicted_y)
print("The accuracy score is " + str(accuracy_score(test_y, predicted_y) * 100) + "%.")

accuracy_values.append(accuracy_score(test_y, predicted_y) * 100)
#%% 计算精度图
ticks = ["SGD", "Decision tree", "Naive bayes", "SVM", "Random Forest"]
generate_plot("Plot of accuracy scores", ticks, accuracy_values, 2)

#%%