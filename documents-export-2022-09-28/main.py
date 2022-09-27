import matplotlib.pyplot as plt
import pandas as pd
from sklearn import impute
from sklearn import decomposition
import Tradaboost as tr
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, f1_score, precision_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import type_of_target
import adaboost as ad
from sklearn import tree,linear_model
import csv
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import numpy as np
import xlwt


def append_feature(dataframe, istest):
    lack_num = np.asarray(dataframe.isnull().sum(axis=1))
    # lack_num = np.asarray(dataframe..sum(axis=1))
    if istest:
        X = dataframe.values
        X = X[:,2:X.shape[1]-1]

    else:
        X = dataframe.values
        X = X[:,2:X.shape[1]-1]
    total_S = np.sum(X, axis=1)
    var_S = np.var(X, axis=1)
    X = np.c_[X, total_S]
    X = np.c_[X, var_S]
    X = np.c_[X, lack_num]

    return X


train_df = pd.DataFrame(pd.read_csv("/Users/jiayue.li/Desktop/data_wind_prod.csv"))
train_df0 = pd.DataFrame(pd.read_csv("/Users/jiayue.li/Desktop/wt(1).csv"))
test_df0 = pd.DataFrame(pd.read_csv("/Users/jiayue.li/Desktop/wt(1).csv"))
train_df=train_df.fillna(value=0)
train_df0=train_df0.fillna(value=0)
test_df0=test_df0.fillna(value=0)
train_data_T = train_df.values
train_data_S0= train_df0.values


#len=train_data_T.shape[0]
#FalseValue1=np.zeros(len)
#for i in range(1,len,1):
    #if round(train_data_T[i,train_data_T.shape[1]-1],2) == round(train_data_T[(i-1),train_data_T.shape[1]-1],2):
        #FalseValue1[i] = 1
    #else:
        #FalseValue1[i] = 0
   # print(round(train_data_T[i-1,train_data_T.shape[1]-1],2))
   #print(round(train_data_T[i,train_data_T.shape[1]-1],2))
    #print(round(train_data_T[i,train_data_T.shape[1]-1],2) == round(train_data_T[(i-1),train_data_T.shape[1]-1],1))
#print(FalseValue1)

#for i in range(0,len,1):
    #train_data_T[i,train_data_T.shape[1]-1]=FalseValue1[i]

len1=train_data_S0.shape[0]
FalseValue2=np.zeros(len1)
for i in range(1,len1,1):
    if round(train_data_S0[i,train_data_S0.shape[1]-1],2) == round(train_data_S0[(i-1),train_data_S0.shape[1]-1],2):
        FalseValue2[i] = 1
    else:
        FalseValue2[i] = 0

for i in range(0,len1,1):
    train_data_S0[i,train_data_S0.shape[1]-1]=FalseValue2[i]
print(FalseValue2)
#(train_data_S0)
train_data_S=train_data_S0[7739:38692,]
test_data_S=train_data_S0[:7739,]
test_df=pd.DataFrame(test_data_S)
train_df1=pd.DataFrame(train_data_S)
resulta=np.sum(train_data_T==0)
resultb=np.sum(train_data_S==0)
resultc=np.sum(test_data_S==0)
#print(resulta)
#print(resultb)
#print(resultc)
print('data loaded.')

label_T = train_data_T[:,train_data_T.shape[1] - 1]
# trans_T = train_data_T[:, 1:train_data_T.shape[1] - 1]
trans_T = append_feature(train_df, istest=False)


label_S = train_data_S[:, train_data_S.shape[1] - 1]
# trans_S = train_data_S[:, 1:train_data_S.shape[1] - 1]
trans_S = append_feature(train_df1, istest=False)

test_data_no = test_data_S[:, 0]
#test_data_S = test_data_S[:, 1:test_data_S.shape[1]]
test_data_S = append_feature(test_df, istest=True)

print('data split end.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape)

# # 加上和、方差、缺失值数量的特征，效果有所提升
# trans_T = append_feature(trans_T, train_df)
# trans_S = append_feature(trans_S, train_df1)
# test_data_S = append_feature(test_data_S, test_df)
#
# print 'append feature end.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape

imputer_T = impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer_S = impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer_T.fit(trans_T,label_T)
imputer_S.fit(trans_S, label_S)

trans_T = imputer_S.transform(trans_T)
trans_S = imputer_S.transform(trans_S)

test_data_S = imputer_S.transform(test_data_S)

# pca_T = decomposition.PCA(n_components=50)
# pca_S = decomposition.PCA(n_components=50)
#
# trans_T = pca_T.fit_transform(trans_T)
# trans_S = pca_S.fit_transform(trans_S)
# test_data_S = pca_S.transform(test_data_S)

print('data preprocessed.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape)

X_train, X_test, y_train, y_test = model_selection.train_test_split(trans_S, label_S, test_size=0.33, random_state=42)
print("split.")
# feature scale
# scaler = preprocessing.StandardScaler()
# X_train = scaler.fit_transform(X_train, y_train)
# X_test = scaler.transform(X_test)
# print 'feature scaled end.'


#tradaboost


#adaboost
#ad = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5))
#ad.fit(X_train,y_train.astype(int))
#predict_results=ad.predict(X_test)
#y_test = y_test.astype('int64')
#fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=predict_results, pos_label=1)
#print('auc:', metrics.auc(fpr, tpr))

#decision tree
#clf = svm.LinearSVC(penalty='l2',dual=False,tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=100000)


#高斯贝叶斯
tree=GaussianNB()
tree.fit(X_train,y_train.astype(int))
#clf.fit(X_train,y_train.astype(int))
#predsvm=clf.predict(X_test)
predc=tree.predict(X_test)
#clf=tree.DecisionTreeClassifier(criterion="gini", max_depth=8,min_samples_leaf=5,max_features="log2", splitter="random")
#clf.fit(X_train, y_train.astype(int))
#pred=clf.predict(X_test)
y_test = y_test.astype('int64')
fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=predc, pos_label=1)
print('auc:', metrics.auc(fpr, tpr))
#acc=[]
#for i in range(1,100,1):
#    pred = tr.tradaboost(X_train, trans_T, y_train, label_T, X_test, 50)
#    y_test = y_test.astype('int64')
#    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=pred, pos_label=1)
#    print('auc:', metrics.auc(fpr, tpr))
    #acc.append(metrics.auc(fpr, tpr))
#print(acc)
#accdb=[]
#plt.figure()
#ax = plt.axes()
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)

#plt.xlabel('iters')
#plt.ylabel('accuracy')
#plt.plot(range(1,100,1), acc,color='blue', linestyle="solid", label="tradaboost accuracy")
#plt.legend()

#plt.title('Accuracy curve')
#plt.show()



#X_train, X_test, y_train, y_test = model_selection.train_test_split(trans_S0, label_S0, test_size=0.33, random_state=42)
#clf = svm.LinearSVC(penalty='l2',dual=False,tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=100000)
#clf.fit(X_train,y_train.astype(int))
#predsvm=clf.predict(X_test)
#y_test = y_test.astype('int64')
#fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=predsvm, pos_label=1)
#plt.figure()
#plt.bar(range(1,2),metrics.auc(fpr, tpr))
#plt.xlabel('SVM')
#plt.ylabel('Accuracy')
#plt.title('Accuracy curve')
#plt.show

#clf=tree.DecisionTreeClassifier(criterion="gini", max_depth=8,min_samples_leaf=5,max_features="log2", splitter="random")
#clf.fit(X_train, y_train.astype(int))
#pred=clf.predict(X_test)
#y_test = y_test.astype('int64')
#print(pred)
#plt.figure()


# 读入数据
plt.style.use('ggplot')
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Times New Roman'],
    'font.size':15
}
plt.rcParams.update(params)
ax=plt.axes()
name=[u'Trada',u'RF',u'NB',u'Tree',u'KNN',u'SVM']
first=[0.86319,0,0,0,0,0]
second=[0,0.7726,0,0,0,0]
third=[0,0,0.5,0,0,0]
forth=[0,0,0,0.636,0,0]
sixth=[0,0,0,0,0.56,0]
last=[0,0,0,0,0,0.5]
bar_width=0.45
ax.bar(name,first,label='Tradaboost',width=bar_width,align='center',color='#F08080')
ax.bar(0.6+bar_width,second,label='Decision Tree',width=bar_width,align='center',color='#FAA460')
ax.bar(1.6+bar_width,third,label='SVM',width=bar_width,align='center',color='#B0E0E6')
ax.bar(1.7+bar_width*3,forth,label='KNN',width=bar_width,align='center',color='#FFFACD')
ax.bar(2.2+bar_width*4,sixth,label='Random Forest',width=bar_width,align='center',color='#C0C0C0')
ax.bar(2.8+bar_width*5,last,label='Naive Bayes(BernoulliNB)',width=bar_width,align='center',color='#5F9EA0')

plt.xlabel(u'Models')
ax.set_xticks(name) # 设置x轴标签
ax.set_yticks(np.arange(0,1.05,0.05))
plt.ylabel('Accuracy')
#添加条形图的标题
plt.title('Accuracies Of Trada And Non-trada Models' )
plt.show()



#data = pd.DataFrame(pred)

#writer = pd.ExcelWriter('pred.xlsx')		# 写入Excel文件
#data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
#writer.save()
#writer.close()




