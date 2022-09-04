import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sklearn
from sklearn.model_selection import  *
import joblib
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, plot_confusion_matrix, classification_report
import time
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import csv


dataset = pd.read_csv("D:\Work_D\MSc_project\Code\data\\trim\DDoS_train_Thuesday-20-02-2018.csv" )
dataset2 = pd.read_csv("D:\Work_D\MSc_project\Code\data\\trim\DDoS_test_Wednesday-21-02-2018.csv" )

dataset = dataset.astype({"Protocol": str})
dataset = dataset.astype({"Label": str})
dataset = pd.get_dummies(dataset, columns=['Protocol'], drop_first=True)
dataset2 = dataset2.astype({"Protocol": str})
dataset2 = dataset2.astype({"Label": str})
dataset2 = pd.get_dummies(dataset2, columns=['Protocol'], drop_first=True)
dataset.insert(len(dataset.columns)-1, 'Label', dataset.pop('Label'))
dataset2.insert(len(dataset2.columns)-1, 'Label', dataset2.pop('Label'))


d1 = dataset.replace('Benign', 0)
# d2 = d1.replace('Brute Force -Web', 1)
# d1 = d2.replace('Brute Force -XSS', 1)
fin_dataset = d1.replace('DDoS attacks-LOIC-HTTP', 1)

# fin_dataset = dataset
# print(fin_dataset)
fin_dataset.drop(columns=['Flow ID','Src IP', 'Src Port','Dst IP','Dst Port', 'Timestamp','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Flow Byts/s','Flow Pkts/s'],axis=1, inplace=True)

# fin_dataset.drop(columns=['Dst Port', 'Timestamp','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Flow Byts/s','Flow Pkts/s'],axis=1, inplace=True)
fin_dataset.dropna(inplace= True)
fin_dataset.drop_duplicates(inplace=True)
fin_dataset.drop(fin_dataset.loc[fin_dataset["Label"] == "Label"].index, inplace=True)



# dataset2.drop(dataset2.loc[dataset2['Timestamp']], axis=1)

d11 = dataset2.replace('Benign', 0)
# d22 = d11.replace('Brute Force -Web', 1)
# d11 = d22.replace('Brute Force -XSS', 1)
f_dataset2 = d11.replace('DDOS attack-HOIC', 1)
fin_dataset2 = f_dataset2.replace('DDOS attack-LOIC-UDP', 1)
# fin_dataset2 = dataset2
# print(fin_dataset)
fin_dataset2.drop(columns=['Dst Port','Timestamp','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Flow Byts/s','Flow Pkts/s'],axis=1, inplace=True)
fin_dataset2.dropna(inplace= True)
fin_dataset2.drop_duplicates(inplace=True)
fin_dataset2.drop(fin_dataset2.loc[fin_dataset2["Label"] == "Label"].index, inplace=True)


fin_dataset.info()
fin_dataset2.info()

#import ML model

models = []
models.append(("LR", LogisticRegression(solver='lbfgs'),{ }))
models.append(("LDA", LinearDiscriminantAnalysis(),{}))
models.append(("DT", DecisionTreeClassifier(),{'max_depth': [i for i in range(1, 20)]} ))
models.append(("GNB", GaussianNB(),{'priors': [None],'var_smoothing': [0.00000001, 0.000000001, 0.00000001]}))
models.append(("MLPC", MLPClassifier(max_iter=100),{ 'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']}))
models.append(("SVC", SVC(),{"C": stats.uniform(2, 10), "gamma": stats.uniform(0.1, 1)}))
models.append(("RF", RandomForestClassifier(),{'n_estimators': [50, 75, 100, 125, 150]}))
models.append(("KNN", KNeighborsClassifier(),{'n_neighbors': [i for i in range(1, 31)]}))
numerical_cols = fin_dataset.columns[0:-1]
min_max_scaler = MinMaxScaler().fit(fin_dataset[numerical_cols])
fin_dataset[numerical_cols] = min_max_scaler.transform(fin_dataset[numerical_cols])
numerical_cols2 = fin_dataset2.columns[0:-1]
min_max_scaler2 = MinMaxScaler().fit(fin_dataset2[numerical_cols2])
fin_dataset2[numerical_cols2] = min_max_scaler2.transform(fin_dataset2[numerical_cols2])
y_train = np.array(fin_dataset.pop("Label")) # pop removes "Label" from the dataframe
X_train = fin_dataset.values
y_test = np.array(fin_dataset2.pop("Label")) # pop removes "Label" from the dataframe
X_test = fin_dataset2.values

model_name = []
fit_time = []
score_time = []
test_f1 = []
test_accuracy = []
test_precision = []
test_recall = []
hyper_value = []

for node_vlaue, node_name, hyperparameters in models:
    timer_start_fit = time.perf_counter()
    clf = RandomizedSearchCV(node_name, hyperparameters, random_state=0, cv=5, n_jobs=3)
    # clf = GridSearchCV(estimator=node_name, param_grid=hyperparameters, cv=5, verbose=1,n_jobs=-1 )
    clf.fit(X=X_train , y=y_train)
    best_estimate = clf.best_params_
    timer_end_fit = time.perf_counter()
    timer_start_estimate = time.perf_counter()
    fin_model = clf.best_estimator_
    evaluate = fin_model.predict(X_test)
    timer_end_estimate = time.perf_counter()
    score_accuracy = accuracy_score(y_test, evaluate)
    score_f1 = f1_score(y_test,evaluate)
    score_recall = recall_score(y_test,evaluate)
    score_precision = precision_score(y_test,evaluate)
    fit_time_s = timer_end_fit - timer_start_fit
    score_time_s = timer_end_estimate- timer_start_estimate
    model_name.append(node_vlaue)
    fit_time.append(fit_time_s)
    score_time.append(score_time_s)
    test_f1.append(score_f1)
    test_accuracy.append(score_accuracy)
    test_precision.append(score_precision)
    test_recall.append(score_recall)
    hyper_value.append(str(best_estimate))
    print("Model: ",node_vlaue,"Fit time: ", fit_time_s,"Score time: ", score_time_s, "Accuracy: ", score_accuracy,
          "Precision: ", score_precision, "Recall: ", score_recall, "F1: ", score_f1, "Hyperparameter: ", best_estimate)
    confusion_matrix_store = confusion_matrix(y_test, evaluate)
    print(confusion_matrix_store)
    plot_confusion_matrix(fin_model, X_test, y_test, cmap="cividis")
    plt.show()
    print(classification_report(y_test, evaluate, digits=5))
    str_value = str("trained_models/"+node_vlaue+"-DDoS2.pkl")
    joblib.dump(fin_model, str_value)

df = pd.DataFrame()

df["Model_Name"] = model_name
df["fit_time"] = fit_time
df["Score_time"] = score_time
df["F1_score"] = test_f1
df["Accuracy"] = test_accuracy
df["Precision"] = test_precision
df["Recall"] = test_recall
df["Hyperparameter"] = hyper_value
print(df)
df.to_pickle("./_New2_DDoS_attack.pki")


# algorithm_name = []
# graph = plt.figure()
# graph.suptitle('Comparing Algorithm Accuracy')
# ax= graph.add_subplot(111)
# plt.boxplot(finalresults)
# for i in range(len(models)):
#     algorithm_name.append(models[i][0])
# ax.set_xticklabels(algorithm_name)
#
# plt.show()