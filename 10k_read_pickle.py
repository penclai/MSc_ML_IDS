import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

dataset = pd.read_pickle("./DoS_dataset_testing500_0.3.pki")
# dataset2 = pd.read_pickle("./testing - Copy.pki")
# Infiltration_1_df = pd.read_pickle("./DoS_dataset_testing_0.4.pki")
# Infiltration_2_df = pd.read_pickle("./DoS_dataset_testing_0.4.pki")
# Infiltration_merge = pd.merge(Infiltration_1_df, Infiltration_2_df, on="Model_Name", how="outer").groupby("Model_Name").mean().reset_index()
print(dataset)
# dataset2.drop(5)
# print(dataset2)
# dataset2.to_csv("./testing2.csv", sep=',', encoding='utf-8')
dataset.info()


dfplot = dataset[["Model_Name", "Recall"]].copy()
# dfplot2 = dataset2[[ "Recall"]].copy()
# dfplot['Recall_2']= dfplot2

X_axis = len(dfplot)

fig = plt.figure()
fig.suptitle('Bruteforce dataset')
ax = fig.add_subplot(111)
width = 0.4
dfplot.plot(kind='bar', x='Model_Name', y=["Recall"], ax=ax)
# plt.ylim(0.992,0.998)
plt.show()

dfplot= dataset[["Model_Name", "Accuracy"]].copy()
# dfplot2 = dataset2[[ "Accuracy"]].copy()
# dfplot['Accuracy_2']= dfplot2
print(dfplot)
fig = plt.figure()
fig.suptitle('Bruteforce dataset')
ax = fig.add_subplot(111)
width = 0.4
dfplot.plot(kind='bar', x='Model_Name', y=["Accuracy"], ax=ax)
# plt.ylim(0.980,1)
plt.show()

dfplot = dataset[["Model_Name", "Precision"]].copy()
# dfplot2 = dataset2[[ "Precision"]].copy()
# dfplot['Precision_2']= dfplot2

fig = plt.figure()
fig.suptitle('Bruteforce dataset')
ax = fig.add_subplot(111)
width = 0.4
dfplot.plot(kind='bar', x='Model_Name', y=["Precision"], ax=ax)
# plt.ylim(0.975,1)
plt.show()

dfplot = dataset[["Model_Name", "F1_score"]].copy()
# dfplot2 = dataset2[[ "F1_score"]].copy()
# dfplot['F1_score_2']= dfplot2

fig = plt.figure()
fig.suptitle('Bruteforce dataset')
ax = fig.add_subplot(111)
width = 0.4
dfplot.plot(kind='bar', x='Model_Name', y=["F1_score"], ax=ax)
# plt.ylim(0.985,1)
plt.show()

dfplot = dataset[["Model_Name", "fit_time"]].copy()
# dfplot2 = dataset2[[ "Score_time"]].copy()
# dfplot['Score_time_2']= dfplot2
fig = plt.figure()
fig.suptitle('Bruteforce dataset')
ax = fig.add_subplot(111)
width = 0.4
dfplot.plot(kind='bar', x='Model_Name', y=["fit_time"], ax=ax)

plt.show()

# fig = plt.figure()
# fig.suptitle('10k 2018 Infiltration dataset')
# ax = fig.add_sutplot(111)
#
# width = 0.4
# dfplot.plot(kind='box', x='Model_tag', ax=ax)
#
# fig = plt.figure()
# fig.suptitle('10k 2018 Infiltration dataset')
# ax = fig.add_sutplot(111)
#
# width = 0.4
# dfplot.plot(kind='box', x='Model_tag', ax=ax)
#
# fig = plt.figure()
# fig.suptitle('10k 2018 Infiltratio dataset')
# ax = fig.add_sutplot(111)
#
# width = 0.4
# dfplot.plot(kind='box', x='Model_tag', ax=ax)
