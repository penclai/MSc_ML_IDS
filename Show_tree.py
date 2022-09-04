import pandas as pd
from sklearn import tree
import joblib
import matplotlib.pyplot as plt
import graphviz
import pydotplus
import os
from IPython.display import Image, display
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
from subprocess import call
import pydotplus


model = joblib.load("./trained_models/RF-DoS.pkl")
train = pd.read_csv('D:\Work_D\MSc_project\Code\\data\\trim\DoS_train_Thursday-15-02-2018.csv')
train.drop(columns=['Dst Port','Timestamp','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags','Flow Byts/s','Flow Pkts/s'],axis=1, inplace=True)
# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)
#
# # tree.plot_tree(model, class_names=train.columns)
# tree.plot_tree(model, class_names=train.columns, filled=True,rounded=True, ax=axes)
# plt.show()
# plt.savefig('tree.png')
# estimator = model.estimators_[5]

export_graphviz(model.estimators_[5],
                out_file='tree.dot',
                class_names = train.columns,
                rounded = True, proportion = False,
                precision = 2, filled = True)
# (graph,) = pydotplus.graph_from_dot_file('D:/Work_D/MSc_project/Code/tree.dot')
# graph.write_png('somefile.png')
call(['dot', '-Tpng', 'D:/Work_D/MSc_project/Code/tree.dot', '-o', 'tree.png', '-Gdpi=600'])
Image(filename = 'tree.png')







# dot_data = tree.export_graphviz(
#     model,
#     out_file=None,
#     feature_names=train.columns
# )
#
# graphviz.Source(dot_data, format="png")
# graph = pydotplus.graph_from_dot_data(dot_data)
# plt = Image(graph.create_png())
# display(plt)
# graph.write_png('./decision_tree.png')