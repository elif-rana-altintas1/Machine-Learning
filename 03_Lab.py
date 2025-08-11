

#!!!   KARAR AĞACI YAPALIM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df=pd.read_csv(
    filepath_or_buffer=r"C:\Users\kursa\Desktop\homewok\05_ML\drug200.csv",

)

print(df.head())

print(df.shape)

print(df.columns)

#!  Features Matrix

X = df[["Age","Sex","BP","Cholesterol","Na_to_K"]].values

#! Target Matrix

y= df["Drug"].values

X[:,1] = LabelEncoder().fit(df["Sex"].unique()).fit_transform(X[:,1])
X[:,2] = LabelEncoder().fit(df["BP"].unique()).fit_transform(X[:,2])
X[:,3] = LabelEncoder().fit(df["Cholesterol"].unique()).fit_transform(X[:,3])

# print(X[:5])

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=42)

dt = DecisionTreeClassifier(
criterion = "entropy",
max_depth=3,
random_state=42
)
dt=dt.fit(X_train , y_train)


print(
    f"Node Count: {dt.tree_.node_count}\n"
    f"Max Depth: {dt.tree_.max_depth}\n"
)
importance = dt.feature_importances_
feature_names = ["Age","Sex","BP","Cholesterol","Na_to_K"]

# plt.figure(figsize=(8,5))

# plt.barh ( feature_names,importance,color="skyblue")
# plt.xlabel("Feature Importance")
# plt.ylabel("Features")
# plt.title("Features Importance İn Decision Tree")
# plt.grid(True)
# plt.show()

GR = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid={
        "max_depth":range(1, dt.tree_.max_depth+1 ,2),
        "max_features": range(1,len(dt.feature_importances_)+1)
    },
    scoring="accuracy",
    n_jobs=1
)
GR= GR.fit(X_train,y_train)

print(
    f"Node Count:{GR.best_estimator_.tree_.node_count}\n"
    f"Max depth:{GR.best_estimator_.tree_.max_depth}\n"
)

dt_1 = DecisionTreeClassifier(
criterion = "entropy",
max_depth=3,
random_state=42
)
dt_1=dt_1.fit(X_train , y_train)


print(
    f"Node Count: {dt.tree_.node_count}\n"
    f"Max Depth: {dt.tree_.max_depth}\n"
)

plt.figure(figsize=(12,8))
tree.plot_tree(
    dt_1,
    feature_names=feature_names,
    class_names=np.unique(y_train),
    filled=True
)

plt.show()