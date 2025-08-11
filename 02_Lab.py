
#!KNN(K-Nearst Neighbhood)

import pandas as pd

df = pd.read_pickle(
    filepath_or_buffer=r"C:\Users\kursa\Desktop\homewok\05_ML\churndata.pkl",
)
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

df.drop(
    columns=["id","phone","total_revenue","cltv","churn_score"],
    axis=1,
    inplace=True
)
print(df.columns)


#! Hangi sütunda kaç tane unique value var
print(
    [
        [col, len(df[col].unique())] for col in df.columns
    ]
)

df_unique = pd.DataFrame(
    data=[ [col, len(df[col].unique())] for col in df.columns ],
    columns=["Feature Name" , "Unique Values"]
).set_index("Feature Name")

print(df_unique)


#!Binary değerleri listeye alalım 

binary_features = list(df_unique[df_unique["Unique Values"]== 2 ].index)
print(binary_features)

#* Categorical değerleri listeye allaım 

categorical_feature = list(
    df_unique[
        (df_unique["Unique Values"] > 2) & (df_unique["Unique Values"] <= 6)
    ].index
)
print(categorical_feature)


#* Categorical featuresları saptadık bunlardan sıralı ve sırasız olanları belirleyelim

print(
    [
        [item , list(df[item].unique())] for item in categorical_feature
    ]
)

ordinal_features = ["satisfaction","contract"]

numeric_features = ["gb_mon","monthly"]
df["months"] = pd.cut(df["months"], bins=5)

ordinal_features.append("months")

#! Data Prepocessing

from sklearn.preprocessing import LabelBinarizer,LabelEncoder,OrdinalEncoder

for column in binary_features:
    df[column] = LabelBinarizer().fit_transform(df[[column]]) 

for column in categorical_feature:
    df[column] = LabelEncoder().fit_transform(df[[column]])  

for column in ordinal_features:
    df[column] = OrdinalEncoder().fit_transform(df[[column]])

print(df.head())
 

#! verilerin çarpıklığına bakalım
from scipy.stats import skew

for col in df.columns:
    print(
        f"======================\n"
        f"{col} name and skewness value : {skew(df[col])}\n"
        f"============================================\n"
    )
#! Verileri scale ederken pipeline kullanalım
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
standard_cols = ["backup","protection"]
robut_cols = ["offer","gb_mon","security","support","payment","churn_value"]
min_max_cols = ["months","multiple","contract"]

#* ColumnTransform ile pipline kuralım

preprpcessor = ColumnTransformer(
    transformers=[
        ("robust",RobustScaler(),robut_cols),
        ("minmax",MinMaxScaler(),min_max_cols),
        ("standard",StandardScaler(),standard_cols),
    ],
    remainder="passthrough" #yukarıda belirttiğimiz sütunlar dışındakilere işlem yapma demek
)

#* dönüşümü uygulayalım
sclaed_data=preprpcessor.fit_transform(df)

#* sütun isimleri verelim 
new_columns = (
    robut_cols + min_max_cols + standard_cols +
    [col for col in df.columns if col not in robut_cols + min_max_cols + standard_cols]
)

scaled_df = pd.DataFrame(
    data=sclaed_data,
    columns=new_columns
)

print(scaled_df)


from sklearn.model_selection import train_test_split

y , X=df["churn_value"],df.drop(columns=["churn_value"])

X_train , X_test , y_train , y_test = train_test_split (X , y ,train_size=0.8,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , classification_report , f1_score

knn=KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(X_train , y_train)
y_pred = knn.predict(X_test)



print(
    f"Classification report \n"
    f"=============================\n"
    f"{classification_report(y_test, y_pred)}"
    f"====================================\n"
    f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}\n"
    f"=================================\n"
    f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}"
)


#! En doğru komşu sayısını saptayalım
max_k=40
f1_scores=list()
error_rates=list()

for k in range(1, max_k):
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',
        metric='euclidean',
        p=2
    )

    knn = knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    f1_scores.append(
        (k, f1_score(y_test, y_pred))
    )

    error_rates.append(
        (k, 1 - accuracy_score(y_test, y_pred))
    )

f1_result_df = pd.DataFrame(
    data=f1_scores,
    columns=['K', 'F1 Score']
)

error_rate_df = pd.DataFrame(
    data=error_rates,
    columns=['K', 'Error Rate']
)

print(
    f'F1 Score Report\n'
    f"===========================\n"
    f"{f1_result_df.sort_values(by="F1 Score",ascending=False).head()}\n"
    f"===================================\n"
    
    
    )

print(
    f"error rate report\n"
    f"===========================\n"
    f"{error_rate_df.sort_values(by="Error Rate", ascending=True).head()}\n"
    f"===========================\n"
)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("talk")
sns.set_style("ticks")

plt.figure(dpi=300)

ax = f1_result_df.set_index("K").plot(color="blue",figsize=(14,7), linewidth=2)

ax.set(xlabel="K",ylabel="F1 Score")
ax.set_xticks(range(1, max_k, 2 ))
plt.grid(True)
plt.title("KNN F1 Score")
plt.show()
