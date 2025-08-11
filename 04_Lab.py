
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
df=pd.read_csv(
    filepath_or_buffer=r"C:\Users\kursa\Desktop\homewok\05_ML\Cust_Segmentation.csv",
)

print(df.head())

df.drop(
    columns=["Customer Id","Address","Defaulted"],
    axis=1,
    inplace=True
)

print(df.head())

df_init = df.copy()

print(df.describe().T)

print(df.shape)

print(df.info())

corr_matrix = df.corr()

# for x in range ( corr_matrix.shape[0]):
#     corr_matrix.iloc[x,x]=0.0 # loc etiket mantığıyla iloc index mantığıyla çalışır

# fig, ax = plt.subplots(fig_size=(16,10)) 
# ax= sns.heatmap(
#     corr_matrix,
#     annot=True,
#     linewidths=0.5,
#     fmt="2f",
#     cmap="YIGnBu"
# )
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom= + 0.5 , top - 0.5)

corr_mat= df.corr()

for x in range ( corr_mat.shape[0]):
    corr_mat.iloc[x,x]=0.0 # loc etiket man

print(corr_mat) 


corr_max=corr_mat.abs().max().to_frame()
corr_id_max = corr_mat.abs().idxmax().to_frame()

pairwise_features_corr= pd.merge(
    right=corr_max,
    left=corr_id_max,
    on=corr_max.index,
    how="inner"
)
pairwise_features_corr.rename(
    columns={
        "key_0":"Feature_One",
        "0_x":"Feature_two",
        "0_y":"Correlation"
    },
    inplace="True"

)
pairwise_features_corr.sort_values(
    by="Correlation",
    ascending=False,
    inplace=True
)
pairwise_features_corr.reset_index(
    drop=True,
    inplace=True
)
print(pairwise_features_corr)

z_score_df=df.copy()
for col in df.columns:
    z_score_df[col]=stats.zscore(z_score_df[col])

for col in z_score_df.columns:
    outliers = z_score_df[z_score_df[col].abs() > 3] 
    print(f"{col} için aykırı değer sayısı:{len(outliers)}\n")

scaler_strategies = {
    'Robust': ['Other Debt', 'Card Debt', 'Income', 'Years Employed', 'DebtIncomeRatio'],
    'Standard': ['Edu'],
    'MinMax': ['Age']
}

preprocessor = ColumnTransformer(
    transformers=[
        ('robust', RobustScaler(), scaler_strategies['Robust']),
        ('standard', StandardScaler(), scaler_strategies['Standard']),
        ('minmax', MinMaxScaler(), scaler_strategies['MinMax']),
    ],
    remainder='passthrough'
)

numeric_cols = df.columns
df_scaled = pd.DataFrame(
    preprocessor.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index
)

df = pd.concat([df.drop(numeric_cols, axis=1), df_scaled], axis=1)
print(df.head())

km_list= list()
result=list()

for clust in range(1,11):
    km= KMeans(
        n_clusters=clust,
        random_state=42,
        n_init=10
    )
    km.fit(df)

    result.append(
        {
            "clusters": clust,
            "inertia": km.inertia_,
            "model":km
        }
    )
km_df = pd. DataFrame(result).set_index("clusters")

print(km_df)

plt.figure(figsize=(14,8))

line = plt. plot ( km_df.index,km_df["inertia"],"bo-", markersize=8, linewidth=2, label ="inertia") 
plt.xticks(km_df.index)

for x, y in zip (km_df.index , km_df["inertia"]):
    plt.text(x,y+20, f"{y:.1f}",ha="center",va="bottom",fontsize=9,color = "darkorange")

plt.axvline(x=4,color="r",linestyle="--",alpha=0.7,label="suggested k=4") 
plt.axvline(x=5,color="g",linestyle="--",alpha=0.7,label="suggested k=5") 

plt.xlabel(xlabel="Number of clusters ( k)",fontsize=12,labelpad=10)
plt.ylabel(ylabel="inertia",fontsize=12,labelpad=10)
plt.title(label="elbow method for optimal k with detaild annotations",fontsize=14,pad=20)
plt.grid(True,alpha=0.3)
plt.legend(fontsize=10)

plt.xlim(0.5 , len(km_df)+0.5)
plt.ylim(0, km_df["inertia"].max()*1.1)

percentage_drops = [(km_df['inertia'].iloc[i - 1] - km_df['inertia'].iloc[i]) / km_df['inertia'].iloc[i - 1] for i in range(2, len(km_df) - 1)]

for i, pct in enumerate(percentage_drops, start=2):
    plt.text(i, km_df['inertia'].iloc[i] * 1.05, f'{pct:.1%}', ha='center', color='purple')

# plt.tight_layout()
# plt.show()

k_means=KMeans(
    init="k-means++",
    n_clusters=4,
    n_init=10
)
k_means.fit(df)
lables = k_means.labels_
df_init["Clus_KM"]= lables
print(df_init.head())
print(df_init["Clus_KM"].value_counts())

