

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from sklearn import linear_model
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from scipy.stats import skew , boxcox
from scipy.stats.mstats import normaltest
from sklearn.model_selection import train_test_split
df= pd.read_csv(
    filepath_or_buffer=r"C:\Users\kursa\Desktop\homewok\05_ML\FuelConsumption.csv",
)
print(df.columns)


#?veri setimizi daraltalım


cdf = df[["ENGINESIZE","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB","CYLINDERS","CO2EMISSIONS"]]
print(cdf.head())

#todo tahmin edilecek değer olan karbon emisyonuna en çok etki eden özelliği korelasyon ile saptayalım 
#* korelasyon sonucu 1 çıkarsa pozitif güçlü ilişki 
#* -1 çıkarsa negatif güçlü ilişki 
#* 0 çıkarsa ilişki yok 

#? yani +1 yakınsadıkça pozitif güçlü ilişki -1 yakınsadıkça negatif güçlü ilişki anlamına gelir

print(cdf.corr()["CO2EMISSIONS"].sort_values(ascending=False))

#!Verilerimiz ne kadar çarpık 

for col in cdf.columns:
    print(
        f"{col} skewness value : {skew(cdf[col]):.4f}\n"
        f"=========================================\n"
        f"\n"
    )


#! Verilerimizi normailize edelim.

df_box = cdf.copy()
for col in cdf.columns:
    df_box[col], _ = boxcox(cdf[col])


print(df_box)

print(df_box.corr()["CO2EMISSIONS"].sort_values(ascending=False))

for col in df_box.columns:
    print(
        f"{col} skewness value : {skew(df_box[col]):.4f}\n"
        f"=========================================\n"
        f"\n"
    )


train_df , test_df = train_test_split(df_box,train_size=0.8,test_size=0.2, random_state=42)

print(
    f"Train set:{train_df.shape}\n"
    f"Test set:{test_df.shape}"
)

#! Model Oluşturalım

regression= linear_model.LinearRegression()
train_x=np.asanyarray(train_df[["FUELCONSUMPTION_COMB"]])
train_y=np.asanyarray(train_df[["CO2EMISSIONS"]])

#!Modeli train edelim

regression.fit(train_x , train_y)

#*simple linear regression bize coefficient ve intercept olmak üzere iki tane katsayı döner
coef = regression.coef_[0][0]
inter = regression.intercept_[0]
print(
    f"Coefficient:{coef:.2f}\n"
    f"Intercept: { coef:.2f}"
)

# Nokta grafiği
plt.scatter(
    train_df["FUELCONSUMPTION_COMB"],
    train_df["CO2EMISSIONS"],
    c="r"
)

# Regresyon doğrusu
plt.plot(
    train_x,
    coef * train_x + inter,
    c="green"
)

# Başlık ve eksen etiketleri
plt.title(
    "Relation between Fuel Consumption and CO2 Emissions",
    fontsize=20,
    color="blue"
)
plt.xlabel("Fuel Consumption (L/100km)", fontsize=15, color="b")
plt.ylabel("CO2 Emissions (g/km)", fontsize=15, color="b")

# Grafiği göster
plt.grid()
plt.show()


test_x=np.asanyarray(train_df[["FUELCONSUMPTION_COMB"]])
test_y=np.asanyarray(train_df[["CO2EMISSIONS"]])

#* Modeli test edelim

test_pred = regression.predict(test_x)

result_df=pd.DataFrame(
    {
        "Real Values":test_y.flatten(),
        "Estimated Values":test_pred.flatten()
    }
)
print(result_df.head())

#*****R2 Score
r2_result = r2_score(test_y,test_pred)
print(f"R2 score:{r2_result:.2f}")

#***** Mean Absolute Error
print(f"Mean Absolute Error:{mean_absolute_error(test_y,test_pred):.2f}")
print(f"Mean Squared Error:{mean_squared_error(test_y,test_pred):.2f}")
