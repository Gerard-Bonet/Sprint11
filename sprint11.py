#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import math
from sklearn.cluster import KMeans
import statistics as stat
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA


# Seguimos el hilo argumental del [ejercicio 9.1](https://github.com/Gerard-Bonet/Sprint9Tasca1.git) y  [ejercicio 9.2](https://github.com/Gerard-Bonet/sprint9-tasca2-Aprenantatge-supervisat.git)
# para trabajarcon el ejercicio 10 . Primero cargamos el Data Set. 

# In[3]:


df = pd.read_csv("DelayedFlights.csv") # este es el conjunto de datos proporcionado en el ejercicio 
df.head(10)


# 0.  **Tratamiento de variables**. 
# 
# En este apartado vamos a hacer tratamiento de variables para luego aplicar a programas de Clustering 
# 
# Hemos dejado los enlaces del [ejercicio 9.1](https://github.com/Gerard-Bonet/Sprint9Tasca1.git) y  [ejercicio 9.2](https://github.com/Gerard-Bonet/sprint9-tasca2-Aprenantatge-supervisat.git) en que se explica los razonamientos para sleccionar una variable un otra, 
# 
# aunque volveremos a exolicar los motivos por los que se seleccionan variables, unas sí u otras no, no entraremos tan en detalle
# 

# 
# 
# a) Variable Unnamed 0 y Year, básicamente son un índice y el año de vuelos del 2008. Year es una constante. 
# Así que las eliminamos. 
# 
# b) las variables "UniqueCarrier', 'FlightNum','TailNum, 'Month', 'DayofMonth', 'DayOfWeek', 'Cancelled', 'CancellationCode', "Diverted", "Origin", "Dest" *son eliminadas por siguiente motivo. KMeans funciona calculando distancias en un espacio ndimensional, de volumen R^n. Si implementamos variables categóricas, por ejemplo binarias, o con más clases como los meses, los puntos van a estar concentrados en hiperplanos de dimensión n- 1. Si las variables no categóricas, están normalizadas o estandarizadas, la distancia entre hiperplanos será por lo general mayor , que la distancia entre puntos dentro de un hiperplano de dimensión n-1.* 
# 
# c) También eliminamos 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime' ya que tienen un 0.95 de correlación o más con Distance y aportan la misma información. ActualElapsed es el tiempo esperado total del vuelo( desembarco, salida, vuelo, más atterizaje), CRSElapsedTime es el mismo tiempo previsto, mientras que AirTime es el tiempo que el avión está en el aire y Distance la distancia recorrida en millas.
# 
# 

# In[4]:


df[["ActualElapsedTime", 'CRSElapsedTime', 'AirTime' , "Distance" ]].corr()


# In[5]:


df1= df.drop(["Unnamed: 0", "Year", "UniqueCarrier", 'FlightNum',"TailNum", "Month", "DayofMonth", 
              'DayOfWeek', 'Cancelled', 'CancellationCode', "Diverted", "Origin", "Dest","ActualElapsedTime", 'CRSElapsedTime', 'AirTime'], axis =1)
df1.head(10)


# Como vamos a empezar a transformar variables, vamos a hacer previamente un muestreo. 
# 

# In[6]:


df2= df1.sample (390000,random_state=55)


# Cómo vimos en los ejercicios 9.1 y 9.2, los valores NaN de Arrdelay, así como de otras variables  coincidían con aquellos que el vuelo había desviado o cancelado, valores imposibles de deducir ( los NaN) por el mismo concepto de cancelación o desvió. 
# por lo que eliminamos los valores NaN de ArrDelay

# In[7]:


df3=df2.dropna( subset=["ArrDelay"]).reset_index(drop=True)
df3.isna().sum()


# In[8]:


df3.shape


# 0.1. **Transformación de variables horarias**
# 
# En esta parte vamos a convertir las variables DepTime,	CRSDepTime,	ArrTime,	CRSArrTime en la función cíclica. 
# 
# Estas cuatro variables vienen en formato horario  hh:mm. Lo que haremos será contar todos los minutos transcurridos durante 
# 
# el día, siendo 0 minutos a las 00:00 y 1440 los minutos transcurridos durante el día a las 23:59.
# 
# Más adelante, en el apartado 0.4, transformaremos estas variables en cíclicas. 
# 

# In[9]:


# Primero de todo convierto las variables horarias en formato hora y para eso tienen que haber 4 digítos, que los relleno por la
#izquierda con ceros
# Primero tengo que convertir en entero las variables DepTime y ArrTime en enteros para evitar los decimales


df3['DepTime'] = df3['DepTime'].astype(int)


# In[10]:



df3['ArrTime'] = df3['ArrTime'].astype(int)


# In[11]:


#relleno por la izquierda con ceros
df3['DepTime'] = df3['DepTime'].astype(str).str.zfill(4)
df3['CRSDepTime'] = df3['CRSDepTime'].astype(str).str.zfill(4)
df3['ArrTime'] = df3['ArrTime'].astype(str).str .zfill(4)
df3['CRSArrTime'] = df3['CRSArrTime'].astype(str).str.zfill(4)
df3.head()


# In[12]:


# las convierto en formato horario( Nota: en un principio lo pasé a formato horario por si lo necesitaba para datetime, pero 
# al final opté por otro tipo de conversión)
df3['DepTime'] = df3['DepTime'].astype(str).str[:2]  + ':' + df3['DepTime'].astype(str).str[2:4] + ':00' 
df3['CRSDepTime'] = df3['CRSDepTime'].astype(str).str[:2] + ':' + df3['CRSDepTime'].astype(str).str[2:4] + ':00' 
df3['ArrTime'] = df3['ArrTime'].astype(str).str[:2] + ':' + df3['ArrTime'].astype(str).str[2:4]  + ':00'
df3['CRSArrTime'] = df3['CRSArrTime'].astype(str).str[:2] + ':' + df3['CRSArrTime'].astype(str).str[2:4] + ':00'


df3


# In[13]:


# creamos la función minutos, que divide la hora hh:mm con un Split, en una lista ("hh","mm"), reconvierte hh y mm en enteros,
# para luego pasarlos a minutos, y con la reconverión ya comentada aplica la función minutos()
def minutos(x):    
    x=x.split( sep=":")
    seg= 60*(int(x[0]))+(int(x[1]))
    
    return seg



dfhoras= df3[["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime"]]


# In[14]:


dfhoras_DT= dfhoras["DepTime"].apply(minutos)
dfhoras_CRSD=dfhoras["CRSDepTime"].apply(minutos)
dfhoras_AT=dfhoras["ArrTime"].apply(minutos)
dfhoras_CRSA=dfhoras["CRSArrTime"].apply(minutos)


# In[15]:


df4= df3.drop([ 'DepTime',
       'CRSDepTime', 'ArrTime', 'CRSArrTime'], axis=1) 


# In[16]:


# ahora añadimos las cuatro columnas nuevas

df5= pd.concat([df4, dfhoras_DT,dfhoras_CRSD , dfhoras_AT, dfhoras_CRSA], axis=1)
df5.columns


# In[17]:


df5[["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime"]].describe()# miramos como quedan para ver si hay alguna anomalía en 
# los máximos y mínimos


# In[18]:


df5.isna().sum()


# 0.2 **Valores NaN** 
# 
# En esta sección vamos a completar las variables del motivo del retraso que tiene varios NaN. 
# Como ya pudimos observar en el ejercicio 9.2, los vuelos con retrasos de menos de 14 minutos, tienen valores Nan
# 
# Carrier Delay es el retraso de la compañía 
# 
# WeatherDelay es el retraso por las condiciones climatológicas
# SecurityDelay es el retraso por cuestiones de seguridad
# 
# LateAircraftDelay es el retraso de la misma aeronave. 
# 
# Nas delay son los retraso causado por el Sistema Nacional del Espacio Aéreo (NAS)
# 
# por lo que vamos a asigarn 0 a los valores Nan
# 

# In[19]:


df_delay= df5[['ArrDelay','DepDelay','CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay',]]
df5_0= df5.drop(  ["CarrierDelay", 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay'], axis=1 )
delay_not_NAN= df_delay[['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay']].fillna(0.0)


# In[20]:


df5= pd.concat([df5_0, delay_not_NAN], axis =1)
df5.head(10)


# In[21]:


df5.isna().sum()


# 0.3 **dos nuevas variables.**
# 
# Se puede ver a simple vista que ArrDelay y DepDelay se obtienen de la resta entre (DepTime-CRSDepTime)y	(ArrTime-CRSArrTime), así que calculamos dos nuevas variables, para reducir la dimensionalidad. 
# 
# 
# Primero miramos el valor más bajo de ArrDelay, para poder hacer los cálculo correctamente 

# In[22]:


minimo=df[df["ArrDelay"]<0].sort_values("ArrDelay")# miramos los valores más bajos de ArrDelay
zmin=minimo["ArrDelay"].min()
zmin


# In[23]:


def rest(z):
    x=z[0]
    y=z[1]
    if (x < y) & ((x-y)<zmin): 
        t= (1440+x)-y
        return t
    else:
        t= x-y
        return t  
    
x11= df5[["DepTime","CRSDepTime" ]].apply(rest, axis=1)
x10= df5[["ArrTime","CRSArrTime" ]].apply(rest,axis=1)

x10=x10.rename("X10")
x11=x11.rename("X11")
x10.describe()


# In[24]:


x11.describe()


# Sean **x10** y **x11** definidas por 
# 
# **x10= df6["ArrTime"]-df6["CRSArrTime"]**
# 
# **x11= df6["DepTime"]-df6["CRSDepTime"]**
# y tienen una relación  lineal con ArrDelay y DepDeplay respectivamente 
# 

# In[25]:


x1= pd.concat([x10,x11,df5["ArrDelay"],df5["DepDelay"] ],axis=1)
x1.corr()


# In[26]:


# remodelamos el data set con las dos nuevas variables 
df5b=df5.drop(['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime'], axis=1)
df6=pd.concat([df5b, x10,x11], axis =1)
df6.head(10)


# In[27]:


df6.shape


# En el siguiente paso vamos a analizar la multicolinealidad. Para ver si alguna variable tiene un multicolinealidad muy elevada
# VIFi = 1/ (1 -$Ri^2$) donde Ri es el coeficiente de  determinación de la regresión lineal 

# In[38]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(X):
    vifDF = pd.DataFrame()
    vifDF["variables"] = X.columns
    vifDF["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vifDF

round(vif(df6),2)


# 0.4 **Reducción de la dimensionalidad por PCA**
# 
# Vamos a reducir las dimensiones para realizar el Cluster y bajar las 12 dimensiones lo máximo posible sin perder Varianza.
# Primero de todo vamos a estandarizar los datos para que no haya una componente que domine sobre el resto. Luego miraremos la 
# perdida de Varianza Explicada
# 
# 

# In[116]:


scaler = StandardScaler()
scaler.fit(df6)
df7=pd.DataFrame(scaler.transform(df6), columns=df6.columns)
df7.head()


# In[117]:




pca =PCA().fit(df7)
varexp= pca.explained_variance_ratio_.cumsum()
plt.plot(varexp)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


#  por lo visto, escogeremos 6 compnentes principales, en que la ganancia de Varianza posterior es menor. 
#  

# In[132]:


pca2= PCA(n_components=6).fit(df7)


# In[133]:


df8= pd.DataFrame(pca2.transform(df7),columns=["componente principal 1", "componente principal 2", "componente principal 3", 
                                               "componente principal 4", "componente principal 5","componente principal 6" ])
df8.head(10)


# 1. **Ejercicio 1**: Agrupa los vuelos por KMeans

# Primero de todos vamos a buscar el número de agrupaciones por el método del codo. 

# In[134]:


sse = []
for k in range(1, 13):
    model = KMeans(n_clusters=k, init="random", )
    model.fit(df8)
    sse.append(model.inertia_)

plt.plot(range(1, 13), sse)
plt.xticks(range(1, 13))
plt.xlabel("Número de agrupaciones")
plt.ylabel("SSE")
plt.show()


# Con el gráfico tal cual tengo dudas si el número agrupaciones debe ser 2, 3 o 5,  así que voy a buscarlo de otra manera

# In[135]:


from kneed import KneeLocator
loccodo = KneeLocator(range(1, 13), sse, curve="convex", direction="decreasing" )

loccodo.elbow


# Una vez conocidos los agrupamientos que necesitamos, vamos a transformar el Data Set

# In[136]:


kmeans= KMeans(n_clusters=3, init="random",n_init=10,max_iter=1000,random_state=57  )
kmeans.fit(df8)
etiquetas= kmeans.predict(df8)


# In[137]:


etiq= pd.DataFrame(etiquetas, columns=["clase"])
etiq.value_counts()


# In[138]:


df9=pd.concat([df7,etiq],axis=1)
df9.head(25)


# In[139]:


indice = ['ArrDelay', 'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut', 'CarrierDelay',
       'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'X10',
       'X11']

for l in indice : 
    sns.displot(df9, x=l, col="clase")


# In[140]:


# contando la proporción de miembros de Clase 
etiq.value_counts()


# 1. Podemos concluir que uno de los motivos de clasificación han sido ArrDelay, DepDelay, X10 y X11, ya que meintras el resto de variable tienen los mismos dominios para cada clase  ,  las variables ArrDelay, DepDelay, X10 y X11  tienen dominios claramente diferenciados en función de la clase a la que pertence  

# In[141]:


# miremos el describe() tras estandarizar el DataSet, para ver juego ha tenido la distancia y posibles Outlayers
pd.DataFrame(scaler.transform(df6), columns=df6.columns).describe()


# Podemos observar que lo Outlayers no son excesivamente grandes 

# vamos a compar también las componentes principales. 
# 

# In[143]:


df10 = pd.concat([df9, df8], axis=1)
df10.head()


# In[145]:


indice2= ["componente principal 1" , "componente principal 2","componente principal 3", "componente principal 4",
         "componente principal 5", "componente principal 6"]
for s in indice2 : 
    sns.displot(df10, x=s, col="clase")


# Podemos observar que la variable que determina la clasificación es la **Componente principal 1**, ya que su dominio varía 
# en función de la clase. Vamos a ver como son las combinaciones lineales de las variables para sacar los componentes principales
# 
# 
# 

# In[147]:


pd.DataFrame( data    = pca2.components_,columns = df7.columns,
    index   = ["componente principal 1" , "componente principal 2","componente principal 3", "componente principal 4",
         "componente principal 5", "componente principal 6"])


# Conclusiones. 
# Tras hacer las agrupaciones con Kmeans podemos decir que : 
# 
# a)Podemos determinar por observación que si la componente principal 1 (CP1) está en el rango CP1>5 pertence a la clase 0, si 
# CP1<0 pertence a clase 1, y si   0>CP1>5 , pertence a la clase 2
# 
# b) la clase principal 1, es una combinación lineal de las variable del DataSet. Donde el coeficiente constante de ArrDelay, 
# DepDelay, X10 y X11 valen aproximadamente 0.45, 
# 
# c) estas 4 variables tienen una alta correlación,
# 
# 

# In[150]:


df7[["ArrDelay","DepDelay", "X10","X11"]].corr()


# d) Así que es posible que la PCA las haya comprido en la componente principal 1, dándoles poco peso en las otras 5 componentes 
# principales. Dando explicación porque estás variables son las afectados en su  dominio, en función de la clase. Clasificando los grupos en función de ArrDelay( o "DepDelay", "X10","X11", ya que están muy correlacionadas. )

# In[ ]:





# In[ ]:





# In[ ]:




