from multiprocessing.reduction import duplicate
from sys import float_repr_style

import numpy as np
import pandas as pd
import missingno as mscno
import seaborn as sns
import matplotlib.pyplot as plt

#Importar arquivo csv
path = r"synthetic_dog_breed_health_data.csv"
df = pd.read_csv(path)

#Análise do DataFrame:
##Primeiras linhas do df
print(df.head())
print('-'*30)

##Informações das colunas com quantidade de valores, dados nulos e tipo de dados
print(df.info())
print('-'*30)

##Variedade de itens de cada coluna
print(df.nunique())
print('-'*30)

##Quantidade de dados nulos em cada coluna
print('Dados nulos em cada coluna:')
print(df.isnull().sum())
print('-'*30)

##Verificar dados faltantes
print(mscno.matrix(df))
print('-'*30)

##Verificar dados duplicados
duplicated = df.duplicated().sum()
print(f'Quantidade de dados duplicados: {duplicated}')
print('-'*30)

##Tipologia das colunas
print(df.dtypes)
print('-'*30)

##Separar colunas float e object
float_columns = df.select_dtypes(include=np.number)
print(f'Colunas numéricas: {float_columns}')
print('-'*30)
object_columns = df.select_dtypes(exclude=np.number)
print(f'Colunas não numéricas: {object_columns}')
print('-'*30)

##Verificar presença de outliers
for num_col in float_columns:
    print(f'Analisando colunas numéricas: {num_col}')
    print('-'*30)

# plt.figure(figsize=(10,8))
# sns.histplot(df[num_col], kde = True)
# plt.title('Histograma para Identificar Outliers Numéricos')
# plt.xlabel('Valor')
# plt.ylabel('Frequência')
# plt.show()

for notnum_col in object_columns:
    print(f'Analisando colunas não numéricas: {notnum_col}')

# plt.figure(figsize=(10,8))
# sns.histplot(df[notnum_col], kde = True)
# plt.title('Histograma para Identificar Outliers Não Numéricos')
# plt.xlabel('Valor')
# plt.ylabel('Frequência')
# plt.show()

#Limpeza dos dados:
##Eliminar as colunas com muitos elementos ausentes
df_clean = df.drop(columns=['Breed','Spay/Neuter Status','Daily Activity Level','Owner Activity Level'])
print(df_clean)
print("-" * 30)

##Quantidade de colunas restantes com elementos ausentes
print(df_clean.isna().any(axis=1).sum())
print("-" * 30)

##Preencher valores ausentes com a média nas colunas numéricas
for num_col in float_columns:
    columns_mean = df_clean[num_col].mean()
    df_clean[num_col] = df_clean[num_col].fillna(columns_mean)
print(df_clean[num_col])
print('-'*30)

print('Dados nulos em cada coluna:')
print(df_clean.isnull().sum())
print('-'*30)

##Preencher valores ausentes com a moda nas colunas não numéricas
# for obj_col in object_columns:
#     columns_mode = df_clean[obj_col].mode()
#     df_clean[obj_col] = df_clean[obj_col].fillna(columns_mode)
# print(df_clean[obj_col])
# print('-' * 30)


