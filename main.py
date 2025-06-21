import numpy as np
import pandas as pd
import missingno as mscno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

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

plt.figure(figsize=(10,8))
sns.histplot(df[num_col], kde = True)
plt.title('Histograma para Identificar Outliers Numéricos')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.show()

for notnum_col in object_columns:
    print(f'Analisando colunas não numéricas: {notnum_col}')

plt.figure(figsize=(10,8))
sns.histplot(df[notnum_col], kde = True)
plt.title('Histograma para Identificar Outliers Não Numéricos')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.show()

#Limpeza dos dados:
##Eliminar as colunas com muitos elementos ausentes e colunas desnecessárias
print('Nome das colunas:')
print(df.columns.tolist())
print("-" * 30)

df_clean = df.drop(columns=['Spay/Neuter Status','Daily Walk Distance (miles)','Synthetic','Seizures',
                            'Play Time (hrs)','Annual Vet Visits','Average Temperature (F)'])
print(df_clean)
print("-" * 30)

##Quantidade de colunas restantes com elementos ausentes
print(df_clean.isna().any(axis=1).sum())
print("-" * 30)

##Atualizar as colunas com o novo data frame
float_columns = df_clean.select_dtypes(include=np.number)
print(f'Colunas numéricas: {float_columns}')
print('-'*30)
object_columns = df_clean.select_dtypes(exclude=np.number)
print(f'Colunas não numéricas: {object_columns}')
print('-'*30)

##Preencher valores ausentes com a média nas colunas numéricas
for num_col in float_columns:
    columns_mean = df_clean[num_col].mean()
    df_clean[num_col] = df_clean[num_col].fillna(columns_mean)
print(df_clean[num_col])
print('-'*30)

##Preencher valores ausentes com a moda nas colunas não numéricas
for obj_col in object_columns:
    columns_mode = df_clean[obj_col].mode()[0]
    df_clean[obj_col] = df_clean[obj_col].fillna(columns_mode)
print(df_clean[obj_col])
print('-' * 30)

print('Dados nulos em cada coluna:')
print(df_clean.isnull().sum())
print('-'*30)

#Modelos preditivos
##Definição do x (features) e Y
x = df_clean.drop('Healthy', axis=1)
y = df_clean['Healthy']

##Separar colunas numéricas e não numéricas do X
x_num = x.select_dtypes(include=np.number).columns.tolist()
x_obj = x.select_dtypes(exclude=np.number).columns.tolist()

##Separar em treinamento e teste
SEED = 20
x_train_raw, x_test_raw, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=SEED, stratify=y)

x_train_num = x_train_raw[x_num]
x_test_num = x_test_raw[x_num]

x_train_obj = x_train_raw[x_obj]
x_test_obj = x_test_raw[x_obj]

##Transformar dados não numéricos em numéricos para a ML compreender
le = LabelEncoder()
oe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
x_train_encoded = oe.fit_transform(x_train_obj)
x_test_encoded = oe.fit_transform(x_test_obj)
y_encoded = le.fit_transform(y)

print(x_test_encoded)
print(x_train_encoded)
print(y_encoded)

scale = MinMaxScaler()
x_train_num_scaled = scale.fit_transform(x_train_num)
x_test_num_scaled = scale.transform(x_test_num)

##Concatenar matrizes esparsas
x_train = hstack([x_train_num_scaled, x_train_encoded])
x_test = hstack([x_test_num_scaled, x_test_encoded])

print(f"Shape de x_train final: {x_train.shape}")
print(f"Shape de x_test final: {x_test.shape}")

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test,predict) * 100
print(f'Acurácia da árvore de decisão: {accuracy}')

model = SVC()
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test,predict) * 100
print(f'Acurácia do SVM: {accuracy}')

model = DummyClassifier()
model.fit(x_train, y_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test,predict) * 100
print(f'Acurácia do DummyClassifier: {accuracy}')

