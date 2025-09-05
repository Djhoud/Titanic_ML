import pandas as pd
import numpy as np 

# Carregando os dados
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Exibindo as primeiras linhas (opcional)
print(train.head()) 

# Criando a função de transformação de sexo
def transformar_sexo(valor):
    if valor == 'female':
        return 1
    else:
        return 0

# Criando a nova coluna 'sex_binario' usando a função
train['sex_binario'] = train['Sex'].map(transformar_sexo)

# Preparando o modelo (note que a variável 'sex_binario' agora existe)
from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
variaveis = ['sex_binario', 'age'] # 'age' precisa ser tratada se tiver valores ausentes