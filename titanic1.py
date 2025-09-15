import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Carregando os dados
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Função para transformar sexo em binário
def transformar_sexo(valor):
    if valor == "female":
        return 1
    else:
        return 0

# Criando a nova coluna no train
train["Sex_binario"] = train["Sex"].map(transformar_sexo)

# Selecionando variáveis
variaveis = ["Sex_binario", "Age"]

# Tratando valores ausentes em Age
train["Age"] = train["Age"].fillna(-1)
test["Age"] = test["Age"].fillna(-1)

# Preparando X e y
x = train[variaveis]
y = train["Survived"]

# Criando o modelo
modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelo.fit(x, y)

# Criando a coluna no test também
test["Sex_binario"] = test["Sex"].map(transformar_sexo)
x_prev = test[variaveis]

# Fazendo previsões
p = modelo.predict(x_prev)

# Criando o arquivo de submissão
submissao = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": p
})

submissao.to_csv("submission.csv", index=False)
print("Arquivo submission.csv criado com sucesso!")
