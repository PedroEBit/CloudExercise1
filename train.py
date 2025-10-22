# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Leitura dos dados
df = pd.read_csv("data/sample.csv")

X = df.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
y = df["diagnosis"].map({'M': 1, 'B': 0})
target_names = ['Benigno (B)', 'Maligno (M)']

# 2. Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinamento do modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Avaliação
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=target_names)

# 5. Salvamento do relatório
with open("report.txt", "w") as f:
    f.write(report)