import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\Home\Documents\USP2024\IC\Codigos\MachineLearning\ML\features_para_modelML.csv", sep=";", decimal=",")
#print(df.head())
#print(df.columns.tolist())

X= df.drop(columns=["HasContrast"])
y= df["HasContrast"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo com peso balanceado
logreg = LogisticRegression(
    class_weight='balanced',
    max_iter=500,
    solver='lbfgs'
)

logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório Classificação:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
