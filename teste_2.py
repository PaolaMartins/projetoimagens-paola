import pandas as pd
import numpy 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\Home\Documents\USP2024\IC\Codigos\MachineLearning\features_para_modelML.csv", sep=";", decimal=",")
print(df.head())
#print(df.columns.tolist())

X= df.drop(columns=["HasContrast"])
y= df["HasContrast"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#SMV precisa dados normalizados
#    ------------------print(X.dtypes)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Treinar o modelo
model = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
model.fit(X_train_scaled, y_train)

y_predict = model.predict(X_test_scaled)

#Resultados
print("Acuracia: ", accuracy_score(y_test, y_predict))
print("\nRelatorio Classificação:\n ", classification_report(y_test, y_predict))
print("\nMatriz de Confusão:\n ", confusion_matrix(y_test, y_predict))