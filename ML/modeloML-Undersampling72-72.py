import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv(r"C:\Users\Home\Documents\USP2024\IC\Codigos\MachineLearning\ML\features_para_modelML.csv", sep=";", decimal=",")
print(df.head())
#print(df.columns.tolist())

X= df.drop(columns=["HasContrast"])
y= df["HasContrast"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print ("Distribuição original no treino:", y_train.value_counts().to_dict())

#RandomSampler
rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)
X_train_res, y_train_res= rus.fit_resample(X_train, y_train)
print("Após undersampling: ", y_train_res.value_counts().to_dict())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

#Treinar o modelo
model = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
model.fit(X_train_scaled, y_train_res)

y_predict = model.predict(X_test_scaled)

#Resultados
print("\nRelatorio Classificação:\n ", classification_report(y_test, y_predict))
print("\nMatriz de Confusão:\n ", confusion_matrix(y_test, y_predict))