import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv(r"C:\Users\Home\Documents\USP2024\IC\Codigos\MachineLearning\ML\features_para_modelML.csv", sep=";", decimal=",")
print(df.head())
#print(df.columns.tolist())

X= df.drop(columns=["HasContrast"])
y= df["HasContrast"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print ("Distribuição original no treino:", y_train.value_counts().to_dict())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#RandomSampler e SMOTE
smote = SMOTE(sampling_strategy={1:150}, random_state=42, k_neighbors=5)
rus = RandomUnderSampler(sampling_strategy={0:150, 1:150}, random_state=42)

X_smote, y_smote = smote.fit_resample(X_train_scaled,y_train)
X_res, y_res= rus.fit_resample(X_smote, y_smote)
print("Após SMOTE + RUS no treino: ", y_res.value_counts().to_dict())

#Treinar o modelo
model = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
model.fit(X_res, y_res)

y_predict = model.predict(X_test_scaled)

#Resultados
print("\nRelatorio Classificação:\n ", classification_report(y_test, y_predict))
print("\nMatriz de Confusão:\n ", confusion_matrix(y_test, y_predict))