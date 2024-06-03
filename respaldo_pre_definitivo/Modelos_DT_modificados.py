# Librerias basicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Funciones para normalizar data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Funciones para arbol de decision 
from sklearn.tree import DecisionTreeClassifier

# Funciones para modelo KNN
from sklearn.neighbors import KNeighborsClassifier

# Funciones para modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

# Funciones para modelo de Bernoulli - Naives Bayes
from sklearn.naive_bayes import BernoulliNB

# Funciones para modelo SVM
from sklearn.svm import LinearSVC
import warnings

# Funciones para mostrar esteticamente las variables de desempeño de forma tabulada
from tabulate import tabulate # requiere de pip install tabulate 
from prettytable import PrettyTable # requiere de pip install prettytable
from sklearn.tree import plot_tree # requiere de graphviz

# Funciones para calcular medidas de desempeño
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# /////////////////////////////////////////////////////////////////////////////////////////

# Carga de datos
df = pd.read_csv('/home/benzo/Mineria_Datos/P2/creditcard_2023.csv')

# Variables para los atributos con los nombres de cada uno
X = df.drop(columns='Class', axis=1)
Y_clase = df['Class']
X_atributos = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

# Reduccion del dataset en un 10% por falta de recursos 
reduction_fraction = 0.1
X_reduced, _, Y_reduced, _ = train_test_split(X_atributos, Y_clase, train_size=reduction_fraction, stratify=Y_clase, random_state=42)
#print(X_reduced)
#print(Y_reduced)

# Division de datos reducidos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y_reduced, test_size=0.2, random_state=42)
#print("Tamaño del conjunto de datos reducido:", X_reduced.shape, Y_reduced.shape)

# //////////////////////////////////////////////////////////////////////////////////////////

# Estandarizacion de caracteristicas o atributos
scaler = StandardScaler()
X_train_scaled = X_train
X_test_scaled = X_test

# //////////////////////////////////////////////////////////////////////////////////////////

# IMPLEMENTACION DEL MODELO DE ARBOL DECISION

# seccion de entrenamiento del arbol decision sin poda
dt_no_prune = DecisionTreeClassifier(random_state=42)
dt_no_prune.fit(X_train_scaled, y_train)

# Predicciones de árbol de decisión sin poda
y_test_pred_no_prune = dt_no_prune.predict(X_test_scaled)
#y_train_pred = dt_no_prune.predict(X_train_scaled)

# Evaluación del modelo de arbol decision sin poda
accuracy_no_prune = accuracy_score(y_test, y_test_pred_no_prune)
precision_no_prune = precision_score(y_test, y_test_pred_no_prune)
recall_no_prune = recall_score(y_test, y_test_pred_no_prune)
f1_no_prune = f1_score(y_test, y_test_pred_no_prune)
conf_matrix_no_prune = confusion_matrix(y_test, y_test_pred_no_prune)

# Entrenamiento del modelo de arbol decision podado
dt_prune = DecisionTreeClassifier(random_state=42, max_depth=5)
#dt_prune = DecisionTreeClassifier(random_state=42, ccp_alpha=0.005)
dt_prune.fit(X_train_scaled, y_train)

# Predicciones arbol decision podado
y_test_pred_prune = dt_prune.predict(X_test_scaled)

# Evaluación del modelo de árbol de decisión con poda
accuracy_prune = accuracy_score(y_test, y_test_pred_prune)
precision_prune = precision_score(y_test, y_test_pred_prune)
recall_prune = recall_score(y_test, y_test_pred_prune)
f1_prune = f1_score(y_test, y_test_pred_prune)
conf_matrix_prune = confusion_matrix(y_test, y_test_pred_prune)

# Visualizacion de arboles
plt.figure(figsize=(20, 10))
plot_tree(dt_no_prune, feature_names=df.drop(columns='Class').columns, class_names=["No Fraude", "Fraude"], filled=True)
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(dt_prune, feature_names=df.drop(columns='Class').columns, class_names=["No Fraude", "Fraude"], filled=True)
plt.show()

# Medidas de desempeño de arboles decision
metrics_table = PrettyTable()
metrics_table.field_names = ["Metric", "Value (No Prune)", "Value (Prune)"]

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values_no_prune = [accuracy_no_prune, precision_no_prune, recall_no_prune, f1_no_prune]
values_prune = [accuracy_prune, precision_prune, recall_prune, f1_prune]

for metric, value_no_prune, value_prune in zip(metrics, values_no_prune, values_prune):
    metrics_table.add_row([metric, f"{value_no_prune:.6f}", f"{value_prune:.6f}"])

print("---------------------------------------------------")
print("Árbol de Decisión - Medidas de desempeño")
print(metrics_table)
print("---------------------------------------------------\n")

# Matrices confusion de arboles decision podados y sin poda
conf_matrix_table_no_prune = PrettyTable()
conf_matrix_table_no_prune.field_names = ["", "Predicted 0", "Predicted 1"]
conf_matrix_table_no_prune.add_row(["Actual 0", conf_matrix_no_prune[0][0], conf_matrix_no_prune[0][1]])
conf_matrix_table_no_prune.add_row(["Actual 1", conf_matrix_no_prune[1][0], conf_matrix_no_prune[1][1]])

conf_matrix_table_prune = PrettyTable()
conf_matrix_table_prune.field_names = ["", "Predicted 0", "Predicted 1"]
conf_matrix_table_prune.add_row(["Actual 0", conf_matrix_prune[0][0], conf_matrix_prune[0][1]])
conf_matrix_table_prune.add_row(["Actual 1", conf_matrix_prune[1][0], conf_matrix_prune[1][1]])

print("Árbol de Decisión Sin Poda - Matriz de Confusión")
print(conf_matrix_table_no_prune)
print("---------------------------------------------------\n")

print("Árbol de Decisión Con Poda - Matriz de Confusión")
print(conf_matrix_table_prune)
print("---------------------------------------------------\n")

# //////////////////////////////////////////////////////////////////////

# IMPLEMENTACION DE KNN

# Entrenamiento del modelo KNN
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# Predicciones de KNN
y_pred_knn = knn.predict(X_test_scaled)

# Evaluación del modelo KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average="macro")
recall_knn = recall_score(y_test, y_pred_knn, average="macro")
f1_knn = f1_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Impresión de medidas de desempeño del KNN
metrics_table_knn = PrettyTable()
metrics_table_knn.field_names = ["Metric", "Value"]
metrics_table_knn.add_row(["Accuracy", f"{accuracy_knn:.6f}"])
metrics_table_knn.add_row(["Precision", f"{precision_knn:.6f}"])
metrics_table_knn.add_row(["Recall", f"{recall_knn:.6f}"])
metrics_table_knn.add_row(["F1 Score", f"{f1_knn:.6f}"])

conf_matrix_table_knn = PrettyTable()
conf_matrix_table_knn.field_names = ["", "Predicted 0", "Predicted 1"]
conf_matrix_table_knn.add_row(["Actual 0", conf_matrix_knn[0][0], conf_matrix_knn[0][1]])
conf_matrix_table_knn.add_row(["Actual 1", conf_matrix_knn[1][0], conf_matrix_knn[1][1]])

print("---------------------------------------------------")

print("KNN - Medidas de Evaluación del Modelo")
print(metrics_table_knn)
print("---------------------------------------------------")
print("KNN - Matriz de Confusión")
print(conf_matrix_table_knn)
print("---------------------------------------------------")

# Validación cruzada para KNN
intr = range(0, 10)
scores = []
for i in intr:
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y_reduced, test_size=0.20, random_state=i)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    scores.append(accuracy_score(y_test, y_pred))

print("KNN - Puntuaciones de Validación Cruzada")
for i, score in enumerate(scores):
    print(f"Iteración {i+1}: {score:.6f}")

print("\n")
# //////////////////////////////////////////////////////////////////////

# IMPLEMENTACION DE RANDOM FOREST

# Entrenamiento del modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Predicciones de Random Forest
y_pred_rf = rf.predict(X_test_scaled)

# Evaluación del modelo Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average="macro")
recall_rf = recall_score(y_test, y_pred_rf, average="macro")
f1_rf = f1_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Impresión de medidas de desempeño del Random Forest
metrics_table_rf = PrettyTable()
metrics_table_rf.field_names = ["Metric", "Value"]
metrics_table_rf.add_row(["Accuracy", f"{accuracy_rf:.6f}"])
metrics_table_rf.add_row(["Precision", f"{precision_rf:.6f}"])
metrics_table_rf.add_row(["Recall", f"{recall_rf:.6f}"])
metrics_table_rf.add_row(["F1 Score", f"{f1_rf:.6f}"])

conf_matrix_table_rf = PrettyTable()
conf_matrix_table_rf.field_names = ["", "Predicted 0", "Predicted 1"]
conf_matrix_table_rf.add_row(["Actual 0", conf_matrix_rf[0][0], conf_matrix_rf[0][1]])
conf_matrix_table_rf.add_row(["Actual 1", conf_matrix_rf[1][0], conf_matrix_rf[1][1]])

print("---------------------------------------------------")
print("Random Forest - Medidas de Evaluación del Modelo")
print(metrics_table_rf)
print("---------------------------------------------------")
print("Random Forest - Matriz de Confusión")
print(conf_matrix_table_rf)
print("---------------------------------------------------")

# Validación cruzada para Random Forest
scores_rf = []
for i in intr:
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y_reduced, test_size=0.20, random_state=i)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    scores_rf.append(accuracy_score(y_test, y_pred))

print("Random Forest - Puntuaciones de Validación Cruzada")
for i, score in enumerate(scores_rf):
    print(f"Iteración {i+1}: {score:.6f}")

# Visualización de los árboles del modelo Random Forest
plt.figure(figsize=(20,10))
for i in range(10):  # Cambia el rango según cuántos árboles quieras visualizar
    plt.subplot(2, 5, i+1)  # Ajusta el número de filas y columnas según tus necesidades
    plot_tree(rf.estimators_[i], filled=True)
    plt.title("Tree {}".format(i+1))
plt.show()

print("\n")
# //////////////////////////////////////////////////////////////////////

# IMPLEMENTACION DE BERNOULLI

# Entrenamiento del modelo Bernoulli Naive Bayes
bern_nb = BernoulliNB(binarize=True)
bern_nb.fit(X_train_scaled, y_train)

# Predicciones de Bernoulli Naive Bayes
y_pred_bern = bern_nb.predict(X_test_scaled)

# Evaluación del modelo Bernoulli Naive Bayes
accuracy_bern = accuracy_score(y_test, y_pred_bern)
precision_bern = precision_score(y_test, y_pred_bern, average="macro")
recall_bern = recall_score(y_test, y_pred_bern, average="macro")
f1_bern = f1_score(y_test, y_pred_bern)
conf_matrix_bern = confusion_matrix(y_test, y_pred_bern)

# Impresión de medidas de desempeño del Bernoulli Naive Bayes
metrics_table_bern = PrettyTable()
metrics_table_bern.field_names = ["Metric", "Value"]
metrics_table_bern.add_row(["Accuracy", f"{accuracy_bern:.6f}"])
metrics_table_bern.add_row(["Precision", f"{precision_bern:.6f}"])
metrics_table_bern.add_row(["Recall", f"{recall_bern:.6f}"])
metrics_table_bern.add_row(["F1 Score", f"{f1_bern:.6f}"])

conf_matrix_table_bern = PrettyTable()
conf_matrix_table_bern.field_names = ["", "Predicted 0", "Predicted 1"]
conf_matrix_table_bern.add_row(["Actual 0", conf_matrix_bern[0][0], conf_matrix_bern[0][1]])
conf_matrix_table_bern.add_row(["Actual 1", conf_matrix_bern[1][0], conf_matrix_bern[1][1]])

print("---------------------------------------------------")
print("Bernoulli Naive Bayes - Medidas de Evaluación del Modelo")
print(metrics_table_bern)
print("---------------------------------------------------")
print("Bernoulli Naive Bayes - Matriz de Confusión")
print(conf_matrix_table_bern)
print("---------------------------------------------------\n")

# Generación de la matriz de confusión en forma de gráfico
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_bern, annot=True, cmap="Blues", fmt="d")
plt.title("Bernoulli Naive Bayes - Matriz de Confusión")
plt.xlabel("Predicciones")
plt.ylabel("Valores reales")
plt.show()

# //////////////////////////////////////////////////////////////////////

# IMPLEMENTACION DE SVM

# Entrenamiento del modelo SVM
warnings.filterwarnings("ignore", category=FutureWarning)

svm = LinearSVC(max_iter=10000)
svm.fit(X_train, y_train)

# Predicciones del modelo SVM
y_pred = svm.predict(X_test)

# Evaluación del modelo SVM
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Impresión de medidas de desempeño
metrics_table = PrettyTable()
metrics_table.field_names = ["Metric", "Value"]
metrics_table.add_row(["Accuracy", f"{accuracy:.6f}"])
metrics_table.add_row(["Precision", f"{precision:.6f}"])
metrics_table.add_row(["Recall", f"{recall:.6f}"])
metrics_table.add_row(["F1 Score", f"{f1:.6f}"])

conf_matrix_table = PrettyTable()
conf_matrix_table.field_names = ["", "Predicted 0", "Predicted 1"]
conf_matrix_table.add_row(["Actual 0", conf_matrix[0][0], conf_matrix[0][1]])
conf_matrix_table.add_row(["Actual 1", conf_matrix[1][0], conf_matrix[1][1]])

print("---------------------------------------------------")
print("SVM - Model Evaluation Metrics")
print(metrics_table)
print("---------------------------------------------------")
print("SVM - Confusion Matrix")
print(conf_matrix_table)
print("---------------------------------------------------")

# Validación cruzada
intr = range(0, 10)
scores = []
for i in intr:
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y_reduced, test_size=0.20, random_state=i)
    svm = LinearSVC(max_iter=10000)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print("SVM - Cross Validation Scores: ")
for i, score in enumerate(scores):
    print(f"Iteration {i+1}: {score:.6f}")

print("\n")
