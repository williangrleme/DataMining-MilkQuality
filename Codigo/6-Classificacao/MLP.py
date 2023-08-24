from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main():
    names = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
    features = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
    target = 'Grade'

    input_file = '0-Datasets/milknew_clear.csv'
    df = pd.read_csv(input_file, names=names, skiprows=1)

    target_names = ['low', 'Medium', 'high']

    # Separating out the features
    X = df.loc[:, features].values

    # Normalizar os dados usando Min-Max scaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Separating out the target
    y = df.loc[:, target]

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Criar uma instância do MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=2000, shuffle=False, random_state=1)

    # Treinar o modelo usando Holdout
    clf.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste (Holdout)
    predictions_holdout = clf.predict(X_test)

    # Calcular acurácia e F1-score para Holdout
    accuracy_holdout = accuracy_score(y_test, predictions_holdout)
    f1_holdout = f1_score(y_test, predictions_holdout, average='macro')
    print("Holdout Metrics:")
    print("Accuracy: {:.4f}%".format(accuracy_holdout * 100))
    print("F1 Score: {:.4f}".format(f1_holdout))
    # Plotar matriz de confusão usando Holdout
    plot_confusion_matrix(y_test, predictions_holdout, classes=np.unique(y), title="Holdout Confusion Matrix")

    # Realizar Cross Validation
    scores_cv = cross_val_score(clf, X, y, cv=10)
    predictions_cv = cross_val_predict(clf, X, y, cv=10)

    # Calcular acurácia e F1-score para Cross Validation
    accuracy_cv = accuracy_score(y, predictions_cv)
    f1_cv = f1_score(y, predictions_cv, average='macro')
    print("Cross Validation Metrics:")
    print("Accuracy: {:.4f}%".format(accuracy_cv * 100))
    print("F1 Score: {:.4f}".format(f1_cv))
    # Plotar matriz de confusão usando Cross-Validation
    plot_confusion_matrix(y, predictions_cv, classes=np.unique(y), title="Cross-Validation Confusion Matrix")

if __name__ == "__main__":
    main()
