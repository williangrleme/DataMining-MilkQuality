# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score , cross_val_predict
from sklearn.preprocessing import StandardScaler,MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import seaborn as sns



# Calculate distance between two points
def minkowski_distance(a, b, p=2):    
    # Store the number of dimensions
    dim = len(a)    
    # Set initial distance to 0
    distance = 0
    
    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
        
    distance = distance**(1/p)    
    return distance


def knn_predict(X_train, X_test, y_train, y_test, k, p):    
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        
        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]
        
        # Append prediction to output list
        y_hat_test.append(prediction)
        
    return y_hat_test

def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    
    names = ['pH','Temprature','Taste','Odor','Fat','Turbidity', 'Colour', 'Grade'] 
    features = ['pH','Temprature','Taste','Odor','Fat','Turbidity', 'Colour'] 
    target = 'Grade'

    input_file = '0-Datasets/milknew_clear.csv' 
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                    names = names,       # Nome das colunas 
                    skiprows=1)          # Pula a primeira linha
                    
    target_names = ['low','Medium','high']
        
    
    # Separating out the features
    X = df.loc[:, features].values

    df['target'] = target

    #X = df.drop('target', axis=1)
    #y = df.target.values

    # Separating out the target
    y = df.loc[:,target]

    print("Total samples: {}".format(X.shape[0]))

    # Dividir os dados em treinamento e teste usando Holdout
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Normalizar os dados usando MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Criar e treinar o classificador KNN usando Holdout
    knn_holdout = KNeighborsClassifier(n_neighbors=6)
    knn_holdout.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste usando Holdout
    y_pred_holdout = knn_holdout.predict(X_test)

    # Avaliar a precisão e F1-score usando Holdout
    accuracy_holdout = accuracy_score(y_test, y_pred_holdout)
    f1_holdout = f1_score(y_test, y_pred_holdout, average='macro')
    print("Holdout Accuracy: {:.4f}%".format(accuracy_holdout * 100))
    print("Holdout F1 Score: {:.4f}".format(f1_holdout))

    # Plotar matriz de confusão usando Holdout
    plot_confusion_matrix(y_test, y_pred_holdout, classes=np.unique(y), title="Holdout Confusion Matrix")

    # Criar e treinar o classificador KNN usando Cross-Validation
    knn_cv = KNeighborsClassifier(n_neighbors=6)
    
    scores = cross_val_score(knn_cv, X, y, cv=10)

    # Fazer previsões usando Cross-Validation
    y_pred_cv = cross_val_predict(knn_cv, X, y, cv=10)

    # Avaliar a precisão e F1-score usando Cross-Validation
    accuracy_cv = np.mean(scores)
    f1_cv = f1_score(y, y_pred_cv, average='macro')
    print("Cross-Validation Accuracy: {:.4f}%".format(accuracy_cv * 100))
    print("Cross-Validation F1 Score: {:.4f}".format(f1_cv))
    # Plotar matriz de confusão usando Cross-Validation
    plot_confusion_matrix(y, y_pred_cv, classes=np.unique(y), title="Cross-Validation Confusion Matrix")



if __name__ == "__main__":
    main()