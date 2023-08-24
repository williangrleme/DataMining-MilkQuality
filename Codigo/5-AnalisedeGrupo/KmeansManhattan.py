import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import homogeneity_score

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/milknew_clear.csv'
    names = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
    features = ['pH', 'Temperature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
    target = 'Grade'
    df = pd.read_csv(input_file, names=names)

    x = df.loc[:, features].values
    y = df.loc[:, [target]].values

    # Min-max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data=x_minmax, columns=features)
    normalized2Df = pd.concat([normalized2Df, df[[target]]], axis=1)

    # PCA
    pca = PCA(n_components=2)  # Defina o número de componentes principais desejados
    principal_components = pca.fit_transform(x_minmax)

    # Cria um novo dataframe para armazenar as componentes principais
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[[target]]], axis=1)

    # Aplicando K-means aos dados após a aplicação do PCA com distância Manhattan
    k = 3
    centroids, labels = kmeans_with_manhattan(principal_components, k)

    # Cálculo do coeficiente de forma (Compactness Score)
    compactness_score = calculate_compactness(principal_components, labels, centroids)
    print("Coeficiente de forma (Compactness Score):", compactness_score)

    # Cálculo da homogeneidade
    homogeneity = homogeneity_score(y.flatten(), labels)
    print("Homogeneidade:", homogeneity)

    # Visualização dos clusters obtidos
    visualize_clusters(principal_components, labels)

def kmeans_with_manhattan(data, k):
    # Inicialização aleatória dos centróides
    centroids_idx = np.random.choice(len(data), k, replace=False)
    centroids = data[centroids_idx]

    # Atribuição inicial de rótulos
    distances = cdist(data, centroids, metric='cityblock')
    labels = np.argmin(distances, axis=1)

    # Loop até convergência
    while True:
        # Atualização dos centróides
        for i in range(k):
            if np.any(labels == i):  # Verifica se o cluster não está vazio
                centroids[i] = np.mean(data[labels == i], axis=0)
            else:
                centroids[i] = data[np.random.choice(len(data))]

        # Atualização dos rótulos
        new_distances = cdist(data, centroids, metric='cityblock')
        new_labels = np.argmin(new_distances, axis=1)

        # Verificar convergência
        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

    return centroids, labels

def calculate_compactness(data, labels, centroids):
    compactness = 0.0
    for i, label in enumerate(labels):
        centroid = centroids[label]
        compactness += np.linalg.norm(data[i] - centroid)
    return compactness / len(data)

def visualize_clusters(data, labels):
    plt.figure(figsize=(8, 6))

    # Plota cada grupo separadamente e atribui uma cor diferente a cada grupo
    for group in np.unique(labels):
        plt.scatter(data[labels == group, 0], data[labels == group, 1], label=f'Grupo {group+1}', cmap='viridis')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusters K-means após PCA (Distância Manhattan)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
