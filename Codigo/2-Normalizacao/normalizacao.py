import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/milknew_clear.csv'
    names = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
    features = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
    target = 'Grade'
    df = pd.read_csv(input_file, names=names)
    ShowInformationDataFrame(df, "Dataframe original")

    x = df.loc[:, features].values
    y = df.loc[:, [target]].values

    # Min-max 
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data=x_minmax, columns=features)
    normalized2Df = pd.concat([normalized2Df, df[[target]]], axis=1)
    ShowInformationDataFrame(normalized2Df, "Base de dados normalizada")

    # PCA
    pca = PCA(n_components=2)  # Defina o número de componentes principais desejados
    principal_components = pca.fit_transform(x_minmax)

    # Cria um novo dataframe para armazenar as componentes principais
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[[target]]], axis=1)

    # Exibe os atributos mais influentes em cada componente principal
    show_pca_attributes(pca, features)

    # Visualização do PCA
    visualize_pca(pca_df)

    # Salva o gráfico do PCA em um arquivo JPEG
    save_pca_plot(pca_df)


def ShowInformationDataFrame(df, message=""):
    print(message + "\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")


def show_pca_attributes(pca, attribute_names):
    print("Atributos mais influentes em cada componente principal:")
    for i, pc in enumerate(pca.components_):
        sorted_indices = np.argsort(pc)[::-1]
        sorted_attributes = [attribute_names[j] for j in sorted_indices]
        print(f"PC{i+1}: {', '.join(sorted_attributes[:3])}")


def visualize_pca(df):
    targets = df['Grade'].unique()
    colors = ['r', 'g', 'b', 'y', 'm']  # Defina cores para cada classe

    plt.figure(figsize=(8, 6))
    for target, color in zip(targets, colors):
        indices = df['Grade'] == target
        plt.scatter(df.loc[indices, 'PC1'], df.loc[indices, 'PC2'], c=color, s=50, label=target)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA')
    plt.legend()
    plt.show()


def save_pca_plot(pca_df):
    plt.figure(figsize=(8, 6))
    targets = pca_df['Grade'].unique()
    colors = ['r', 'g', 'b', 'y', 'm']  # Defina cores para cada classe

    for target, color in zip(targets, colors):
        indices = pca_df['Grade'] == target
        plt.scatter(pca_df.loc[indices, 'PC1'], pca_df.loc[indices, 'PC2'], c=color, s=50, label=target)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA')
    plt.legend()
    plt.savefig('pca_plot.jpg')  # Salva o gráfico como um arquivo JPEG
    plt.close()


if __name__ == "__main__":
    main()
