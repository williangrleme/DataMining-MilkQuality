import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/milknew_clear.csv'
    names = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
    df = pd.read_csv(input_file, names=names)

    # Medidas de tendência central
    print("---------------------")
    print("MEDIDAS DE TENDENCIA CENTRAL")
    print("---------------------")
    print("Temperatura")
    print("Media:")
    print(df['Temprature'].mean())
    print("Mediana:")
    print(df['Temprature'].median())
    print("Ponto médio:")
    print((df['Temprature'].max() + df['Temprature'].min()) / 2)
    print("Moda:")
    print(df['Temprature'].mode())
    print("---------------------")

    print("pH")
    print("Media:")
    print(df['pH'].mean())
    print("Mediana:")
    print(df['pH'].median())
    print("Ponto médio:")
    print((df['pH'].max() + df['pH'].min()) / 2)
    print("Moda:")
    print(df['pH'].mode())
    print("---------------------")

    print("Cor")
    print("Media:")
    print(df['Colour'].mean())
    print("Mediana:")
    print(df['Colour'].median())
    print("Ponto médio:")
    print((df['Colour'].max() + df['Colour'].min()) / 2)
    print("Moda:")
    print(df['Colour'].mode())
    print("---------------------")

    print("Gosto")
    print("Moda:")
    print(df['Odor'].mode())
    print("---------------------")

    print("Odor")
    print("Moda:")
    print(df['Taste'].mode())
    print("---------------------")

    print("Gordura")
    print("Moda:")
    print(df['Fat'].mode())
    print("---------------------")

    print("Turbidez")
    print("Moda:")
    print(df['Turbidity'].mode())
    print("---------------------")

    # Medidas de dispersão
    print("MEDIDAS DE DISPERSAO")
    print("---------------------")
    print("Temperatura")
    print("Amplitude:")
    print(df['Temprature'].max() - df['Temprature'].min())
    print("Desvio padrão:")
    print((df['Temprature'].std())/100)
    print("Variância:")
    print(df['Temprature'].var())
    print("Coeficiente de variação:")
    print((df['Temprature'].std() / df['Temprature'].mean()) * 100)
    print("---------------------")

    print("pH")
    print("Amplitude:")
    print(df['pH'].max() - df['pH'].min())
    print("Desvio padrão:")
    print((df['pH'].std())/100)
    print("Variância:")
    print(df['pH'].var())
    print("Coeficiente de variação:")
    print((df['pH'].std() / df['pH'].mean()) * 100)
    print("---------------------")

    print("Cor")
    print("Amplitude:")
    print(df['Colour'].max() - df['Colour'].min())
    print("Desvio padrão:")
    print((df['Colour'].std())/100)
    print("Variância:")
    print(df['Colour'].var())
    print("Coeficiente de variação:")
    print((df['Colour'].std() / df['Colour'].mean()) * 100)
    print("---------------------")

    # Medidas de posição relativa
    print("MEDIDAS DE POSICAO RELATIVA")
    print("---------------------")
    print("Temperatura")
    print("Quartis:")
    print(df['Temprature'].quantile([0.25, 0.5, 0.75]))
    z_score = (df['Temprature'] - df['Temprature'].mean()) / df['Temprature'].std()
    print("Escore-z:")
    print(z_score)
    print("---------------------")

    # Boxplot
    plt.boxplot(df['Temprature'])
    plt.xlabel('Temperatura')
    plt.title('Boxplot - Temperatura')
    plt.show()

    # Medidas de associação
    print("MEDIDAS DE ASSOCIACAO")
    print("---------------------")
    print("Covariância entre Cor e Turbidez:")
    cov = df['Colour'].cov(df['Turbidity'])
    print(cov)
    correlation = df['Colour'].corr(df['Turbidity'])
    print("Correlação entre Cor e Turbidez:")
    print(correlation)
    plt.scatter(df['Colour'], df['Turbidity'])
    plt.xlabel('Cor')
    plt.ylabel('Turbidez')
    plt.title('Gráfico de Dispersão - Cor vs Turbidez')
    plt.show()
    sns.regplot(x='Colour', y='Turbidity', data=df)
    plt.xlabel('Cor')
    plt.ylabel('Turbidez')
    plt.title('Gráfico de Dispersão - Cor vs Turbidez')
    plt.show()

if __name__ == "__main__":
    main()
