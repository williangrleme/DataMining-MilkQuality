import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/milknew_clear.csv'
    names = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
    df = pd.read_csv(input_file, names=names)

    #pH
    plt.hist(df['pH'], bins=10, edgecolor='black')
    plt.xlabel('pH')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Frequência do pH')
    x_ticks = np.arange(df['pH'].min(), df['pH'].max() + 0.5, 0.5) #Mostrar a cada 0,5 no eixo x
    plt.xticks(x_ticks)
    plt.show()
    plt.savefig('histograma_pH.jpeg')
    plt.close()

    #Temperatura
    plt.hist(df['Temprature'], bins=10, edgecolor='black')
    plt.xlabel('Temprature')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Frequência da temperatura')
    plt.show()
    plt.savefig('histograma_Temperatura.jpeg')
    plt.close()

    #Cor
    plt.hist(df['Colour'], bins=10, edgecolor='black')
    plt.xlabel('Colour')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Frequência da cor')
    plt.show()
    plt.savefig('histograma_cor.jpeg')
    plt.close()


if __name__ == "__main__":
    main()
