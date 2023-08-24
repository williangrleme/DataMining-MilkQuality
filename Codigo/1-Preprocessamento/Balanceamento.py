import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler

def main():
    output_file = '0-Datasets/milknew_clear.csv'
    input_file = '0-Datasets/milknew.csv'
    names = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour', 'Grade']
    features = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
    target = 'Grade'
    
    df = pd.read_csv(input_file, skiprows=3, names=names)
    df_original = df.copy()
    
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")
    
    print("Quantidade de dados por target antes do balanceamento:")
    print(df[target].value_counts())
    print("\n")
    
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")
    
    print(df.describe())
    print("\n")
    print(df.head(15))
    print(df_original.head(15))
    print("\n")
    
    # Realiza oversampling para balancear a base de dados
    X = df[features]
    y = df[target]
    
    sampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # Cria um novo DataFrame com os dados balanceados
    df_resampled = pd.DataFrame(X_resampled, columns=features)
    df_resampled[target] = y_resampled
    
    print("Quantidade de dados por target após o balanceamento:")
    print(df_resampled[target].value_counts())
    print("\n")
    
    # Salva o arquivo com o balanceamento de dados
    df_resampled.to_csv(output_file, header=False, index=False)

if __name__ == "__main__":
    main()
