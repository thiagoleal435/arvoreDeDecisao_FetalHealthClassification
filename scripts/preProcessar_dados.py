import pandas as pd

print(pd.__version__)

'''try:
    df = pd.read_csv('./dados/fatal_health.csv')
    print("Dados carregados com sucesso!")
    print(df.head()) # Mostra as primeiras 5 linhas para verificar
except FileNotFoundError:
    print("Erro: O arquivo 'cardiotocograma_data.csv' não foi encontrado. Por favor, verifique o nome e o caminho do arquivo.")

nome_da_coluna_alvo = 'NSP' # Exemplo, ajuste para o nome correto da sua coluna de classes
X = df.drop(columns=[nome_da_coluna_alvo])
y = df[nome_da_coluna_alvo]

print("\nFeatures (X) - Exemplo:")
print(X.head())
print("\nVariável Alvo (y) - Exemplo:")
print(y.head())'''