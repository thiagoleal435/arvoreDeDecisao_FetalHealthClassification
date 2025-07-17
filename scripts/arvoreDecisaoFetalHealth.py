# 1. Importar Bibliotecas Necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Para salvar/carregar o modelo

# 2. Configurações e Caminhos de Arquivo
# Certifique-se de que o caminho para o seu CSV esteja correto
CAMINHO_DADOS = './dados/fetal_health.csv'
CAMINHO_MODELO_SALVO = './modelos/arvore_de_decisao_modelo.pkl'
CAMINHO_RELATORIO = './relatorio/modelo_relatorio.txt'
CAMINHO_MATRIZ_DE_CONFUSAO = './relatorio/matriz_de_confusao.png'
SEMENTE_ALEATORIA = 42 # Para reprodutibilidade

# 3. Carregamento dos Dados
def carregaDados(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Dados carregados com sucesso de: {filepath}")
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filepath}' não foi encontrado.")
        exit() # Encerra o script se o arquivo não for encontrado

# 4. Pré-processamento e Divisão dos Dados
def preprocessaEDivideDados(df, colunaAlvo, test_size=0.3, random_state=SEMENTE_ALEATORIA):
    X = df.drop(columns=[colunaAlvo])
    y = df[colunaAlvo]

    # Codificar a variável alvo categórica para numérica
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes originais: {le.classes_}")
    print(f"Classes mapeadas para números: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    print(f"Dados divididos: Treino ({len(X_train)} amostras), Teste ({len(X_test)} amostras)")
    return X_train, X_test, y_train, y_test, le

# 5. Treinamento do Modelo
def treinaModelo(X_train, y_train, random_state=SEMENTE_ALEATORIA):
    print("Treinando o modelo Árvore de Decisão...")
    modelo = DecisionTreeClassifier(random_state=random_state)
    modelo.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")
    return modelo

# 6. Avaliação do Modelo
def avaliaModelo(model, X_test, y_test, codifica_rotulos, caminho_relatorio, caminho_matriz_de_confusao):
    print("\nAvaliando o modelo...")
    y_pred = model.predict(X_test)

    # Acurácia
    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {acuracia:.4f}")

    # Relatório de Classificação
    relatorio = classification_report(y_test, y_pred, nomes_alvos=codifica_rotulos.classes_)
    print("\nRelatório de Classificação:")
    print(relatorio)

    # Salvar o relatório
    with open(caminho_relatorio, 'w') as f:
        f.write(f"Acurácia: {acuracia:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(relatorio)
    print(f"Relatório de classificação salvo em: {caminho_relatorio}")

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(tamnhoFigure=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=codifica_rotulos.classes_, yticklabels=codifica_rotulos.classes_)
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.savefig(caminho_matriz_de_confusao) # Salva a imagem
    plt.close() # Fecha a figura para liberar memória
    print(f"Matriz de Confusão salva em: {caminho_matriz_de_confusao}")

# 7. Função Principal (Executa todo o fluxo)
def main():
    df = carregaDados(CAMINHO_DADOS)
    if df is not None:
        # Assumindo 'NSP' como a coluna alvo. Adapte se for diferente.
        nomeColunaAlvo = 'percentage_of_time_with_abnormal_long_term_variability'
        X_train, X_test, y_train, y_test, le = preprocessaEDivideDados(df, nomeColunaAlvo)
        
        model = treinaModelo(X_train, y_train)
        
        # Opcional: Salvar o modelo treinado
        joblib.dump(model, CAMINHO_MODELO_SALVO)
        print(f"Modelo salvo em: {CAMINHO_MODELO_SALVO}")

        avaliaModelo(model, X_test, y_test, le, CAMINHO_RELATORIO, CAMINHO_MATRIZ_DE_CONFUSAO)

if __name__ == "__main__":
    main()