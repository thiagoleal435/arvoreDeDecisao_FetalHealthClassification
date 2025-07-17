# arvoreDecisaoFetalHealth.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Configurações e Caminhos de Arquivo
CAMINHO_DADOS = './dados/fetal_health.csv'
CAMINHO_MODELO_SALVO = './modelos/arvore_de_decisao_modelo.pkl'
CAMINHO_RELATORIO = './relatorio/modelo_relatorio.txt'
CAMINHO_MATRIZ_DE_CONFUSAO = './relatorio/matriz_de_confusao.png'
CAMINHO_MATRIZ_DE_CONFUSAO_CSV = './relatorio/matriz_de_confusao.csv'
SEMENTE_ALEATORIA = 42

def carregaDados(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Dados carregados com sucesso de: {filepath}")
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo '{filepath}' não foi encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        sys.exit(1)

def preprocessaEDivideDados(df, colunaAlvo, test_size=0.3, random_state=SEMENTE_ALEATORIA):
    # Tratamento de valores ausentes
    if df.isnull().sum().sum() > 0:
        print("Aviso: Existem valores ausentes no dataset. Preenchendo com a mediana.")
        df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=[colunaAlvo])
    y = df[colunaAlvo]

    # Verifica se a coluna alvo tem mais de um valor único para LabelEncoder
    if y.nunique() < 2:
        print(f"Erro: A coluna alvo '{colunaAlvo}' tem menos de 2 classes únicas. Não é um problema de classificação adequado.")
        exit()
        
    # Codificar a variável alvo categórica para numérica
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes originais:'{colunaAlvo}': {le.classes_}")
    print(f"Classes mapeadas para números: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    print(f"Dados divididos: Treino ({len(X_train)} amostras), Teste ({len(X_test)} amostras)")
    return X_train, X_test, y_train, y_test, le

def treinaModelo(X_train, y_train, random_state=SEMENTE_ALEATORIA):
    print("Treinando o modelo Árvore de Decisão...")
    modelo = DecisionTreeClassifier(random_state=random_state)
    modelo.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")
    return modelo

def avaliaModelo(model, X_test, y_test, codifica_rotulos, caminho_relatorio, caminho_matriz_de_confusao, caminho_matriz_csv):
    print("\nAvaliando o modelo...")
    y_pred = model.predict(X_test)

    # Acurácia
    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {acuracia:.4f}")

    # Relatório de Classificação
    relatorio = classification_report(y_test, y_pred, target_names=codifica_rotulos.classes_)
    print("\nRelatório de Classificação:")
    print(relatorio)

    # Garante que as pastas existem
    os.makedirs(os.path.dirname(caminho_relatorio), exist_ok=True)
    os.makedirs(os.path.dirname(caminho_matriz_de_confusao), exist_ok=True)

    # Salvar o relatório
    with open(CAMINHO_RELATORIO, 'w') as f:
        f.write(f"Acurácia: {acuracia:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(relatorio)
    print(f"Relatório de classificação salvo em: {caminho_relatorio}")

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=codifica_rotulos.classes_, yticklabels=codifica_rotulos.classes_)
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.savefig(caminho_matriz_de_confusao) # Salva a imagem
    plt.close() # Fecha a figura para liberar memória
    print(f"Matriz de Confusão salva em: {caminho_matriz_de_confusao}")

    # Salvar matriz de confusão como CSV
    pd.DataFrame(cm, index=codifica_rotulos.classes_, columns=codifica_rotulos.classes_).to_csv(caminho_matriz_csv)
    print(f"Matriz de Confusão (CSV) salva em: {caminho_matriz_csv}")

def main():
    df = carregaDados(CAMINHO_DADOS)
    if df is not None:
        # Assumindo 'fetal_health' como a coluna alvo. Adapte se for diferente.
        nomeColunaAlvo = 'fetal_health'
        X_train, X_test, y_train, y_test, le = preprocessaEDivideDados(df, nomeColunaAlvo)
        
        modelo = treinaModelo(X_train, y_train)
        
        os.makedirs(os.path.dirname(CAMINHO_MODELO_SALVO), exist_ok=True)
        joblib.dump(modelo, CAMINHO_MODELO_SALVO)
        print(f"Modelo salvo em: {CAMINHO_MODELO_SALVO}")

        avaliaModelo(modelo, X_test, y_test, le, CAMINHO_RELATORIO, CAMINHO_MATRIZ_DE_CONFUSAO, CAMINHO_MATRIZ_DE_CONFUSAO_CSV)

if __name__ == "__main__":
    main()