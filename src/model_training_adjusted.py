# src/model_training_adjusted.py

import pandas as pd
import joblib
import os  # Importar os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Importações necessárias do imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Importa as funções do nosso outro script
from data_processing import load_and_combine_data, create_preprocessing_pipeline


def train_and_evaluate_model():
    """
    Função principal para orquestrar o carregamento, processamento e treinamento do modelo.
    """
    print("--- Iniciando o Treinamento do Modelo (Versão com SMOTE) ---")

    # Define o caminho para a raiz do projeto de forma robusta
    # Isso obtém o diretório do arquivo atual (src/), depois volta um nível (para datathon-ml/)
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Agora project_root é J:\Projetos\datathon-ml
    project_root = os.path.dirname(project_root)

    # Define os caminhos completos para as pastas data e models
    data_folder = os.path.join(project_root, 'data')
    models_folder = os.path.join(project_root, 'models')

    # Garante que o diretório 'models' exista antes de tentar salvar qualquer coisa nele
    os.makedirs(models_folder, exist_ok=True)

    # 1. Carregar e Processar os Dados
    print("\nCarregando e processando dados...")
    # Passa o caminho absoluto para a pasta 'data'
    final_df = load_and_combine_data(data_path=data_folder)
    # Passa o caminho absoluto para a pasta 'models' para a função de pré-processamento
    X_processed, y_target, preprocessing_pipeline = create_preprocessing_pipeline(
        final_df, models_path=models_folder)

    print(
        f"\nDados processados. X_processed shape: {X_processed.shape}, y_target shape: {y_target.shape}")
    print("Distribuição da variável alvo antes da divisão:")
    print(y_target.value_counts())

    # Salva a pipeline de pré-processamento usando o caminho absoluto
    joblib.dump(preprocessing_pipeline, os.path.join(
        models_folder, 'preprocessing_pipeline.pkl'))
    print(
        f"\nPipeline de pré-processamento salva em: {os.path.join(models_folder, 'preprocessing_pipeline.pkl')}")

    # 2. Dividir os Dados em Treino e Teste
    print("\nDividindo dados em conjuntos de treino e teste (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_target, test_size=0.2, random_state=42, stratify=y_target
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 3. Criar a Pipeline com SMOTE e o Modelo
    print("\nConstruindo a pipeline final com SMOTE e RandomForestClassifier...")

    model_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # 4. Treinar o Modelo
    print("\nTreinando o modelo com a pipeline SMOTE...")
    model_pipeline.fit(X_train, y_train)
    print("Treinamento do modelo concluído.")

    # 5. Avaliar o Modelo
    print("\nAvaliando o desempenho do modelo no conjunto de teste...")
    y_pred = model_pipeline.predict(X_test)

    print("\n--- Relatório de Classificação (com SMOTE) ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Matriz de Confusão (com SMOTE) ---")
    print(confusion_matrix(y_test, y_pred))

    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score (com SMOTE): {roc_score:.4f}")

    # 6. Salvar a Pipeline Completa (Modelo + SMOTE) usando o caminho absoluto
    joblib.dump(model_pipeline, os.path.join(
        models_folder, 'final_model_pipeline.pkl'))
    print(
        f"\nPipeline completa do modelo (SMOTE + Classifier) salva em: {os.path.join(models_folder, 'final_model_pipeline.pkl')}")

    print("\n--- Treinamento do Modelo Concluído com Sucesso! ---")


if __name__ == "__main__":
    train_and_evaluate_model()
