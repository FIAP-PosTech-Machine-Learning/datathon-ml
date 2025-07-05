# src/model_training_adjusted.py

import pandas as pd
import joblib
import os
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

    # 1. Carregar e Processar os Dados
    print("\nCarregando e processando dados...")
    final_df = load_and_combine_data(data_path='../data/')
    X_processed, y_target, preprocessing_pipeline = create_preprocessing_pipeline(final_df)
    
    print(f"\nDados processados. X_processed shape: {X_processed.shape}, y_target shape: {y_target.shape}")
    print("Distribuição da variável alvo antes da divisão:")
    print(y_target.value_counts())

    # Salva a pipeline de pré-processamento (isso não muda)
    os.makedirs('../models', exist_ok=True)
    joblib.dump(preprocessing_pipeline, '../models/preprocessing_pipeline.pkl')
    print("\nPipeline de pré-processamento salva em: ../models/preprocessing_pipeline.pkl")

    # 2. Dividir os Dados em Treino e Teste
    print("\nDividindo dados em conjuntos de treino e teste (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_target, test_size=0.2, random_state=42, stratify=y_target
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 3. Criar a Pipeline com SMOTE e o Modelo
    # Esta é a principal mudança: criamos uma pipeline que primeiro aplica o SMOTE
    # e DEPOIS treina o classificador. O SMOTE só será aplicado nos dados de fit (X_train, y_train).
    print("\nConstruindo a pipeline final com SMOTE e RandomForestClassifier...")
    
    model_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1)) # n_jobs=-1 usa todos os cores do processador
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

    # Calcula as probabilidades para a classe 1 para o ROC AUC
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score (com SMOTE): {roc_score:.4f}")

    # 6. Salvar a Pipeline Completa (Modelo + SMOTE)
    # Agora salvamos a pipeline inteira, que contém o SMOTE e o classificador.
    # Isso simplifica o deployment, pois não precisaremos recriar a pipeline na API.
    # No entanto, para o Datathon, vamos salvar o modelo treinado separadamente
    # para seguir o fluxo original, mas a pipeline é a melhor prática.
    joblib.dump(model_pipeline, '../models/final_model_pipeline.pkl')
    print("\nPipeline completa do modelo (SMOTE + Classifier) salva em: ../models/final_model_pipeline.pkl")

    print("\n--- Treinamento do Modelo Concluído com Sucesso! ---")

if __name__ == "__main__":
    train_and_evaluate_model()