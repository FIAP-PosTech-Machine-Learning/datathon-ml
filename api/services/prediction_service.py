import numpy as np
import pandas as pd # Precisamos do pandas para criar o DataFrame de entrada
import joblib       # Para carregar o modelo .pkl
import os           # Para verificar a existência do arquivo

# Caminho para o seu modelo .pkl completo
FINAL_MODEL_PATH = "models/final_model_pipeline.pkl"

# Variável global para armazenar a pipeline do modelo carregada
_final_model_pipeline = None

def load_model_pipeline():
    """Carrega a pipeline completa do modelo na memória."""
    global _final_model_pipeline

    if not os.path.exists(FINAL_MODEL_PATH):
        raise FileNotFoundError(f"O arquivo do modelo não foi encontrado em: {FINAL_MODEL_PATH}. "
                                "Certifique-se de que o caminho está correto e o arquivo existe.")
    try:
        # joblib.load é o método correto para carregar arquivos .pkl do scikit-learn
        _final_model_pipeline = joblib.load(FINAL_MODEL_PATH)
        print(f"Pipeline do modelo '{FINAL_MODEL_PATH}' carregada com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar a pipeline do modelo. "
                           f"Verifique a compatibilidade das versões das bibliotecas (scikit-learn, imblearn, etc.) "
                           f"entre o ambiente de treinamento e o ambiente da API: {e}")

# Carrega a pipeline do modelo uma vez, quando o serviço é importado
load_model_pipeline()


def make_prediction(input_data_dict: dict) -> float:
    """
    Faz a previsão usando a pipeline completa do modelo.

    Args:
        input_data_dict (dict): Um dicionário contendo os dados brutos de entrada,
                                com as chaves correspondendo aos nomes das colunas originais.

    Returns:
        float: A previsão do modelo (probabilidade da classe positiva ou resultado da classificação).

    Raises:
        RuntimeError: Se a pipeline do modelo não foi carregada.
        Exception: Para quaisquer outros erros durante o processo de previsão.
    """
    if _final_model_pipeline is None:
        raise RuntimeError("A pipeline do modelo não foi carregada. Ocorreu um erro na inicialização do serviço.")

    try:
        # A pipeline final deve receber um DataFrame pandas com as colunas originais
        # exatamente como elas apareceriam em 'final_df' do seu script data_processing.py
        input_df = pd.DataFrame([input_data_dict])

        # A pipeline completa (carregada do .pkl) faz TODO o pré-processamento
        # e então aplica o classificador.
        
        # Para modelos de classificação binária, geralmente queremos a probabilidade da classe positiva
        prediction_proba = _final_model_pipeline.predict_proba(input_df)[:, 1]
        
        # Ou, se você só quer a classe prevista (0 ou 1):
        # prediction_class = _final_model_pipeline.predict(input_df)

        # Retorna a probabilidade da classe positiva
        return float(prediction_proba[0])

    except Exception as e:
        # É crucial ter um log aqui para entender falhas inesperadas
        print(f"Erro detalhado na função make_prediction: {e}")
        raise e