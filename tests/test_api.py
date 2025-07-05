# tests/test_api.py

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest
import os
from api.main import app
from api.schemas.schemas import InputData  # Importa InputData para o exemplo

client = TestClient(app)

# Dados de exemplo para o teste (mantém os mesmos)
example_input_data = {
    "cv_pt": "Sou um profissional com experiência em Python e Machine Learning...",
    "informacoes_profissionais_conhecimentos_tecnicos": "Python, SQL, AWS, Docker, Machine Learning",
    "perfil_vaga_principais_atividades": "Desenvolvimento de modelos de ML, análise de dados",
    "perfil_vaga_competencia_tecnicas_e_comportamentais": "Python, Data Science, Comunicação, Resolução de problemas",
    "formacao_e_idiomas_nivel_ingles": "Avançado",
    "perfil_vaga_nivel_ingles": "Intermediário",
    "formacao_e_idiomas_nivel_espanhol": "Nenhum",
    "perfil_vaga_nivel_espanhol": "Nenhum",
    "informacoes_profissionais_nivel_profissional": "Pleno",
    "perfil_vaga_nivel_profissional": "Júnior",
    "formacao_e_idiomas_nivel_academico": "Superior Completo",
    "perfil_vaga_nivel_academico": "Superior Completo",
    "informacoes_profissionais_remuneracao": "7.500,00",
    "perfil_vaga_faixa_etaria": "De: 25 Até: 35",
    "informacoes_basicas_data_requicisao": "01/06/2025",
    "data_candidatura": "15/06/2025",
    "informacoes_basicas_tipo_contratacao": "CLT",
    "informacoes_pessoais_sexo": "Masculino",
    "informacoes_pessoais_pcd": "Não",
    "informacoes_basicas_vaga_sap": "Não",
    "informacoes_basicas_origem_vaga": "LinkedIn",
    "perfil_vaga_pais": "Brasil",
    "informacoes_pessoais_estado_civil": "Solteiro"
}

# --- ATENÇÃO: Patching no local onde a função é usada/importada (api.routes.predict) ---


@patch('api.routes.predict.make_prediction')
def test_predict_success(mock_make_prediction):
    """
    Testa o endpoint /predict com dados válidos e um retorno de sucesso do modelo.
    """
    # Configura o mock para simular o comportamento de um modelo treinado
    mock_make_prediction.return_value = 0.9  # Simula a probabilidade de match

    response = client.post("/predict", json=example_input_data)

    assert response.status_code == 200
    assert response.json() == {"match_probability": 0.9}
    mock_make_prediction.assert_called_once()


@patch('api.routes.predict.make_prediction')
def test_predict_file_not_found_error(mock_make_prediction):
    """
    Testa o cenário onde o arquivo do modelo não é encontrado (simulando exceção da make_prediction).
    """
    mock_make_prediction.side_effect = FileNotFoundError(
        "Simulated File Not Found")

    response = client.post("/predict", json=example_input_data)

    assert response.status_code == 500
    assert "O arquivo do modelo não foi encontrado." in response.json()[
        "detail"]


@patch('api.routes.predict.make_prediction')
def test_predict_runtime_error(mock_make_prediction):
    """
    Testa o cenário onde ocorre um RuntimeError (simulando exceção da make_prediction).
    """
    mock_make_prediction.side_effect = RuntimeError("Simulated Runtime Error")

    response = client.post("/predict", json=example_input_data)

    assert response.status_code == 500
    assert "Erro interno do servidor ao carregar/utilizar o modelo:" in response.json()[
        "detail"]


@patch('api.routes.predict.make_prediction')
def test_predict_generic_exception(mock_make_prediction):
    """
    Testa o cenário onde ocorre uma exceção genérica (simulando exceção da make_prediction).
    """
    mock_make_prediction.side_effect = ValueError("Simulated Generic Error")

    response = client.post("/predict", json=example_input_data)

    assert response.status_code == 500
    assert "Ocorreu um erro inesperado ao fazer a previsão:" in response.json()[
        "detail"]


def test_predict_invalid_input_data():
    """
    Testa o endpoint /predict com dados de entrada inválidos (validação Pydantic).
    """
    invalid_data = example_input_data.copy()
    # Espera string, recebe int
    invalid_data["formacao_e_idiomas_nivel_ingles"] = 123

    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422  # Código de erro de validação do FastAPI
    # --- ATENÇÃO: Ajuste para Pydantic V2 ---
    # Verifica o 'type' do erro Pydantic ou a mensagem de erro
    error_detail = response.json()["detail"][0]
    # Pydantic V2 geralmente usa 'string_type' para este tipo de erro de validação
    assert error_detail["type"] == "string_type"
    assert "Input should be a valid string" in error_detail["msg"]
