from fastapi.routing import APIRouter
from fastapi import HTTPException, status
from api.schemas.schema import Prediction, InputData # Caminho ajustado
from api.services.service import make_prediction    # Caminho ajustado

router = APIRouter()

@router.post(
    "/predict",
    response_model=Prediction,
    summary="Prevê a probabilidade de um match de candidato-vaga",
    description="Recebe dados brutos de um candidato e uma vaga e retorna a probabilidade de ser um match bem-sucedido."
)
async def predict(data: InputData):
    """
    Endpoint para fazer previsões de match candidato-vaga.

    Recebe:
        data (InputData): Um objeto Pydantic contendo todos os dados brutos necessários
                          para a previsão.

    Retorna:
        Prediction: Um objeto Pydantic com a probabilidade prevista de match.

    Raises:
        HTTPException: Se houver um erro durante o processo de previsão (ex: modelo não carregado, erro interno).
    """
    try:
        # Converte o objeto Pydantic para um dicionário, que é o que make_prediction espera
        prediction_probability = make_prediction(data.dict())
        return {"match_probability": prediction_probability}
    except FileNotFoundError as fnfe:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro de configuração do servidor: {fnfe}. O arquivo do modelo não foi encontrado."
        )
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno do servidor ao carregar/utilizar o modelo: {re}"
        )
    except Exception as e:
        import traceback
        traceback.print_exc() # Para ver o erro no console do servidor durante o desenvolvimento
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocorreu um erro inesperado ao fazer a previsão: {e}. "
                   "Verifique os dados de entrada e os logs do servidor."
        )