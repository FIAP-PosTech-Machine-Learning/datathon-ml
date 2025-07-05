from fastapi import FastAPI
from api.routes import predict
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="DATATHON FIAP - API",
    version="1.0.0",
    description="Esta é a API criada para o Datathon curso de Machine Learning Engineering da FIAP. Seu objetivo e disponibilizar \
        o modelo treinado para previsões através da API."
)
app.include_router(predict.router)

instrumentator = Instrumentator().instrument(app).expose(app)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")