from pydantic import BaseModel, Field
from typing import List, Optional

# --- Se você usar esta classe, ajuste-a para as suas colunas reais! ---
# As colunas e tipos devem corresponder às entradas do seu 'final_df'
# antes de qualquer pré-processamento no script de treino.
class InputData(BaseModel):
    # Colunas de Texto (devem ser strings)
    cv_pt: Optional[str] = Field(None, description="Conteúdo do CV em português.")
    informacoes_profissionais_conhecimentos_tecnicos: Optional[str] = Field(None, description="Conhecimentos técnicos do candidato.")
    perfil_vaga_principais_atividades: Optional[str] = Field(None, description="Principais atividades da vaga.")
    perfil_vaga_competencia_tecnicas_e_comportamentais: Optional[str] = Field(None, description="Competências técnicas e comportamentais da vaga.")

    # Colunas de Nível (devem ser strings que serão mapeadas numericamente)
    formacao_e_idiomas_nivel_ingles: Optional[str] = Field(None, description="Nível de inglês do candidato (e.g., 'Básico', 'Intermediário', 'Avançado').")
    perfil_vaga_nivel_ingles: Optional[str] = Field(None, description="Nível de inglês requerido pela vaga.")
    formacao_e_idiomas_nivel_espanhol: Optional[str] = Field(None, description="Nível de espanhol do candidato.")
    perfil_vaga_nivel_espanhol: Optional[str] = Field(None, description="Nível de espanhol requerido pela vaga.")
    informacoes_profissionais_nivel_profissional: Optional[str] = Field(None, description="Nível profissional do candidato (e.g., 'Júnior', 'Pleno', 'Sênior').")
    perfil_vaga_nivel_profissional: Optional[str] = Field(None, description="Nível profissional da vaga.")
    formacao_e_idiomas_nivel_academico: Optional[str] = Field(None, description="Nível acadêmico do candidato.")
    perfil_vaga_nivel_academico: Optional[str] = Field(None, description="Nível acadêmico requerido pela vaga.")

    # Colunas Numéricas/Misc (string para extração de números ou datas)
    informacoes_profissionais_remuneracao: Optional[str] = Field(None, description="Remuneração atual ou esperada do candidato (texto, ex: 'R$ 5.000,00').")
    perfil_vaga_faixa_etaria: Optional[str] = Field(None, description="Faixa etária da vaga (texto, ex: 'De: 25 Até: 35').")
    informacoes_basicas_data_requicisao: Optional[str] = Field(None, description="Data de requisição da vaga (texto, ex: 'DD/MM/AAAA').")
    data_candidatura: Optional[str] = Field(None, description="Data da candidatura (texto, ex: 'DD/MM/AAAA').")

    # Colunas Categóricas (devem ser strings)
    informacoes_basicas_tipo_contratacao: Optional[str] = Field(None, description="Tipo de contratação da vaga (e.g., 'CLT', 'PJ').")
    informacoes_pessoais_sexo: Optional[str] = Field(None, description="Sexo do candidato.")
    informacoes_pessoais_pcd: Optional[str] = Field(None, description="Indicador de Pessoa com Deficiência (e.g., 'Sim', 'Não').")
    informacoes_basicas_vaga_sap: Optional[str] = Field(None, description="Se a vaga é SAP (e.g., 'Sim', 'Não').")
    informacoes_basicas_origem_vaga: Optional[str] = Field(None, description="Origem da vaga.")
    perfil_vaga_pais: Optional[str] = Field(None, description="País da vaga.")
    informacoes_pessoais_estado_civil: Optional[str] = Field(None, description="Estado civil do candidato.")

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class Prediction(BaseModel):
    """
    Define o esquema de saída para a resposta da previsão.
    Retorna a probabilidade de um match bem-sucedido.
    """
    match_probability: float = Field(
        ...,
        description="A probabilidade de um match bem-sucedido (entre 0 e 1).",
        example=0.85
    )