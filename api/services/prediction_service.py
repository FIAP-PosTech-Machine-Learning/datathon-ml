import numpy as np
import pandas as pd
import joblib
import os

# Caminhos para suas pipelines .pkl
FINAL_MODEL_PATH = "models/final_model_pipeline.pkl"
# Adicionado caminho da pipeline de pré-processamento
PREPROCESSING_PIPELINE_PATH = "models/preprocessing_pipeline.pkl"

# Variáveis globais para armazenar as pipelines carregadas
_final_model_pipeline = None
# Adicionada variável para a pipeline de pré-processamento
_preprocessing_pipeline = None


def load_model_pipelines():  # Renomeada para indicar que carrega múltiplas pipelines
    """Carrega as pipelines completa do modelo e de pré-processamento na memória."""
    global _final_model_pipeline
    global _preprocessing_pipeline

    # Carrega a pipeline de pré-processamento
    if not os.path.exists(PREPROCESSING_PIPELINE_PATH):
        raise FileNotFoundError(
            f"O arquivo da pipeline de pré-processamento não foi encontrado em: {PREPROCESSING_PIPELINE_PATH}.")
    try:
        _preprocessing_pipeline = joblib.load(PREPROCESSING_PIPELINE_PATH)
        print(
            f"Pipeline de pré-processamento '{PREPROCESSING_PIPELINE_PATH}' carregada com sucesso!")
    except Exception as e:
        raise RuntimeError(
            f"Erro ao carregar a pipeline de pré-processamento: {e}. Verifique o arquivo.")

    # Carrega a pipeline do modelo final
    if not os.path.exists(FINAL_MODEL_PATH):
        raise FileNotFoundError(
            f"O arquivo do modelo não foi encontrado em: {FINAL_MODEL_PATH}.")
    try:
        _final_model_pipeline = joblib.load(FINAL_MODEL_PATH)
        print(
            f"Pipeline do modelo '{FINAL_MODEL_PATH}' carregada com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar a pipeline do modelo. "
                           f"Verifique a compatibilidade das versões das bibliotecas (scikit-learn, imblearn, etc.) "
                           f"entre o ambiente de treinamento e o ambiente da API: {e}")


# Carrega as pipelines uma vez, quando o serviço é importado
load_model_pipelines()  # Chamada atualizada

# Adicione esta função auxiliar para replicar a engenharia de features de nível e datas
# Esta é uma parte crucial que a pipeline de pré-processamento não faz por si só,
# pois envolve a criação de colunas de comparação e parsing de strings.
# Ela deve ser idêntica à lógica em src/data_processing.py ANTES do ColumnTransformer.


def _apply_manual_feature_engineering(df):
    df_temp = df.copy()  # Use uma cópia para evitar SettingWithCopyWarning

    # Mapeamento para Label Encoding de Níveis (ordinais) - Copiado de data_processing.py
    nivel_mapping = {
        'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Avançado': 3, 'Fluente': 4,
        'Júnior': 0, 'Pleno': 1, 'Sênior': 2, 'Especialista': 3, 'Líder': 4, 'Gerente': 5,
        'Técnico de Nível Médio': 1, 'Técnico': 1, 'Estagiário': 0, 'Nao Informado': -1,
        '': -1
    }

    level_cols = [
        'formacao_e_idiomas_nivel_ingles', 'perfil_vaga_nivel_ingles',
        'formacao_e_idiomas_nivel_espanhol', 'perfil_vaga_nivel_espanhol',
        'informacoes_profissionais_nivel_profissional', 'perfil_vaga_nivel_profissional',
        'formacao_e_idiomas_nivel_academico',
        'perfil_vaga_nivel_academico'
    ]
    for col in level_cols:
        if col in df_temp.columns:
            # Garante que a coluna existe antes de tentar mapear
            df_temp[f'{col}_encoded'] = df_temp[col].fillna('').astype(
                str).map(nivel_mapping).fillna(-1).astype(float)
        else:
            # Cria a coluna com valor padrão se não existir na entrada da API
            df_temp[f'{col}_encoded'] = -1.0

    # Features de Match de Nível - Copiado de data_processing.py
    # Garante que as colunas codificadas existem antes de criar as features de match
    if 'formacao_e_idiomas_nivel_ingles_encoded' in df_temp.columns and \
       'perfil_vaga_nivel_ingles_encoded' in df_temp.columns:
        df_temp['match_nivel_ingles'] = (
            df_temp['formacao_e_idiomas_nivel_ingles_encoded'] >= df_temp['perfil_vaga_nivel_ingles_encoded']).astype(int)
    else:
        # Valor padrão ou tratamento adequado
        df_temp['match_nivel_ingles'] = 0

    if 'informacoes_profissionais_nivel_profissional_encoded' in df_temp.columns and \
       'perfil_vaga_nivel_profissional_encoded' in df_temp.columns:
        df_temp['match_nivel_profissional'] = (
            df_temp['informacoes_profissionais_nivel_profissional_encoded'] >= df_temp['perfil_vaga_nivel_profissional_encoded']).astype(int)
    else:
        # Valor padrão ou tratamento adequado
        df_temp['match_nivel_profissional'] = 0

    # Extração de features de Remuneração e Faixa Etária - Copiado de data_processing.py
    if 'informacoes_profissionais_remuneracao' in df_temp.columns:
        df_temp['informacoes_profissionais_remuneracao_cleaned'] = df_temp['informacoes_profissionais_remuneracao'].fillna(
            '').astype(str).str.replace(',', '.').str.extract(r'(\d+\.?\d*)').astype(float)
    else:
        df_temp['informacoes_profissionais_remuneracao_cleaned'] = np.nan

    if 'perfil_vaga_faixa_etaria' in df_temp.columns:
        df_temp['perfil_vaga_faixa_etaria_min'] = df_temp['perfil_vaga_faixa_etaria'].fillna(
            '').astype(str).str.extract(r'De:\s*(\d+)').astype(float)
        df_temp['perfil_vaga_faixa_etaria_max'] = df_temp['perfil_vaga_faixa_etaria'].fillna(
            '').astype(str).str.extract(r'Até:\s*(\d+)').astype(float)
        df_temp['perfil_vaga_faixa_etaria_mid'] = df_temp[[
            'perfil_vaga_faixa_etaria_min', 'perfil_vaga_faixa_etaria_max']].mean(axis=1)
    else:
        df_temp['perfil_vaga_faixa_etaria_min'], df_temp['perfil_vaga_faixa_etaria_max'], df_temp['perfil_vaga_faixa_etaria_mid'] = np.nan, np.nan, np.nan

    # Extração de features de tempo - Copiado de data_processing.py
    # Note: 'pd.to_datetime('today')' pode causar inconsistências se o modelo for treinado em uma data e usado em outra.
    # Para consistência em produção, é ideal usar uma data de referência fixa ou passada como input.
    if 'informacoes_basicas_data_requicisao' in df_temp.columns:
        df_temp['vaga_idade_dias'] = (pd.to_datetime('today') - pd.to_datetime(
            df_temp['informacoes_basicas_data_requicisao'], errors='coerce', dayfirst=True)).dt.days.fillna(-1)
    else:
        df_temp['vaga_idade_dias'] = -1.0
    if 'data_candidatura' in df_temp.columns:
        df_temp['dias_desde_candidatura'] = (pd.to_datetime('today') - pd.to_datetime(
            df_temp['data_candidatura'], errors='coerce', dayfirst=True)).dt.days.fillna(-1)
    else:
        df_temp['dias_desde_candidatura'] = -1.0

    return df_temp


def make_prediction(input_data_dict: dict) -> float:
    """
    Faz a previsão usando as pipelines completas de pré-processamento e modelo.

    Args:
        input_data_dict (dict): Um dicionário contendo os dados brutos de entrada,
                                com as chaves correspondendo aos nomes das colunas originais.

    Returns:
        float: A previsão do modelo (probabilidade da classe positiva ou resultado da classificação).

    Raises:
        RuntimeError: Se as pipelines do modelo não foram carregadas.
        Exception: Para quaisquer outros erros durante o processo de previsão.
    """
    if _preprocessing_pipeline is None or _final_model_pipeline is None:
        raise RuntimeError(
            "As pipelines do modelo e/ou pré-processamento não foram carregadas. Erro na inicialização do serviço.")

    try:
        # 1. Converte o objeto Pydantic para um DataFrame pandas
        input_df = pd.DataFrame([input_data_dict])

        # 2. Aplicar pré-processamento de texto (função externa, se necessário, ou reimplementada aqui)
        # O TF-IDF Vectorizer está DENTRO da sua preprocessing_pipeline (ColumnTransformer)
        # em src/data_processing.py. Então, ele será aplicado pelo .transform()
        # Mas as colunas _processed são criadas ANTES de entrar no ColumnTransformer.
        # Precisamos replicar isso aqui.

        # Importa as stopwords (necessário para preprocess_text)
        import re
        from nltk.corpus import stopwords
        try:
            STOPWORDS_PT = set(stopwords.words('portuguese'))
        except LookupError:
            import nltk
            nltk.download('stopwords')
            STOPWORDS_PT = set(stopwords.words('portuguese'))

        def preprocess_text(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+",
                          '', text, flags=re.MULTILINE)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            tokens = text.split()
            processed_tokens = [
                word for word in tokens if word not in STOPWORDS_PT]
            return " ".join(processed_tokens)

        text_cols_for_processing = [
            'cv_pt',
            'informacoes_profissionais_conhecimentos_tecnicos',
            'perfil_vaga_principais_atividades',
            'perfil_vaga_competencia_tecnicas_e_comportamentais'
        ]

        for col in text_cols_for_processing:
            if col in input_df.columns:
                input_df[f'{col}_processed'] = input_df[col].fillna(
                    '').apply(preprocess_text)
            else:
                # Garante que a coluna existe mesmo que vazia
                input_df[f'{col}_processed'] = ""

        # 3. Aplicar as features de engenharia manual (níveis, datas, remuneração)
        input_df_engineered = _apply_manual_feature_engineering(input_df)

        # 4. As features de texto (TF-IDF) e similaridade, além de codificações manuais (níveis, datas, remuneração),
        #    foram criadas ANTES do ColumnTransformer no script de treinamento (data_processing.py).
        #    Portanto, essas etapas de engenharia de features devem ser replicadas AQUI
        #    antes que os dados sejam transformados pela `_preprocessing_pipeline`.
        #    O `_tfidf_vectorizer` carregado é usado para gerar as features de similaridade.
        #  Carregar o TF-IDF Vectorizer

        TFIDF_VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"  # Novo caminho
        global _tfidf_vectorizer  # Nova variável global

        if not os.path.exists(TFIDF_VECTORIZER_PATH):
            raise FileNotFoundError(
                f"O arquivo do TF-IDF Vectorizer não foi encontrado em: {TFIDF_VECTORIZER_PATH}.")
        try:
            _tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            print(
                f"TF-IDF Vectorizer '{TFIDF_VECTORIZER_PATH}' carregado com sucesso!")
        except Exception as e:
            raise RuntimeError(
                f"Erro ao carregar o TF-IDF Vectorizer: {e}. Verifique o arquivo.")

        # Replicar a função get_row_wise_similarity de data_processing.py
        def get_row_wise_similarity(text_col1, text_col2):
            try:
                # Transforma as colunas _processed, pois o vetorizador foi fitado no corpus delas
                matrix1 = _tfidf_vectorizer.transform(
                    input_df_engineered[text_col1])
                matrix2 = _tfidf_vectorizer.transform(
                    input_df_engineered[text_col2])
                similarity = np.array(matrix1.multiply(
                    matrix2).sum(axis=1)).flatten()
                return similarity
            except Exception as e:
                print(
                    f"Erro ao calcular similaridade entre '{text_col1}' e '{text_col2}': {e}")
                return np.zeros(len(input_df_engineered))

        # Adicionar as features de similaridade ao DataFrame
        input_df_engineered['sim_cv_job_activities'] = get_row_wise_similarity(
            'cv_pt_processed', 'perfil_vaga_principais_atividades_processed')
        input_df_engineered['sim_cv_job_competencies'] = get_row_wise_similarity(
            'cv_pt_processed', 'perfil_vaga_competencia_tecnicas_e_comportamentais_processed')
        input_df_engineered['sim_tech_skills_job_activities'] = get_row_wise_similarity(
            'informacoes_profissionais_conhecimentos_tecnicos_processed', 'perfil_vaga_principais_atividades_processed')
        input_df_engineered['sim_tech_skills_job_competencies'] = get_row_wise_similarity(
            'informacoes_profissionais_conhecimentos_tecnicos_processed', 'perfil_vaga_competencia_tecnicas_e_comportamentais_processed')
        input_df_engineered['sim_cv_tech_skills'] = get_row_wise_similarity(
            'cv_pt_processed', 'informacoes_profissionais_conhecimentos_tecnicos_processed')

        # Agora, transformamos o DataFrame com a pipeline de pré-processamento
        # Esta pipeline espera TODAS as colunas que foram criadas acima.
        X_processed_for_prediction = _preprocessing_pipeline.transform(
            input_df_engineered)

        # Para predição, precisamos garantir que as colunas do X_processed_for_prediction
        # estejam na mesma ordem e com os mesmos nomes que o modelo espera.
        # _preprocessing_pipeline.get_feature_names_out() pode ajudar aqui.
        # No entanto, a predict_proba() geralmente aceita um array numpy.
        # O importante é que a ordem das features seja a mesma.

        prediction_proba = _final_model_pipeline.predict_proba(
            X_processed_for_prediction)[:, 1]

        return float(prediction_proba[0])

    except Exception as e:
        print(f"Erro detalhado na função make_prediction: {e}")
        raise e
