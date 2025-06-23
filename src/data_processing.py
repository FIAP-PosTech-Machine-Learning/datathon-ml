# src/data_processing.py

import pandas as pd
import json
import os
import re # Para expressões regulares no pré-processamento de texto
import nltk # Biblioteca para NLP
from nltk.corpus import stopwords # Para remover stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer # Para stemming ou lematização
from sklearn.feature_extraction.text import TfidfVectorizer # Para vetorização de texto
from sklearn.metrics.pairwise import cosine_similarity # Para calcular similaridade de cosseno
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Para codificação de variáveis categóricas
from sklearn.impute import SimpleImputer # Para tratamento de valores faltantes
from sklearn.compose import ColumnTransformer # Para aplicar diferentes transformações a diferentes colunas
from sklearn.pipeline import Pipeline # Para orquestrar as etapas de pré-processamento
import numpy as np # Para operações numéricas

# Garante que as stopwords e outros recursos do NLTK estejam disponíveis
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# --- #1--- Carregamento e Combinação dos Dados ---
def load_and_combine_data(data_path='../data/'):
    """
    Carrega os dados dos arquivos JSON e os combina em um único DataFrame.

    Args:
        data_path (str): Caminho para a pasta que contém os arquivos JSON.

    Returns:
        pd.DataFrame: DataFrame combinado contendo informações de vagas, prospecções e candidatos.
    """
    print(f"Carregando dados da pasta: {os.path.abspath(data_path)}")
    # Carrega os arquivos JSON
    with open(os.path.join(data_path, 'vagas.json'), 'r', encoding='utf-8') as f:
        jobs_data = json.load(f)
    with open(os.path.join(data_path, 'prospects.json'), 'r', encoding='utf-8') as f:
        prospects_data = json.load(f)
    with open(os.path.join(data_path, 'applicants.json'), 'r', encoding='utf-8') as f:
        applicants_data = json.load(f)

    # Convertendo JSONs para DataFrames Pandas (lógica da Célula #3--- do EDA)
    jobs_df = pd.DataFrame.from_dict(jobs_data, orient='index')
    jobs_df.index.name = 'job_id'

    prospects_records = []
    for job_id, job_info in prospects_data.items():
        if 'prospects' in job_info and isinstance(job_info['prospects'], list):
            for prospect in job_info['prospects']:
                prospect_record = {'job_id': job_id}
                prospect_record.update({k: v for k, v in job_info.items() if k != 'prospects'})
                prospect_record.update(prospect)
                prospects_records.append(prospect_record)
    prospects_df = pd.DataFrame(prospects_records)
    prospects_df['job_id'] = prospects_df['job_id'].astype(str)
    prospects_df['codigo'] = prospects_df['codigo'].astype(str)

    applicants_df = pd.DataFrame.from_dict(applicants_data, orient='index')
    applicants_df.index.name = 'applicant_id'
    applicants_df.reset_index(inplace=True)
    applicants_df['applicant_id'] = applicants_df['applicant_id'].astype(str)

    # Achatar JSONs Aninhados (lógica da Célula #4--- do EDA)
    def flatten_json_column_local(df_to_flatten, column_name_to_flatten):
        flattened_data = pd.json_normalize(df_to_flatten[column_name_to_flatten])
        flattened_data.columns = [f"{column_name_to_flatten}_{col}" for col in flattened_data.columns]
        return pd.concat([df_to_flatten.drop(columns=[column_name_to_flatten]), flattened_data], axis=1)

    jobs_df_flat = jobs_df.copy()
    nested_job_cols = ['informacoes_basicas', 'perfil_vaga', 'beneficios']
    for col in nested_job_cols:
        if col in jobs_df_flat.columns:
            if jobs_df_flat[col].apply(lambda x: isinstance(x, dict)).any():
                jobs_df_flat = flatten_json_column_local(jobs_df_flat, col)

    # Tratamento para 'job_id' após flatten e reset_index
    if 'job_id' not in jobs_df_flat.columns: # Verifica se job_id já é coluna
        jobs_df_flat.reset_index(inplace=True)
        if 'index' in jobs_df_flat.columns:
            jobs_df_flat.rename(columns={'index': 'job_id'}, inplace=True)
    jobs_df_flat['job_id'] = jobs_df_flat['job_id'].astype(str)


    applicants_df_flat = applicants_df.copy()
    nested_applicant_cols = ['infos_basicas', 'informacoes_pessoais', 'informacoes_profissionais', 'formacao_e_idiomas', 'cargo_atual']
    for col in nested_applicant_cols:
        if col in applicants_df_flat.columns:
            if applicants_df_flat[col].apply(lambda x: isinstance(x, dict)).any():
                applicants_df_flat = flatten_json_column_local(applicants_df_flat, col)

    # Combinação dos DataFrames (lógica da Célula #5--- do EDA)
    matches_df = pd.merge(prospects_df, applicants_df_flat,
                          left_on='codigo', right_on='applicant_id',
                          how='left', suffixes=('_prospect', '_applicant'))

    final_df = pd.merge(matches_df, jobs_df_flat,
                        on='job_id',
                        how='left', suffixes=('_match', '_job'))

    print(f"Dados combinados. Shape do DataFrame final: {final_df.shape}")
    return final_df

# --- #2--- Pré-processamento de Texto (NLP) ---
# stopwords em português
STOPWORDS_PT = set(stopwords.words('portuguese'))

def preprocess_text(text):
    """
    Realiza o pré-processamento básico em uma string de texto.
    - Converte para minúsculas
    - Removendo URLs
    - Removendo caracteres não alfanuméricos e números
    - Removendo espaços extras
    - Removendo stopwords
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    processed_tokens = [word for word in tokens if word not in STOPWORDS_PT]
    return " ".join(processed_tokens)

# --- #3--- Funções de Feature Engineering e Pipeline de Pré-processamento ---

def create_preprocessing_pipeline(df):
    """
    Cria e executa a pipeline completa de pré-processamento e engenharia de features.

    Args:
        df (pd.DataFrame): DataFrame final combinado da EDA.

    Returns:
        tuple: (pd.DataFrame, pd.Series, Pipeline) DataFrame processado, a Series do target e o objeto Pipeline treinado.
    """
    df_processed = df.copy()

    # 1. Definição da Variável Alvo
    df_processed['is_successful_match'] = (df_processed['situacao_candidado'] == 'Contratado pela Decision').astype(int)

    # Separar X e y
    X = df_processed.drop(columns=['is_successful_match', 'situacao_candidado'], errors='ignore')
    y = df_processed['is_successful_match']

    print(f"X inicial após separar target. Shape: {X.shape}")

    # --- Pré-processamento e Feature Engineering ---

    # 2. Aplicar pré-processamento de texto
    text_cols_for_processing = [
        'cv_pt',
        'informacoes_profissionais_conhecimentos_tecnicos',
        'perfil_vaga_principais_atividades',
        'perfil_vaga_competencia_tecnicas_e_comportamentais'
    ]
    
    for col in text_cols_for_processing:
        if col in X.columns:
            X[f'{col}_processed'] = X[col].fillna('').apply(preprocess_text)
        else:
            print(f"Aviso: Coluna de texto '{col}' não encontrada. Criando coluna vazia.")
            X[f'{col}_processed'] = ""

    # 3. Mapeamento para Label Encoding de Níveis (ordinais)
    nivel_mapping = {
        'Nenhum': 0, 'Básico': 1, 'Intermediário': 2, 'Avançado': 3, 'Fluente': 4,
        'Júnior': 0, 'Pleno': 1, 'Sênior': 2, 'Especialista': 3, 'Líder': 4, 'Gerente': 5,
        'Técnico de Nível Médio': 1, 'Técnico': 1, 'Estagiário':0, 'Nao Informado': -1,
        '': -1
    }
    
    # CORREÇÃO: Removido espaço em 'perfil_vaga_nivel profissional'
    level_cols = [
        'formacao_e_idiomas_nivel_ingles', 'perfil_vaga_nivel_ingles',
        'formacao_e_idiomas_nivel_espanhol', 'perfil_vaga_nivel_espanhol',
        'informacoes_profissionais_nivel_profissional', 'perfil_vaga_nivel_profissional', # Corrigido aqui
        'formacao_e_idiomas_nivel_academico',
        'perfil_vaga_nivel_academico'
    ]
    for col in level_cols:
        if col in X.columns:
            X[f'{col}_encoded'] = X[col].fillna('').astype(str).map(nivel_mapping).fillna(-1).astype(float)
        else:
            X[f'{col}_encoded'] = -1.0


    # 4. Features de Match de Nível
    X['match_nivel_ingles'] = (X['formacao_e_idiomas_nivel_ingles_encoded'] >= X['perfil_vaga_nivel_ingles_encoded']).astype(int)
    X['match_nivel_profissional'] = (X['informacoes_profissionais_nivel_profissional_encoded'] >= X['perfil_vaga_nivel_profissional_encoded']).astype(int)

    # 5. Extração de features de Remuneração e Faixa Etária
    if 'informacoes_profissionais_remuneracao' in X.columns:
        X['informacoes_profissionais_remuneracao_cleaned'] = X['informacoes_profissionais_remuneracao'].fillna('').astype(str).str.replace(',', '.').str.extract(r'(\d+\.?\d*)').astype(float)
    else: X['informacoes_profissionais_remuneracao_cleaned'] = np.nan
    if 'perfil_vaga_faixa_etaria' in X.columns:
        X['perfil_vaga_faixa_etaria_min'] = X['perfil_vaga_faixa_etaria'].fillna('').astype(str).str.extract(r'De:\s*(\d+)').astype(float)
        X['perfil_vaga_faixa_etaria_max'] = X['perfil_vaga_faixa_etaria'].fillna('').astype(str).str.extract(r'Até:\s*(\d+)').astype(float)
        X['perfil_vaga_faixa_etaria_mid'] = X[['perfil_vaga_faixa_etaria_min', 'perfil_vaga_faixa_etaria_max']].mean(axis=1)
    else:
        X['perfil_vaga_faixa_etaria_min'], X['perfil_vaga_faixa_etaria_max'], X['perfil_vaga_faixa_etaria_mid'] = np.nan, np.nan, np.nan

    # 6. Extração de features de tempo
    if 'informacoes_basicas_data_requicisao' in X.columns:
        X['vaga_idade_dias'] = (pd.to_datetime('today') - pd.to_datetime(X['informacoes_basicas_data_requicisao'], errors='coerce', dayfirst=True)).dt.days.fillna(-1)
    else: X['vaga_idade_dias'] = -1.0
    if 'data_candidatura' in X.columns:
        X['dias_desde_candidatura'] = (pd.to_datetime('today') - pd.to_datetime(X['data_candidatura'], errors='coerce', dayfirst=True)).dt.days.fillna(-1)
    else: X['dias_desde_candidatura'] = -1.0


    # --- 7. TF-IDF e Similaridade (LÓGICA COMPLETAMENTE REFEITA) ---
    print("Iniciando cálculo de similaridade TF-IDF (versão corrigida e eficiente)...")
    
    # Juntar todos os textos para criar um vocabulário único
    corpus = pd.concat([
        X['cv_pt_processed'],
        X['informacoes_profissionais_conhecimentos_tecnicos_processed'],
        X['perfil_vaga_principais_atividades_processed'],
        X['perfil_vaga_competencia_tecnicas_e_comportamentais_processed']
    ], axis=0).unique()

    # Criar e treinar UM ÚNICO vetorizador
    tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words=list(STOPWORDS_PT))
    tfidf_vectorizer.fit(corpus)

    # Função para transformar texto e calcular similaridade de forma eficiente
    def get_row_wise_similarity(text_col1, text_col2):
        try:
            matrix1 = tfidf_vectorizer.transform(X[text_col1])
            matrix2 = tfidf_vectorizer.transform(X[text_col2])
            # Multiplicação elemento a elemento para matrizes esparsas e soma por linha
            # Isso é equivalente à similaridade de cosseno para vetores normalizados (norma L2), que o TF-IDF produz
            similarity = np.array(matrix1.multiply(matrix2).sum(axis=1)).flatten()
            return similarity
        except Exception as e:
            print(f"Erro ao calcular similaridade entre '{text_col1}' e '{text_col2}': {e}")
            return np.zeros(len(X))

    X['sim_cv_job_activities'] = get_row_wise_similarity('cv_pt_processed', 'perfil_vaga_principais_atividades_processed')
    X['sim_cv_job_competencies'] = get_row_wise_similarity('cv_pt_processed', 'perfil_vaga_competencia_tecnicas_e_comportamentais_processed')
    X['sim_tech_skills_job_activities'] = get_row_wise_similarity('informacoes_profissionais_conhecimentos_tecnicos_processed', 'perfil_vaga_principais_atividades_processed')
    X['sim_tech_skills_job_competencies'] = get_row_wise_similarity('informacoes_profissionais_conhecimentos_tecnicos_processed', 'perfil_vaga_competencia_tecnicas_e_comportamentais_processed')
    X['sim_cv_tech_skills'] = get_row_wise_similarity('cv_pt_processed', 'informacoes_profissionais_conhecimentos_tecnicos_processed')
    print("Cálculo de similaridade concluído.")

    # 8. Definir as listas finais de features para o ColumnTransformer
    numerical_features_final = [
        'informacoes_profissionais_remuneracao_cleaned', 'perfil_vaga_faixa_etaria_mid',
        'vaga_idade_dias', 'dias_desde_candidatura',
        'sim_cv_job_activities', 'sim_cv_job_competencies',
        'sim_tech_skills_job_activities', 'sim_tech_skills_job_competencies',
        'sim_cv_tech_skills',
        'formacao_e_idiomas_nivel_ingles_encoded', 'perfil_vaga_nivel_ingles_encoded',
        'formacao_e_idiomas_nivel_espanhol_encoded', 'perfil_vaga_nivel_espanhol_encoded',
        'informacoes_profissionais_nivel_profissional_encoded', 'perfil_vaga_nivel_profissional_encoded',
        'match_nivel_ingles', 'match_nivel_profissional',
        'formacao_e_idiomas_nivel_academico_encoded', 'perfil_vaga_nivel_academico_encoded'
    ]

    categorical_onehot_features_final = [
        'informacoes_basicas_tipo_contratacao', 'informacoes_pessoais_sexo',
        'informacoes_pessoais_pcd', 'informacoes_basicas_vaga_sap',
        'informacoes_basicas_origem_vaga', 'perfil_vaga_pais',
        'informacoes_pessoais_estado_civil',
    ]

    # Filtrar para garantir que as features existam em X
    numerical_features_final = [col for col in numerical_features_final if col in X.columns]
    categorical_onehot_features_final = [col for col in categorical_onehot_features_final if col in X.columns]

    print(f"Colunas numéricas finais para o pipeline: {numerical_features_final}")
    print(f"Colunas categóricas One-Hot finais para o pipeline: {categorical_onehot_features_final}")

    # Criar um ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_features_final),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_onehot_features_final)
        ],
        remainder='drop'
    )

    # Cria a pipeline final de pré-processamento
    preprocessing_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    print("Iniciando fit_transform da pipeline de pré-processamento...")
    X_processed_array = preprocessing_pipeline.fit_transform(X)

    # Obter os nomes das features APÓS a transformação
    feature_names = preprocessing_pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    print(f"Shape do array processado: {X_processed_array.shape}")
    print(f"Número de nomes de features: {len(feature_names)}")

    X_processed = pd.DataFrame(X_processed_array, columns=feature_names, index=X.index)
    
    print(f"Pré-processamento concluído. Shape do DataFrame processado: {X_processed.shape}")
    return X_processed, y, preprocessing_pipeline


# --- Exemplo de Uso (para testar o script localmente) ---
if __name__ == "__main__":
    # Define o caminho para os dados (assumindo que você está executando de DATATHON/src/)
    base_data_path = '../data/'
    
    print("--- Testando a função load_and_combine_data ---")
    final_df_loaded = load_and_combine_data(data_path=base_data_path)
    print("\nDataFrame final carregado e combinado:")
    print(final_df_loaded.head())
    # print("\nInformações do DataFrame final carregado:")
    # final_df_loaded.info(verbose=True, show_counts=True) # Pode ser muito longo, opcional

    print("\n--- Testando a pipeline de pré-processamento ---")
    X_processed_df, y_target, pipeline_trained = create_preprocessing_pipeline(final_df_loaded.copy())
    
    print("\nDataFrame X processado (primeiras 5 linhas):")
    print(X_processed_df.head())
    print("\nInformações do DataFrame X processado:")
    X_processed_df.info(verbose=True, show_counts=True)
    print(f"\nShape de y_target: {y_target.shape}")
    print("\nDistribuição de y_target:")
    print(y_target.value_counts())

    print("\nPré-processamento e Feature Engineering do script data_processing.py concluídos com sucesso!")