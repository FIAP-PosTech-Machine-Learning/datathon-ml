# datathon-ml

##   Análise Exploratória de Dados (EDA) - Projeto Datathon Decision
Visão Geral do Projeto##
Este projeto de Machine Learning tem como objetivo otimizar o processo de recrutamento e seleção da empresa Decision, especializada em conectar talentos de TI a vagas específicas. Atualmente, a Decision enfrenta desafios na agilidade do "match" entre candidatos e vagas, padronização de entrevistas e identificação do real engajamento do candidato. Nossa solução visa propor uma inteligência artificial para prever a probabilidade de sucesso de um "match" (contratação), aprendendo com dados históricos.

Fase 1: Análise Exploratória de Dados (EDA)
A Análise Exploratória de Dados (EDA) é a primeira etapa crucial do nosso trabalho. Nela, mergulhamos nos dados brutos da Decision (vagas.json, prospects.json, applicants.json) para entender sua estrutura, identificar padrões, avaliar a qualidade e extrair insights que guiarão a construção do modelo preditivo.

Passo a Passo da Análise (Células do Notebook EDA.ipynb)
Cada seção abaixo corresponde a uma célula executada no notebook, com seu propósito e resultados chave:

#1--- Importando as bibliotecas:

Propósito: Preparar o ambiente, carregando as ferramentas essenciais (Pandas para manipulação de dados, JSON para leitura de arquivos, Matplotlib/Seaborn para visualização).

Resultado para o Negócio: Garante que temos as ferramentas necessárias para trabalhar com grandes volumes de dados e visualizá-los eficientemente.

#2--- Carregamento dos Dados:

Propósito: Carregar os três arquivos JSON fornecidos pela Decision (vagas.json, prospects.json, applicants.json) para a memória, tornando-os acessíveis para análise.

Resultado para o Negócio: Valida que temos acesso aos dados históricos e que eles podem ser lidos, sendo a base para qualquer análise futura.

#3--- Convertendo JSONs para DataFrames Pandas:

Propósito: Transformar a estrutura complexa dos arquivos JSON (que são como árvores de informação) em tabelas organizadas (DataFrames do Pandas). Isso torna os dados fáceis de visualizar e manipular.

Resultado para o Negócio: Dados brutos são convertidos em um formato estruturado, que permite uma compreensão clara do número de vagas, candidatos e prospecções.

#4--- Normalização (Achatar JSONs Aninhados para Facilitação):

Propósito: Os dados continham informações "aninhadas" (dicionários dentro de dicionários, como informacoes_basicas dentro de vagas). Esta etapa "achata" essa estrutura, transformando cada informação aninhada em uma coluna própria (ex: informacoes_basicas_titulo_vaga). Também corrigimos um problema com o ID da vaga (job_id) que não estava sendo reconhecido corretamente.

Resultado para o Negócio: Organiza os dados em um formato tabular amplo (DataFrame jobs_df_flat e applicants_df_flat com muitas colunas), onde cada característica da vaga ou do candidato é uma coluna acessível, facilitando a análise e a engenharia de features.

#5--- Combinação dos DataFrames para Análise de Match:

Propósito: Juntar as informações de vagas, prospecções e candidatos em uma única tabela (final_df). Cada linha dessa tabela agora representa um par "candidato-vaga" e seu status de prospecção.

Resultado para o Negócio: Criamos uma visão unificada de todo o processo de recrutamento, com 107.494 registros de interações candidato-vaga e 112 colunas de informações combinadas. Isso é a base completa para a predição. A execução prolongada desta célula destacou o volume considerável de dados que estamos manipulando.

#6--- Definição da Variável Alvo: is_successful_match:

Propósito: Identificar e criar a variável que o nosso modelo irá prever: se um "match" foi bem-sucedido ou não. Definimos "sucesso" como "Contratado pela Decision".

Resultado para o Negócio: Transformamos o status de prospecção em um indicador claro de sucesso (1 para Contratado, 0 para outros). A análise revelou um forte desbalanceamento de classes (cerca de 5.515 contratados vs. mais de 101.979 não contratados). Isso é uma descoberta crucial que exigirá atenção especial na fase de treinamento do modelo para evitar previsões enviesadas.

#7--- Análise de Dados Faltantes em final_df:

Propósito: Avaliar a quantidade de informações ausentes (NaN) em cada coluna do DataFrame combinado.

Resultado para o Negócio: Identificamos que muitas colunas têm um volume significativo de dados faltantes (algumas com mais de 98% de ausência, outras com 16%, 75%). Isso aponta para a necessidade de estratégias de limpeza de dados, como remoção de colunas com excesso de NaNs ou imputação de valores (preencher com "Não Informado", média, moda) para garantir que o modelo não seja prejudicado por informações incompletas.

#8--- Exploração de Colunas Chave para Feature Engineering:

Propósito: Aprofundar a análise em colunas específicas consideradas mais importantes para o "match", como níveis profissionais, de idiomas, conhecimentos técnicos e atividades de vagas.

Resultado para o Negócio: Confirmamos que campos de texto (currículos, descrições de vagas) são extremamente ricos e exigirão processamento de linguagem natural (NLP) para extrair similaridades. Também validamos a presença de dados categóricos (tipos de contratação, áreas de atuação) e numéricos (remuneração, datas) que serão transformados em features.

Conclusão da EDA e Próximos Passos
A fase de EDA foi concluída com sucesso, nos fornecendo uma compreensão robusta dos dados da Decision. Entendemos suas estruturas, a qualidade das informações, a distribuição da variável de sucesso e os principais desafios (desbalanceamento e dados faltantes).

Próximo Passo: Com base nestas observações, o projeto avançará para a implementação das funções de Pré-processamento e Feature Engineering em um script Python modular (src/data_processing.py). Esta fase transformará os dados brutos em um formato numérico e estruturado, pronto para alimentar o modelo de Machine Learning.
