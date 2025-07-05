<table align="right">
 <tr><td><a href="README_PTBR.md"><img src="imgs/brazil.png" height="15"> Português</a></td></tr>
 <tr><td><a href="README.md"><img src="imgs/united-states.png" height="15"> Inglês</a></td></tr>
</table>

# **FIAP - DATATHON**

<br/>
<p align="center">
  <a href="https://www.fiap.com.br/"><img src="https://upload.wikimedia.org/wikipedia/commons/d/d4/Fiap-logo-novo.jpg" width="300" alt="FIAP"></a>
</p>
<br>

## **Infraestrutura do repositório**

O reposório possui diversos diretórios, cada um contendo sua respectiva funcionalidade:

### **API**

Esse diretório possui a API do projeto, construida em FastAPI e integrada com o Prometheus para monitoramento.

### **Models**

Esse diretório possui o modelo treinado, usado na API.

### **Notebooks**

Esse diretório possui os notebooks de processamento de dados e treinamento do modelo utilizado no projeto.

### **Streamlit**

Esse diretório possui um chatbot que interage com o modelo.

## **Tecnologias Utilizadas**

- **FastAPI para Desenvolvimento de API:** O FastAPI, um framework web moderno e de alto desempenho para construção de APIs, foi utilizado para desenvolver os serviços do back-end e fornecimento do modelo. Ele oferece suporte assíncrono, documentação interativa automática da API e alta velocidade, tornando-o ideal para criar APIs escaláveis e eficientes.

- **Prometheus para monitoramento:** O Prometheus é uma ferramenta de monitoramento e alerta open-source amplamente utilizada para coletar e armazenar métricas de aplicações em tempo real. Nesta aplicação, o Prometheus é utilizado para monitorar o desempenho da API FastAPI, incluindo métricas como número de requisições, tempo de resposta por rota e status HTTP.

As métricas são expostas no endpoint /metrics e podem ser consumidas pelo Prometheus para análise e visualização em dashboards como o Grafana.

- **Streamlit:** Streamlit é uma biblioteca Python open-source que permite criar interfaces web interativas de forma rápida e simples, sem a necessidade de conhecimentos em front-end.

<br>

## **Como usar?**

Este projeto utiliza o FastAPI para o desenvolvimento do back-end. Siga os passos abaixo para configurar o ambiente e executar a aplicação.

### **Pré-requisitos**
Antes de executar a aplicação, certifique-se de ter os seguintes itens instalados:

- Python 3.12+
- pip (gerenciador de pacotes Python)
- FastAPI e dependências relacionadas
- Prometheus e dependências relacionadas
- Streamlit e dependências relacionadas
  
#### **01. Clone o repositório**

```bash
git clone https://github.com/FIAP-PosTech-Machine-Learning/datathon-ml.git
cd datathon-ml
```

#### **02. Instale as dependências do Python**
Certifique-se de ter um ambiente virtual configurado para evitar conflitos de dependências. Em seguida, instale as dependências listadas em requirements.txt.
```bash
python -m venv .venv        # Crie e ative o ambiente virtual
source .venv/bin/activate   # Para Linux/Mac
.venv\Scripts\activate      # Para Windows
```

```bash

pip install -r requirements.txt     # Instale as dependências
```

#### **03. Execute a aplicação FastAPI**
Após configurar o ambiente, você pode iniciar o servidor FastAPI:
```bash
uvicorn api.main:app --reload
```

#### **04. Acesse a API**

- Abra seu navegador e navegue para http://127.0.0.1:8000/ para acessar a API.

O FastAPI gera automaticamente uma documentação interativa da API, que pode ser acessada em:
- Swagger UI: http://127.0.0.1:8000
- ReDoc: http://127.0.0.1:8000/redoc

#### **05. Testando a Aplicação**

Você pode usar o Swagger UI para testar os endpoints da API ou usar ferramentas como curl, Postman ou httpie para enviar requisições à API. Caso prefire usar o streamlit execute o comando de inicialização do mesmo e usa o chat web para fazer as previsões do modelo.

### **06. Usando o Prometheus**

Para usar o prometheus, deve apontar o arquivo de configuração do mesmo. Ele se encontra dentro do diretório da `API`, no diretório `config`
```bash
prometheus --config.file=api/config/prometheus.yml
```


## Problemas Comuns
- **Dependências faltantes:** Se faltar alguma dependência, verifique o arquivo requirements.txt para garantir que tudo esteja instalado.
<br>

## **Desenvolvedores**

<table border="0" align="center">
  <tr>
  <td align="center">
      <img src="https://avatars.githubusercontent.com/u/71346377?v=4" width="160px" alt="Foto do Alexandre"/><br>
      <sub>
        <a href="https://www.github.com/alexandre-tvrs">@Alexandre Tavares</a>
      </sub>
    </td>
    <td align="center">
      <img src="https://avatars.githubusercontent.com/u/160500127?v=4" width="160px" alt="Foto do Paulo"/><br>
      <sub>
        <a href="https://github.com/PauloMukai">@Paulo Mukai</a>
      </sub>
    </td>
    </td>
        <td align="center">
      <img src="https://avatars.githubusercontent.com/u/160500128?v=4" width="160px" alt="Foto da Vanessa"/><br>
      <sub>
        <a href="https://github.com/AnjosVanessa">@AnjosVanessa</a>
      </sub>
    </td>
  </tr>
</table>