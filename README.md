<table align="right">
 <tr><td><a href="README_PTBR.md"><img src="imgs/brazil.png" height="15"> Portuguese</a></td></tr>
 <tr><td><a href="README.md"><img src="imgs/united-states.png" height="15"> English</a></td></tr>
</table>

# **FIAP - DATATHON**

<br/>
<p align="center">
  <a href="https://www.fiap.com.br/"><img src="https://upload.wikimedia.org/wikipedia/commons/d/d4/Fiap-logo-novo.jpg" width="300" alt="FIAP"></a>
</p>
<br>

## **Repository Structure**

The repository contains several directories, each with its specific functionality:

### **API**

This directory contains the project's API, built with FastAPI and integrated with Prometheus for monitoring.

### **Models**

This directory contains the trained model used by the API.

### **Notebooks**

This directory contains data processing and model training notebooks used in the project.

### **Streamlit**

This directory contains a chatbot that interacts with the model.

## **Technologies Used**

- **FastAPI for API Development:** FastAPI, a modern, high-performance web framework for building APIs, was used to develop the back-end services. It provides asynchronous support, automatic interactive API documentation, and high speed, making it optimal for creating scalable and efficient APIs.
  
- **Prometheus for Monitoring:** Prometheus is a widely used open-source monitoring and alerting tool designed to collect and store application metrics in real time. In this application, Prometheus is used to monitor the performance of the FastAPI service, including metrics like request count, response time per route, and HTTP status codes.

Metrics are exposed at the `/metrics` endpoint and can be consumed by Prometheus for analysis and visualization in dashboards such as Grafana.

- **Streamlit:** Streamlit is an open-source Python library that allows you to quickly create interactive web interfaces with minimal effort and no frontend knowledge required.

<br>

## **How to use?**

This project uses a FastAPI for back-end development. Follow the steps below to configure the environment and run the application.

### **Pre-requisites**
Before running the application, ensure you have the following installed:

- Python 3.12+
- pip (Python package manager)
- FastAPI and related dependencies
- Prometheus and related dependencies
- Streamlit and related dependencies

#### **01. Clone the repository**

```bash
git clone https://github.com/FIAP-PosTech-Machine-Learning/datathon-ml.git
cd datathon-ml
```

#### **02. Install Python dependencies**
Make sure you have a virtual environment set up to avoid dependency conflicts. Then install the required dependencies from requirements.txt.
```bash
python -m venv .venv        # Create and activate the virtual environment
source .venv/bin/activate   # For Linux/Mac
.venv\Scripts\activate      # For Windows
```

```bash
pip install -r requirements.txt     # Install dependencies
```

#### **03. Run the FastAPI application**
Once the environment are set up, you can start the FastAPI server:
```bash
uvicorn api.main:app --reload
```

#### **04. Access the API**

- Open your browser and navigate to http://127.0.0.1:8000/ to access the API.

FastAPI automatically generates interactive API documentation, which can be accessed at:
- Swagger UI: http://127.0.0.1:8000
- ReDoc: http://127.0.0.1:8000/redoc

#### **05. Testing the Application**
You can use Swagger UI to test the API endpoints, or use tools like curl, Postman, or httpie to send requests to the API.
If you prefer using the Streamlit interface, run the Streamlit app and use the web-based chat to make model predictions.

#### **06. Using Prometheus**
To use Prometheus, you must specify its configuration file. It is located inside the `API` directory, under the `config` folder:
```bash
prometheus --config.file=api/config/prometheus.yml
```

## Common Issues
- **Missing dependencies:** If any dependencies are missing, check requirements.txt to ensure everything is installed.
<br>

## **Developers**

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