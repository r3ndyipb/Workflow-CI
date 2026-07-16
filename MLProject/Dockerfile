FROM python:3.12-slim

WORKDIR /app

COPY conda.yaml .
COPY modelling.py .
COPY skin_type_preprocessing.csv .

RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy joblib

RUN python modelling.py

EXPOSE 5001

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001"]
