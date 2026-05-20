# LSTM Tech Challenge


Link do video: https://youtu.be/CrTJIRe7JcI

## Visao Geral
Pipeline completo para treinar redes LSTM que preveem o fechamento de acoes, cobrindo coleta de dados, tratamento, treinamento, avaliacao, exportacao de artefatos e uma API de inferencia.

## Estrutura
- `src/data.py`: download de dados historicos e preparacao de sequencias temporais.
- `src/model.py`: definicao do modelo LSTM em PyTorch.
- `src/train.py`: script de treino com metricas MAE, RMSE, MAPE e exportacao de modelo, scaler e metadados.
- `src/predict.py`: carregamento dos artefatos e geracao de previsao a partir de uma janela historica recente.
- `src/api.py`: FastAPI com o endpoint `/predict`.
- `docker/Dockerfile`: imagem leve para expor a API com Uvicorn.
- `models/`: diretorio para artefatos gerados no treinamento.

## Setup inicial
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Treinar o modelo
```bash
python -m src.train \
  --symbol ITUB4.SA \
  --years 1 \
  --lookback 30 \
  --epochs 30
```
O script baixa automaticamente o ultimo ano (`--years`) quando `--start` nao for informado (use `--start/--end` para intervalos especificos), salva o CSV usado para treino em `data/raw/DIS_<data_final>.csv` e grava os artefatos em `models/DIS_lstm.pth`, `models/DIS_scaler.pkl` e `models/DIS_metadata.json`. As metricas MAE, RMSE e MAPE sao exibidas para a base de validacao.

## Rodar a API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Chame `POST /predict` com o corpo JSON (a lista deve conter exatamente `lookback` valores de fechamento do mais antigo ao mais recente):
```json
{
  "symbol": "ITUB4.SA",
  "history": [120.5, 121.4, 123.1, 125.3]
}
```
A resposta inclui a previsao para o proximo timestamp.

## Deploy com Docker
```bash
docker build -t lstm-api -f docker/Dockerfile .
docker run -p 8000:8000 -v $(pwd)/models:/app/models lstm-api
```
