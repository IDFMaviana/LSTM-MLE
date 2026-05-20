import argparse
import json
import os
from datetime import date
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src import data, model


def inverse_transform(scaler: MinMaxScaler, values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().cpu().numpy().reshape(-1, 1)
    return torch.from_numpy(scaler.inverse_transform(flat).reshape(-1, 1))


def compute_metrics(
    scaler: MinMaxScaler, preds: torch.Tensor, targets: torch.Tensor
) -> dict:
    pred_orig = inverse_transform(scaler, preds)
    target_orig = inverse_transform(scaler, targets)
    pred_arr = pred_orig.squeeze().numpy()
    target_arr = target_orig.squeeze().numpy()
    mae = mean_absolute_error(target_arr, pred_arr)
    mse = mean_squared_error(target_arr, pred_arr)
    rmse = mse ** 0.5
    epsilon = 1e-8
    mape = (abs(target_arr - pred_arr) / (target_arr + epsilon)).mean() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina modelo LSTM sobre preços históricos")
    parser.add_argument("--symbol", required=True, help="Ticker Yahoo Finance (ex: DIS)")
    parser.add_argument("--start", help="Data inicial no formato YYYY-MM-DD (opcional se --years for usado)")
    parser.add_argument("--end", help="Data final no formato YYYY-MM-DD (opcional, padrão hoje)")
    parser.add_argument("--years", type=int, default=1, help="Quantidade de anos anteriores usados quando --start não é informado")
    parser.add_argument("--lookback", type=int, default=30, help="Quantidade de passos históricos usados")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do lote para treino")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="models", help="Diretório para salvar artefatos")
    parser.add_argument("--raw-dir", default="data/raw", help="Onde salvar o CSV com os dados baixados")
    return parser


def train(args: argparse.Namespace) -> None:
    end_date = args.end or date.today().isoformat()
    df = data.download_price_data(args.symbol, args.start, end_date, years=args.years)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"{args.symbol}_{end_date}.csv"
    df.to_csv(raw_file)
    print(f"Dados brutos salvos em {raw_file}")
    train_dataset, val_dataset, scaler = data.prepare_data(df, args.lookback)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.StockLSTM(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(train_dataset.features), torch.from_numpy(train_dataset.targets)
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(val_dataset.features), torch.from_numpy(val_dataset.targets)
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimiser = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        net.train()
        losses = []
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimiser.zero_grad()
            preds = net(X)
            loss = criterion(preds, y)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        print(f"Epoca {epoch}/{args.epochs} - loss medio {avg_loss:.6f}")

    net.eval()
    with torch.no_grad():
        val_preds = []
        val_targets = []
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            logits = net(X)
            val_preds.append(logits.cpu())
            val_targets.append(y.cpu())
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
    metrics = compute_metrics(scaler, val_preds, val_targets)
    print("Avaliacao em validacao:", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = Path(args.output_dir) / f"{args.symbol}_lstm.pth"
    torch.save(net.state_dict(), model_path)

    scaler_path = Path(args.output_dir) / f"{args.symbol}_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    metadata = {
        "symbol": args.symbol,
        "lookback": args.lookback,
        "feature": "Close",
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "model_filename": model_path.name,
        "scaler_filename": scaler_path.name,
    }
    metadata_path = Path(args.output_dir) / f"{args.symbol}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print(f"Modelos e metadados salvos em {args.output_dir}")


def main() -> None:
    parser = build_parser()
    train(parser.parse_args())


if __name__ == "__main__":
    main()
