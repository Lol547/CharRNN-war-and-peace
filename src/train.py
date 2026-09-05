import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import CharLSTMModel
from utils import load_and_clean_text, build_vocab, create_sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Train CharRNN on War and Peace")
    parser.add_argument("--data_url", type=str,
                        default="https://gist.githubusercontent.com/romaklimenko/c95f3a864828f7f034b7a33d1676e420/raw/warandpeace.txt",
                        help="URL текста")
    parser.add_argument("--epochs", type=int, default=40, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=64, help="Размер батча")
    parser.add_argument("--seq_len", type=int, default=150, help="Длина последовательности")
    parser.add_argument("--step", type=int, default=25, help="Шаг скользящего окна")
    parser.add_argument("--embed_dim", type=int, default=128, help="Размер эмбеддинга")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Размер скрытого состояния")
    parser.add_argument("--num_layers", type=int, default=2, help="Количество слоёв LSTM")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Папка для сохранения весов")
    parser.add_argument("--device", type=str, default="cuda", help="cuda или cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Используемое устройство: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Загрузка текста...")
    text = load_and_clean_text(args.data_url)
    print(f"Длина текста: {len(text)} символов")

    vocab, char_to_id, id_to_char = build_vocab(text)
    vocab_size = len(vocab)
    print(f"Размер словаря: {vocab_size}")

    full_tensor = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)
    split_idx = int(len(full_tensor) * 0.9)
    train_tensor = full_tensor[:split_idx]
    val_tensor = full_tensor[split_idx:]

    x_train, y_train = create_sequences(train_tensor, args.seq_len, args.step)
    x_val, y_val = create_sequences(val_tensor, args.seq_len, args.step)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = CharLSTMModel(
        vocab_size=vocab_size,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits, _ = model(x_batch)
                loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total * 100

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pt"))
            print(f"Сохранена лучшая модель (Val Loss: {avg_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping на эпохе {epoch+1}")
                break

    torch.save({
        "vocab_size": vocab_size,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "embedding_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout
    }, os.path.join(args.checkpoint_dir, "model_info.pt"))

    print("Обучение завершено.")


if __name__ == "__main__":
    main()
