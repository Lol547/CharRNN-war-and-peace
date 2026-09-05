import argparse
import torch
from model import CharLSTMModel
from utils import greedy_search, random_sample, top_k_sample, generate_text


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using trained CharRNN")
    parser.add_argument("--prompt", type=str, required=True, help="Начальная строка")
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "random", "top_k"],
                        help="Метод сэмплирования")
    parser.add_argument("--temperature", type=float, default=1.0, help="Температура (для random/top_k)")
    parser.add_argument("--k", type=int, default=5, help="K для top_k sampling")
    parser.add_argument("--length", type=int, default=300, help="Длина генерируемого текста")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Папка с весами")
    parser.add_argument("--device", type=str, default="cuda", help="cuda или cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    info_path = f"{args.checkpoint_dir}/model_info.pt"
    try:
        info = torch.load(info_path, map_location="cpu")
        char_to_id = info["char_to_id"]
        id_to_char = info["id_to_char"]
    except FileNotFoundError:
        print("Ошибка: не найден файл model_info.pt. Сначала обучите модель.")
        return

    model = CharLSTMModel(
        vocab_size=info["vocab_size"],
        embedding_dim=info["embedding_dim"],
        hidden_dim=info["hidden_dim"],
        num_layers=info["num_layers"],
        dropout=info["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(f"{args.checkpoint_dir}/best_model.pt", map_location=device))
    model.eval()

    if args.method == "greedy":
        sample_fn = greedy_search
    elif args.method == "random":
        sample_fn = lambda logits: random_sample(logits, temperature=args.temperature)
    elif args.method == "top_k":
        sample_fn = lambda logits: top_k_sample(logits, k=args.k, temperature=args.temperature)

    generated = generate_text(
        model,
        args.prompt,
        sample_fn,
        gen_length=args.length,
        char_to_id=char_to_id,
        id_to_char=id_to_char,
        device=device
    )

    print()
    print(f"Метод: {args.method.upper()}")
    if args.method != "greedy":
        print(f"Температура: {args.temperature}")
    if args.method == "top_k":
        print(f"k = {args.k}")
    print()
    print(generated)
  

if __name__ == "__main__":
    main()
