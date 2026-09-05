import re
import torch
import torch.nn.functional as F
import urllib.request
from functools import partial

def load_and_clean_text(url):
    """Скачивает и очищает текст по ссылке"""
    filename, _ = urllib.request.urlretrieve(url, "warandpeace.txt")
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    text = (
        text.replace("ё", "е")
        .replace("Ё", "Е")
        .replace("\xa0", " ")
        .replace("\xad", "")
    )
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def build_vocab(text):
    """Строит словарь символов"""
    vocab = sorted(set(text))
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_char = {i: ch for i, ch in enumerate(vocab)}
    return vocab, char_to_id, id_to_char


def create_sequences(text_tensor, sequence_length, step):
    """Создаёт входные и целевые последовательности"""
    x = text_tensor[:-1].unfold(dimension=0, size=sequence_length, step=step)
    y = text_tensor[1:].unfold(dimension=0, size=sequence_length, step=step)
    return x, y


def greedy_search(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)


def random_sample(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature должна быть больше 0")
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def top_k_sample(logits, k=5, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature должна быть больше 0")
    if k <= 0:
        raise ValueError("k должна быть больше 0")
    k = min(k, logits.size(-1))
    logits = logits / temperature
    top_logits, top_indices = torch.topk(logits, k=k, dim=-1)
    probs = F.softmax(top_logits, dim=-1)
    choice_idx = torch.multinomial(probs, num_samples=1)
    return top_indices.gather(-1, choice_idx)


def generate_text(model, prompt, sample_fn, gen_length=250, char_to_id=None, id_to_char=None, device="cpu"):
    if not prompt:
        raise ValueError("Prompt не может быть пустым")
    if char_to_id is None or id_to_char is None:
        raise ValueError("Необходимо передать словари char_to_id и id_to_char")

    unknown_chars = set(prompt) - set(char_to_id.keys())
    if unknown_chars:
        raise ValueError(f"В prompt присутствуют неизвестные символы: {unknown_chars}")

    model.eval()
    hidden = None

    input_ids = torch.tensor(
        [[char_to_id[c] for c in prompt]],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():
        logits, hidden = model(input_ids, hidden)

    last_logits = logits[0, -1]
    generated = prompt

    for _ in range(gen_length):
        with torch.no_grad():
            next_id = sample_fn(last_logits)
            next_char = id_to_char[next_id.item()]
            generated += next_char

            input_tensor = next_id.view(1, 1)
            logits, hidden = model(input_tensor, hidden)
            last_logits = logits[0, -1]

    return generated
