import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from main import BPETokenizer, TokenizerConfig


class TokenizedDataset(Dataset):
    def __init__(self, text_path: str, tokenizer: BPETokenizer, block_size: int):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        encoded, _ = tokenizer.encode_with_debug(text)
        self.vocab_size = tokenizer.config.vocab_size
        data = torch.tensor(encoded, dtype=torch.long)

        n = len(data) - block_size
        self.x = torch.stack([data[i:i + block_size] for i in range(n)])
        self.y = torch.stack([data[i + 1:i + block_size + 1] for i in range(n)])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        assert self.n_embd % self.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer=6, n_head=8, n_embd=256, dropout=0.1):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class GPT(nn.Module):
    def __init__(self, config, tokenizer: BPETokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def train(data_path: str, tokenizer: BPETokenizer, save_path: str,
          block_size: int = 64, batch_size: int = 32, epochs: int = 10,
          n_layer: int = 6, n_head: int = 8, n_embd: int = 256):
    # Создаем датасет
    dataset = TokenizedDataset(data_path, tokenizer, block_size)

    # Конфигурация модели
    config = GPTConfig(
        vocab_size=tokenizer.config.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )

    # Создаем модель и оптимизатор
    model = GPT(config, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Training on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

        # Сохраняем модель
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config,
        }, save_path)

    return model


def generate(model: GPT, prompt: str, max_tokens: int = 100,
             temperature: float = 0.8, top_k: int = 40):
    model.eval()
    encoded, _ = model.tokenizer.encode_with_debug(prompt)
    x = torch.tensor([encoded], dtype=torch.long)

    if torch.cuda.is_available():
        x = x.cuda()
        model = model.cuda()

    with torch.no_grad():
        idx = model.generate(x, max_new_tokens=max_tokens,
                             temperature=temperature, top_k=top_k)

    return model.tokenizer.decode(idx[0].tolist())


if __name__ == "__main__":
    # Загружаем предобученный токенизатор
    config = TokenizerConfig(vocab_size=1000, min_freq=2)
    tokenizer = BPETokenizer(config)
    tokenizer.load_vocab("vocab.json")

    # Пути к файлам
    data_path = "onegin.txt"  # путь к тренировочному тексту
    model_path = "model.pt"  # путь для сохранения модели

    # Обучаем модель
    model = train(
        data_path=data_path,
        tokenizer=tokenizer,
        save_path=model_path,
        epochs=10,
        batch_size=32
    )

    # Генерируем примеры
    prompts = [
        "Однажды в студеную зимнюю пору",
        "Искусственный интеллект",
        "В далеком будущем"
    ]

    print("\nПримеры генерации:")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate(model, prompt)
        print(f"Generated: {generated}")