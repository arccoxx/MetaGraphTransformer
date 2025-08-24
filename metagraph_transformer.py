import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gc
import json
import os

# Toy dataset for causal LM: Simple sentences, predict next token
vocab = ['<pad>', '<sos>', '<eos>', 'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'loves', 'running']
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

sequences = [
    ['<sos>', 'the', 'quick', 'brown', 'fox', '<eos>'],
    ['<sos>', 'fox', 'jumps', 'over', 'lazy', 'dog', '<eos>'],
    ['<sos>', 'dog', 'loves', 'running', '<eos>'],
    ['<sos>', 'the', 'brown', 'dog', '<eos>'],
    ['<sos>', 'quick', 'fox', 'jumps', '<eos>']
]
tokenized = [[word_to_ix[w] for w in seq] for seq in sequences]
max_len = max(len(s) for s in tokenized)

# Pad sequences
padded = [s + [0] * (max_len - len(s)) for s in tokenized]
inputs = torch.tensor(padded)[:, :-1]
targets = torch.tensor(padded)[:, 1:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs = inputs.to(device)
targets = targets.to(device)

# --- STABLE MODEL DEFINITIONS ---

def _init_weights(module):
    """Initializes weights using Xavier uniform for linear layers."""
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


# Causal Transformer Layer (Pre-Norm)
class CausalTransformerLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, num_heads=2, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        # Pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + attn_out

        # Pre-norm
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

# Basic Transformer for Causal LM
class BasicTransformer(nn.Module):
    def __init__(self, vocab_size, d, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len - 1, d) * 0.02) # Scaled init
        self.layers = nn.ModuleList([CausalTransformerLayer(d) for _ in range(num_layers)])
        self.head = nn.Linear(d, vocab_size)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(x)

# Graph Transformer Layer (Pre-Norm)
class GraphTransformerLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, num_heads=2, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, adj_mask):
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=adj_mask)
        x = x + attn_out
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

# Graph Transformer
class GraphTransformer(nn.Module):
    def __init__(self, vocab_size, d, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len-1, d) * 0.02) # Scaled init
        self.layers = nn.ModuleList([GraphTransformerLayer(d) for _ in range(num_layers)])
        self.head = nn.Linear(d, vocab_size)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        n = x.size(1)
        adj_mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, adj_mask)
        return self.head(x)

# Meta Graph Transformer Layer (Fully Pre-Norm and Stable)
class MetaGraphTransformerLayer(nn.Module):
    def __init__(self, d, meta_window=2):
        super().__init__()
        self.meta_window = meta_window
        self.self_attn = nn.MultiheadAttention(d, num_heads=2, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        b, n, d = x.shape
        # Pre-norm before pooling and attention
        x_norm = self.norm1(x)
        meta_n = (n + self.meta_window - 1) // self.meta_window
        pad_len = meta_n * self.meta_window - n
        x_pad = F.pad(x_norm, (0, 0, 0, pad_len)) if pad_len > 0 else x_norm

        pooled = x_pad.view(b, meta_n, self.meta_window, d).mean(dim=2)
        meta_mask = torch.triu(torch.ones(meta_n, meta_n, device=x.device), diagonal=1).bool()

        attn_out, _ = self.self_attn(pooled, pooled, pooled, attn_mask=meta_mask)
        pooled = pooled + attn_out

        # Pre-norm for FFN
        pooled_norm = self.norm2(pooled)
        ffn_out = self.ffn(pooled_norm)
        pooled = pooled + ffn_out
        
        # Upsample and return the residual
        return pooled.repeat_interleave(self.meta_window, dim=1)[:, :n]

# Meta Graph Transformer for Causal LM
class MetaGraphTransformer(nn.Module):
    def __init__(self, vocab_size, d, num_layers=2, meta_window=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len-1, d) * 0.02) # Scaled init
        self.layers = nn.ModuleList([MetaGraphTransformerLayer(d, meta_window) for _ in range(num_layers)])
        self.head = nn.Linear(d, vocab_size)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            x = x + layer(x) # Additive residual update
        return self.head(x)

# Training function with gradient clipping
def train_model(model, inputs, targets, epochs=50, lr=1e-4): # REDUCED LEARNING RATE
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=0)
        
        # Check for NaN loss before backward pass
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch}. Stopping training.")
            return [float('nan')] * epochs
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return losses

# Test multilayer: Vary num_layers
num_layers_list = [1, 2, 3, 4]
d = 32
results_file = 'model_results.json'
results = {'basic': {}, 'graph': {}, 'meta': {}}

for num_layers in num_layers_list:
    print(f"--- Training with {num_layers} layers ---")
    # Basic
    print("Training BasicTransformer...")
    basic_model = BasicTransformer(vocab_size, d, num_layers)
    basic_losses = train_model(basic_model, inputs, targets)
    results['basic'][num_layers] = basic_losses[-1]
    del basic_model
    gc.collect()
    torch.cuda.empty_cache()

    # Graph
    print("Training GraphTransformer...")
    graph_model = GraphTransformer(vocab_size, d, num_layers)
    graph_losses = train_model(graph_model, inputs, targets)
    results['graph'][num_layers] = graph_losses[-1]
    del graph_model
    gc.collect()
    torch.cuda.empty_cache()

    # Meta
    print("Training MetaGraphTransformer...")
    meta_model = MetaGraphTransformer(vocab_size, d, num_layers)
    meta_losses = train_model(meta_model, inputs, targets)
    results['meta'][num_layers] = meta_losses[-1]
    del meta_model
    gc.collect()
    torch.cuda.empty_cache()

    # Save intermediate
    with open(results_file, 'w') as f:
        json.dump(results, f)

# Load and print final results
with open(results_file, 'r') as f:
    results = json.load(f)

print("\nFinal Results:")
for model_type in results:
    print(f"{model_type.capitalize()} Transformer:")
    for layers, loss in results[model_type].items():
        if loss is not None and not isinstance(loss, str):
             print(f"  Layers {layers}: Final Loss {loss:.4f}")
        else:
             print(f"  Layers {layers}: Final Loss {loss}")
