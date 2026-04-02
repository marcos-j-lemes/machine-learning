# xzy_mini.py
#
# A minimal GPT-style language model trained on motivational phrases.
# Built from scratch using PyTorch to demonstrate the core Transformer stack.
#
# Author: Marcos Júnior Lemes Ferreira
# Repository: https://github.com/marcos-j-ferreira/model-xzy-generative
# Hugging Face: https://huggingface.co/spaces/marcos-j-ferreira/xzy_mini

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

DATASET = [
    "eu não desisto",
    "ainda não terminei",
    "não acabou",
    "eu vou continuar",
    "eu não vou parar",
    "eu sigo em frente",
    "eu permaneço forte",
    "eu acredito em mim",
    "eu vou vencer",
    "eu vou conseguir",
    "eu continuo lutando",
    "eu não paro",
    "eu vou até o fim",
    "eu nunca desisto",
    "isso não é o fim",
    "eu ainda estou aqui",
    "eu estou ficando mais forte",
    "eu estou melhorando",
    "eu estou aprendendo",
    "eu mantenho o foco",
    "eu continuo avançando",
    "eu vou chegar lá",
    "eu não terminei",
    "eu me levanto de novo",
    "eu tento novamente",
    "eu continuo me movendo",
    "eu permaneço determinado",
    "eu não perco",
    "eu mantenho meu objetivo",
    "eu luto pelo meu sonho",
    "eu sou persistente",
    "eu supero os obstáculos",
    "eu enfrento os desafios",
    "eu sou mais forte que isso",
    "eu não me rendo",
    "eu tenho força interior",
    "eu sigo lutando",
    "eu me reconstruo",
    "eu me reinvento",
    "eu cresci com as quedas",
    "cada dificuldade me torna melhor",
    "eu sou capaz",
    "eu tenho potencial",
    "eu vou realizar meu sonho",
    "nada me para",
    "eu sou imparável",
    "eu persisto",
    "eu aguento firme",
    "eu me mantenho firme",
    "eu não fraquejo",
    "eu volto mais forte",
    "a luta continua",
    "eu sigo no caminho",
    "eu não abandono meus objetivos",
    "eu sou resiliente",
    "eu tenho garra",
    "eu tenho determinação",
    "eu não me entrego",
    "eu vou superar isso",
    "eu estou progredindo",
    "eu avanço passo a passo",
    "eu não olho para trás",
    "eu foco no que posso controlar",
    "eu sou maior que meus medos",
    "eu transformo dor em força",
    "eu aprendo com os erros",
    "eu me levanto quantas vezes for preciso",
    "eu não aceito derrota",
    "eu luto até o último segundo",
    "eu tenho coragem",
    "eu sou guerreiro",
    "eu sou guerreira",
    "eu não desanimo",
    "eu mantenho a fé",
    "eu confio no processo",
    "eu sigo firme",
    "eu continuo crescendo",
    "eu estou em evolução constante",
    "eu vou conquistar",
    "meu sonho vale a luta",
    "eu sou mais forte do que ontem",
    "eu não desisto dos meus objetivos",
    "eu persigo minha visão",
    "eu tenho fogo dentro de mim",
    "eu não me abato",
    "eu me supero todos os dias",
    "eu sou feito de aço",
    "eu aguento a pressão",
    "eu transformo impossíveis em possíveis",
    "eu não aceito limites",
    "eu crio meu próprio caminho",
    "eu sou dono do meu destino",
    "eu continuo mesmo cansado",
    "eu dou mais um passo",
    "eu nunca paro de tentar",
    "eu me recuso a desistir",
    "eu tenho força de vontade",
    "eu sou incansável",
    "eu levanto a cabeça",
    "eu sigo com determinação",
    "eu não baixo a guarda",
    "eu estou no caminho certo",
    "eu vou alcançar o topo",
    "eu sou vencedor",
    "eu sou vitorioso",
    "eu não me comparo, eu melhoro",
    "eu foco no meu progresso",
    "eu celebro pequenas vitórias",
    "eu sou grato pelo desafio",
    "eu uso a dor como combustível",
    "eu transformo fracasso em lição",
    "eu renasço das cinzas",
    "eu sou inabalável",
    "eu tenho mentalidade de campeão",
    "eu treino minha mente todos os dias",
    "eu escolho persistir",
    "eu escolho lutar",
    "eu escolho vencer",
    "eu sou mais forte que as dificuldades",
    "eu não me entrego ao cansaço",
    "eu respiro fundo e continuo",
    "eu tenho poder dentro de mim",
    "eu sou capaz de muito mais",
    "eu estou construindo meu futuro",
    "eu não paro no meio do caminho",
    "eu vou até onde ninguém foi",
    "eu sou exemplo de persistência",
    "eu inspiro outros com minha luta",
    "eu não desisto fácil",
    "eu sou determinado",
    "eu sou focado",
    "eu sou disciplinado",
    "eu sou consistente",
    "eu mostro resultado com atitude",
    "eu transformo minha vida",
    "eu crio minha própria sorte",
    "eu mereço o sucesso",
    "eu trabalho pelo meu sonho",
    "eu dou tudo de mim",
    "eu me dedico totalmente",
    "eu não aceito menos que o meu melhor",
    "eu sou um guerreiro da vida",
    "eu enfrento tudo de cabeça erguida",
    "eu nunca perco, ou eu ganho ou eu aprendo",
    "eu sigo em frente apesar de tudo",
    "eu sou mais forte do que pareço",
    "eu tenho um coração de leão",
    "eu não tenho tempo para desistir",
    "eu faço acontecer",
    "eu vou realizar tudo que eu desejo",
    "eu sou o autor da minha história",
    "eu escrevo meu sucesso com persistência",
    "eu não me entrego nunca",
    "eu continuo mesmo quando dói",
    "eu sou inquebrável",
    "eu levanto, sacudo a poeira e sigo em frente",
    "eu tenho alma de vencedor",
    "eu nasci para vencer",
    "eu vou chegar no meu objetivo",
    "eu sou persistente como poucos",
    "eu não largo o osso",
    "eu mordo e não solto",
    "eu vou até o fim, custe o que custar",
    "eu sou feito para grandes coisas",
    "eu acredito no meu potencial",
    "eu confio na minha força",
    "eu sou capaz de superar qualquer coisa",
    "eu nunca estou sozinho na luta",
    "eu carrego minha própria motivação",
    "eu sou minha maior força",
    "eu vou brilhar",
    "eu vou conquistar o impossível",
    "eu sou o próximo a vencer",
    "eu não desisto dos meus objetivos",
    "eu mantenho o fogo aceso",
    "eu sou movido a determinação",
    "eu transformo sonhos em realidade",
    "eu sou o exemplo de que é possível",
    "eu continuo mesmo no escuro",
    "eu encontro força onde os outros desistem",
    "eu sou forte, sou capaz, sou vencedor",
]

print(len(DATASET))  # Para você ver o tamanho

# ─────────────────────────────────────────────
#  Vocabulary
# ─────────────────────────────────────────────

all_words   = " ".join(DATASET)
vocab       = sorted(set(all_words.split()))
word2idx    = {word: idx for idx, word in enumerate(vocab)}
idx2word    = {idx: word for idx, word in enumerate(vocab)}
VOCAB_SIZE  = len(vocab)


def tokenize(text: str) -> list[int]:
    """Converts a string of known words into a list of token indices."""
    return [word2idx[word] for word in text.strip().split() if word in word2idx]


tokenized_dataset = [tokenize(sentence) for sentence in DATASET]

print("-" * 50)
print("Vocabulary summary")
print(f"  Size   : {VOCAB_SIZE} words")
#print(f"  Words  : {vocab}")
print("-" * 50)


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

class MiniGPT(nn.Module):
    """
    Minimal GPT-style model using PyTorch's built-in TransformerEncoder.

    Architecture:
        Token Embedding  →  Positional Embedding  →  Transformer Stack  →  Linear Head
    """

    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int,
        num_heads:     int,
        num_layers:    int,
        max_seq_len:   int,
        ffn_dim:       int,
    ):
        super().__init__()

        # Maps each token index to a dense vector
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)

        # Learnable positional encoding (one vector per position)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)

        # Stack of Transformer encoder layers with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.15,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projects the hidden state to vocabulary logits
        self.linear_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: LongTensor of shape (batch, seq_len)
        Returns:
            logits: FloatTensor of shape (batch, seq_len, vocab_size)
        """
        seq_len = tokens.shape[1]

        token_emb = self.token_emb(tokens)
        positions = torch.arange(seq_len, device=tokens.device)
        pos_emb   = self.pos_emb(positions).unsqueeze(0)   # (1, seq_len, dim)
        x         = token_emb + pos_emb

        # Causal mask: each position can only attend to past positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tokens.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        return self.linear_head(x)


# ─────────────────────────────────────────────
#  Hyperparameters
# ─────────────────────────────────────────────

# EMBEDDING_DIM = 32
# NUM_HEADS     = 2
# NUM_LAYERS    = 2
# MAX_SEQ_LEN   = 10
# FFN_DIM       = 64       # rule of thumb: 2–4× embedding_dim

# EMBEDDING_DIM = 64
# NUM_HEADS     = 4
# NUM_LAYERS    = 2
# MAX_SEQ_LEN   = 16
# FFN_DIM       = 256

# EMBEDDING_DIM = 64
# NUM_HEADS     = 4
# NUM_LAYERS    = 3
# MAX_SEQ_LEN   = 20
# FFN_DIM       = 256

EMBEDDING_DIM = 256
NUM_HEADS     = 8
NUM_LAYERS    = 6
MAX_SEQ_LEN   = 32
FFN_DIM       = 1024

model     = MiniGPT(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN, FFN_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)

best_loss = float('inf')
patience = 10
counter = 0

# EPOCHS = 500

# print("\nTraining...\n")

# for epoch in range(EPOCHS):
#     total_loss = 0.0
#     model.train()

#     for seq in tokenized_dataset:
#         if len(seq) < 2:
#             continue

#         tokens = torch.tensor(seq).unsqueeze(0).to(device)

#         input_tokens  = tokens[:, :-1]
#         target_tokens = tokens[:, 1:]

#         logits = model(input_tokens)

#         loss = F.cross_entropy(
#             logits.view(-1, VOCAB_SIZE),
#             target_tokens.view(-1),
#         )

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(tokenized_dataset)

#     if epoch % 50 == 0:
#         print(f"  Epoch {epoch:>4} | Loss: {avg_loss:.4f}")

#     # Early stopping
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         counter = 0
#         torch.save(model.state_dict(), "best_model.pt")
#     else:
#         counter += 1
#         if counter >= patience:
#             print(f"\nEarly stopping at epoch {epoch}")
#             break

# print("\nTraining complete.\n")


EPOCHS = 500

print("\nTraining...\n")

for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()

    for seq in tokenized_dataset:
        if len(seq) < 2:
            continue

        tokens        = torch.tensor(seq).unsqueeze(0)   # (1, seq_len)
        input_tokens  = tokens[:, :-1]                   # all but last
        target_tokens = tokens[:, 1:]                    # all but first

        logits = model(input_tokens)                     # (1, seq_len-1, vocab_size)

        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            target_tokens.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(tokenized_dataset)

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:>4} | Loss: {avg_loss:.4f}")

print("\nTraining complete.\n")

# ─────────────────────────────────────────────
#  Text generation
# ─────────────────────────────────────────────

def generate(model: nn.Module, prompt: str, max_new_tokens: int = 25) -> str:
    """
    Generates a sequence of tokens given a prompt string.

    Args:
        model          : trained MiniGPT instance
        prompt         : starting words (must exist in vocabulary)
        max_new_tokens : how many tokens to generate after the prompt

    Returns:
        Full generated sentence as a string.
    """
    model.eval()
    tokens = torch.tensor([tokenize(prompt)])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits     = model(tokens[:, -MAX_SEQ_LEN:])  # respect max context window
            last_logits = logits[:, -1, :]
            probs      = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens     = torch.cat([tokens, next_token], dim=1)

    return " ".join(idx2word[i] for i in tokens[0].tolist())


# ─────────────────────────────────────────────
#  Interactive demo  (Hugging Face / terminal)
# ─────────────────────────────────────────────


def stream_output(text: str, delay: float = 0.08) -> None:
    """Prints words one by one to mimic streaming output."""
    words = text.split()
    for i, word in enumerate(words):
        print(word, end=" ", flush=True)
        if (i + 1) % 10 == 0:
            print()
        time.sleep(delay)
    print("\n")


print("=" * 50)
print("MiniGPT — Motivational Text Generator")
print("=" * 50)
print(f"Available starting words: {vocab}\n")

while True:
    user_input = input("Enter a word to start (or 'quit' to exit): ").strip().lower()

    if user_input in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    # if user_input not in word2idx:
    #     print(f"  ⚠  '{user_input}' is not in the vocabulary. Try one of: {vocab}\n")
    #     continue

    print("\nGenerating", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print("\n")

    output = generate(model, user_input, max_new_tokens=15)
    stream_output(output)
