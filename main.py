import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
import math
from torch.utils.data import Dataset, DataLoader
import re

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

text ="""
    pergunta qual é a capital do brasil resposta brasília é a capital federal desde 1960
    pergunta quem descobriu o brasil resposta os portugueses liderados por pedro alvares cabral chegaram em 1500
    pergunta quantos estados tem o brasil resposta o brasil tem vinte e seis estados e um distrito federal
    pergunta qual é a maior floresta do mundo resposta a amazonia é a maior floresta tropical do mundo
    pergunta o que é inteligencia artificial resposta inteligencia artificial é a simulacao de processos humanos por maquinas
    pergunta quem escreveu dom casmurro resposta machado de assis escreveu dom casmurro em 1899
    pergunta qual é o maior planeta do sistema solar resposta jupiter é o maior planeta do sistema solar
    pergunta como se faz arroz doce resposta cozinha se arroz com leite acucar e canela ate engrossar
    pergunta qual a função do fígado resposta o figado produz bile e metaboliza nutrientes e toxinas
    pergunta quem pintou a mona lisa resposta leonardo da vinci pintou a mona lisa no renascimento
    pergunta qual é o pais mais populoso do mundo resposta a china é o pais mais populoso com mais de 1 bilhao de pessoas
    pergunta o que é fotossintese resposta fotossintese é o processo onde plantas produzem energia usando luz solar
    pergunta quem foi albert einstein resposta albert einstein foi um fisico alema criador da teoria da relatividade
    pergunta qual é o menor pais do mundo resposta o vaticano é o menor pais do mundo com menos de um quilometro quadrado
    pergunta como se calcula a area de um circulo resposta a area do circulo e pi vezes o raio ao quadrado
    pergunta o que significa a sigla onu resposta a onu significa organizacao das nacoes unidas fundada em 1945
    pergunta quem foi frida kahlo resposta frida kahlo foi uma pintora mexicana conhecida por seus autorretratos
    pergunta qual o maior oceano do mundo resposta o oceano pacifico é o maior oceano da terra
    pergunta o que é cambio resposta cambio é a conversao de uma moeda em outra moeda estrangeira
    pergunta como se faz pao caseiro resposta mistura se farinha agua fermento sal e acucar depois assa se
    pergunta qual a formula da agua resposta a formula da agua e h dois o ou h2o
    pergunta quem criou o facebook resposta mark zuckerberg criou o facebook em 2004 em harvard
    pergunta qual é o time de futebol mais vitorioso do mundo resposta o real madrid tem mais titulos da champions league
    pergunta o que é a globalizacao resposta globalizacao é a integracao economica cultural e politica mundial
    pergunta quem foi steve jobs resposta steve jobs foi cofundador da apple e criador do iphone
    pergunta como funciona a energia solar resposta paineis solares captam luz e transformam em eletricidade
    pergunta qual é o maior animal terrestre resposta o elefante africano é o maior animal terrestre
    pergunta o que é a pangeia resposta pangeia foi o supercontinente que existiu ha duzentos milhoes de anos
    pergunta quem escreveu os lusiadas resposta luis de camoes escreveu os lusiadas em 1572
    pergunta qual é o livro mais vendido do mundo resposta a biblia é o livro mais vendido da historia
    pergunta o que é a lei da gravidade resposta a lei da gravidade explica a atracao entre objetos com massa
    pergunta qual a diferenca entre virus e bacteria resposta virus precisam de celulas para replicar e bacterias sao unicelulares
    pergunta como se faz um curriculo resposta curriculo deve ter dados pessoais formacao e experiencia profissional
    pergunta qual é a moeda oficial do japao resposta o iene é a moeda oficial do japao
    pergunta o que é o efeito estufa resposta efeito estufa é o aquecimento da terra por gases na atmosfera
    pergunta quem foi cleopatra resposta cleopatra foi a ultima rainha do egito antigo
    pergunta qual é o ponto mais alto do mundo resposta o monte everest tem 8848 metros de altura
    pergunta o que significa a sigla uf resposta uf pode significar unidade federativa ou ufologia
    pergunta como se faz um bolo de chocolate resposta mistura farinha ovos leite chocolate e fermento depois assa
    pergunta qual é o maior deserto do mundo resposta o deserto da antartida é o maior do mundo
    pergunta o que e a reciclagem resposta reciclagem é transformar lixo em novos produtos para reduzir impacto ambiental
    pergunta quem foi martin luther king resposta martin luther king foi ativista dos direitos civis nos eua
    pergunta qual a diferenca entre mito e lenda resposta mito explica origem do mundo e lenda tem base historica
    pergunta como se escreve a palavra saudade em ingles resposta saudade pode ser traduzida como missing ou longing
    pergunta qual é o pais com mais dias de chuva resposta a colombia tem mais de trezentos dias de chuva por ano
    pergunta o que é a matematica resposta matematica é a ciencia que estuda numeros formas e logicas
    pergunta quem foi nelson mandela resposta nelson mandela foi lider anti apartheid e presidente da africa do sul
    pergunta qual é a velocidade da luz resposta a luz viaja a trezentos mil quilometros por segundo
    pergunta o que é o dna resposta dna é o acido desoxirribonucleico com informacao genetica
    pergunta como se faz uma horta organica resposta usa se terra fertil sem pesticidas com adubo natural e agua

    pergunta qual é a capital da frança resposta paris é a capital da franca e cidade da torre eiffel
    pergunta quem escreveu a divina comedia resposta dante alighieri escreveu a divina comedia no seculo quatorze
    pergunta qual é o maior animal marinho resposta a baleia azul pode chegar a trinta metros de comprimento
    pergunta o que é a ressonancia magnetica resposta ressonancia magnetica é um exame de imagem por campos magneticos
    pergunta como se prepara uma lasanha resposta intercala camadas de massa molho e queijo e vai ao forno
    pergunta qual é o pais mais rico do mundo resposta os estados unidos tem o maior pib do mundo
    pergunta o que é a filosofia resposta filosofia é o estudo dos problemas fundamentais sobre existencia conhecimento e valores
    pergunta quem foi isaac newton resposta isaac newton formulou as leis da gravidade e do movimento
    pergunta qual é o rio mais longo do mundo resposta o rio nilo tem aproximadamente 6650 quilometros de extensao
    pergunta o que significa a palavra amor resposta amor é um sentimento de afeto profundo por alguem ou algo
    pergunta como se calcula a velocidade media resposta divide se a distancia percorrida pelo tempo gasto
    pergunta qual é a capital do japao resposta toquio é a capital do japao e uma das maiores cidades do mundo
    pergunta quem foi marie curie resposta marie curie ganhou dois premios nobel por descobertas sobre radioatividade
    pergunta o que é a internet resposta internet é uma rede global de computadores conectados
    pergunta qual é o maior predador terrestre resposta o urso polar é considerado o maior predador terrestre
    pergunta como se faz um poema resposta poema tem versos e estrofes com ritmo e emocao
    pergunta qual é a diferenca entre calor e temperatura resposta calor é energia em transferencia e temperatura mede agitacao molecular
    pergunta o que é a democracia resposta democracia é um sistema onde o povo escolhe seus governantes
    pergunta quem foi viktor frankl resposta viktor frankl criou a logoterapia e sobreviveu aos campos nazistas
    pergunta qual é o instrumento musical mais antigo resposta a flauta de osso tem cerca de quarenta mil anos
    pergunta como se cultiva orquideas resposta orquideas precisam de luz indireta umidade e rega moderada
    pergunta o que e a empatia resposta empatia é a capacidade de entender os sentimentos dos outros
    pergunta quem pintou o teto da capela sistina resposta michelangelo pintou o teto da capela sistina
    pergunta qual é o maior arranha ceu do mundo resposta o burj khalifa em dubai tem 828 metros de altura
    pergunta o que e a sindrome de down resposta sindrome de down é uma condicao genetica com tres cromossomos vinte e um
    pergunta como se escreve a palavra conhecimento resposta conhecimento se escreve com c e com g e com mento no final
    pergunta qual é a cidade mais populosa do mundo resposta toquio tem mais de trinta e sete milhoes de habitantes
    pergunta o que e a gravidez resposta gravidez é o periodo onde um feto se desenvolve no utero materno
    pergunta quem foi socrates resposta socrates foi um filosofo grego mestre de platao
    pergunta qual é a capital do egito resposta o cairo é a capital do egito e cidade das piramides
    pergunta como se calcula o imc resposta divide se o peso pela altura ao quadrado
    pergunta o que é a logica resposta logica é o estudo dos principios do raciocinio valido
    pergunta quem escreveu o pequeno principe resposta antoine de saint exupery escreveu o pequeno principe
    pergunta qual é o maior lago do mundo resposta o mar caspio é o maior lago com area de 371 mil quilometros quadrados
    pergunta o que e a ansiedade resposta ansiedade é uma resposta natural ao estresse que pode se tornar patologica
    pergunta como se faz um sumo de laranja resposta espreme se laranjas e coa se o sumo para retirar sementes
    pergunta qual é o pais mais antigo do mundo resposta o san marino foi fundado em 301 e existe ate hoje
    pergunta o que é a felicidade resposta felicidade é um estado de bem estar e realizacao pessoal
    pergunta quem foi charles darwin resposta charles darwin propos a teoria da evolucao por selecao natural
    pergunta qual é a velocidade do som resposta o som viaja a aproximadamente 343 metros por segundo no ar
    pergunta como se escreve a palavra psicologia resposta psicologia se escreve com p e com s e com g e com i a
    pergunta o que é a fisica quantica resposta fisica quantica estuda particulas subatomicas e suas interacoes
    pergunta qual a diferenca entre ética e moral resposta etica reflete sobre a moral e a moral sao costumes de uma sociedade
    pergunta quem foi galileu galilei resposta galileu foi astronomo que defendeu que a terra gira em torno do sol
    pergunta qual é o maior festival de musica do mundo resposta o rock in rio e o glastonbury estao entre os maiores
    pergunta como se prepara uma salada de frutas resposta corta se varias frutas e mistura se com suco de laranja
    pergunta o que é a sustentabilidade resposta sustentabilidade é usar recursos naturais sem esgotar para o futuro
    pergunta quem foi mozart resposta wolfgang amadeus mozart foi compositor prodigio da musica classica
    pergunta qual é a capital da alemanha resposta berlim é a capital da alemanha e berlin

    pergunta o que é um algoritmo resposta algoritmo é uma sequencia de passos para resolver um problema
    pergunta como se escreve o numero mil em algarismos romanos resposta mil em algarismos romanos é a letra m
    pergunta qual é o maior time do brasil resposta varios times como flamengo corinthians e palmeiras tem grandes torcidas
    pergunta o que é a terapia cognitiva resposta terapia cognitiva ajuda a mudar pensamentos negativos e comportamentos
    pergunta quem foi monteiro lobato resposta monteiro lobato escreveu o sitio do picapau amarelo e literatura infantil
    pergunta qual é a diferenca entre covid e gripe resposta covid é causada por coronavirus e gripe por virus influenza
    pergunta como se faz um planejamento financeiro resposta organiza se receitas despesas e metas de economia mensal
    pergunta o que é a neurociencia resposta neurociencia estuda o sistema nervoso e o cerebro humano
    pergunta quem criou a teoria da relativiade resposta albert einstein criou a teoria da relatividade em 1905
    pergunta qual é o ponto mais frio da terra resposta a estacao vostok na antartida registrou menos 89 graus celsius
    pergunta o que é a astrologia resposta astrologia estuda influencia de astros na personalidade mas nao e ciencia
    pergunta como se escreve a palavra exuberante resposta exuberante se escreve com x e com u e com b e com e no meio
    pergunta qual é o pais com mais vulcoes ativos resposta a indonesia tem mais de cento e trinta vulcoes ativos
    pergunta o que é o estresse resposta estresse é uma reacao do corpo a situacoes de pressao ou perigo
    pergunta quem foi aristoteles resposta aristoteles foi filosofo grego que estudou logica biologia e politica
    pergunta qual é a capital da italia resposta roma é a capital da italia e cidade do coliseu
    pergunta como se calcula a media aritmetica resposta soma se todos os valores e divide pelo numero de elementos
    pergunta o que é a odontologia resposta odontologia cuida da saude dos dentes e da boca humana
    pergunta quem foi sigmund freud resposta sigmund freud criou a psicanalise e o conceito de inconsciente
    pergunta qual é o maior museu do mundo resposta o louvre em paris tem mais de trinta e cinco mil obras de arte
    pergunta o que é a logistica resposta logistica planeja o fluxo de materiais produtos e informacoes
    pergunta como se faz um texto dissertativo resposta tem introducao desenvolvimento e conclusao com argumentacao logica
    pergunta qual é a fruta mais consumida do mundo resposta a banana é a fruta mais consumida globalmente
    pergunta o que é a nanotecnologia resposta nanotecnologia manipula materiais em escala nanometrica
    pergunta quem foi virginia woolf resposta virginia woolf foi escritora modernista e feminista inglesa
    pergunta qual é a capital da russia resposta moscou é a capital da russia e cidade da praca vermelha
    pergunta como se escreve a palavra consciencia resposta consciencia se escreve com c e com s e com c e com i a
    pergunta o que é a meteorologia resposta meteorologia estuda os fenomenos da atmosfera e o clima
    pergunta quem foi beethoven resposta beethoven foi compositor alemao que criou a nona sinfonia
    pergunta qual é o maior estadio de futebol do mundo resposta o estadio de rungrado tem cento e catorze mil lugares
    pergunta o que é a empatia no trabalho resposta empatia no trabalho melhora a colaboracao e o clima organizacional


    """

class PositionalEncoding(nn.Module):
    """Posicional Encoding seno/cosseno do Transformer original"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Criar matriz de positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Termos de frequência
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Seno para posições pares, cosseno para ímpares
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention implementada corretamente"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Projeções
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 1. Projetar
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Reshape para heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Calcular atenção
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 4. Aplicar aos valores
        output = torch.matmul(attn_weights, V)
        
        # 5. Juntar heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 6. Projeção final
        output = self.out_proj(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """Feed-Forward Network com GELU"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU é melhor que ReLU para transformers
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Bloco Transformer completo (Pre-Norm)"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Subcamada 1: Atenção
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Subcamada 2: FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm (estilo moderno)
        # 1. Atenção com residual
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # 2. FFN com residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x


class TransformerModel(nn.Module):
    """Modelo Transformer completo para geração de texto"""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, 
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 1. Embedding de tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding (ISSO ESTAVA FALTANDO!)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 3. Blocos Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Output Head
        self.output_head = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Inicialização dos pesos (importante!)
        self._init_weights()
    
    def _init_weights(self):
        """Inicialização Xavier/Glorot para melhor convergência"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        x: tokens (batch, seq_len)
        mask: máscara causal para geração (batch, seq_len, seq_len)
        """
        # 1. Token embedding + scaling
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # 2. Adicionar positional encoding
        x = self.pos_encoding(x)
        
        # 3. Passar pelos blocos Transformer
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # 4. Projetar para vocabulário
        logits = self.output_head(x)
        
        return logits
    
    def generate_causal_mask(self, size):
        """Criar máscara causal (triangular inferior) para geração"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = ~mask  # Inverter para que 1 = pode ver, 0 = não pode ver
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Gerar texto token por token
        input_ids: (batch, seq_len) - tokens iniciais
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        for _ in range(max_new_tokens):
            # Criar máscara causal para os tokens atuais
            seq_len = input_ids.shape[1]
            mask = self.generate_causal_mask(seq_len).to(input_ids.device)
            
            # Forward pass
            logits = self(input_ids, mask)
            
            # Pegar logits do último token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling (se especificado)
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample próximo token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Adicionar à sequência
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

def simple_bpe(text, num_merges=10):
    # Tokenização inicial
    tokens = list(text)
    vocab = set(tokens)
    merges = {}
    
    for _ in range(num_merges):
        # Conta pares
        pairs = Counter(zip(tokens[:-1], tokens[1:]))
        if not pairs:
            break
        
        # Merge mais frequente
        best_pair = max(pairs, key=pairs.get)
        new_token = ''.join(best_pair)
        
        # Aplica merge
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        tokens = new_tokens
        merges[best_pair] = new_token
        vocab.add(new_token)
    
    return tokens, list(vocab), merges

# Limpeza mínima — mantém a estrutura do texto
text = text.lower()
text = re.sub(r"[^a-z\s]", "", text)   # só letras e espaços
text = re.sub(r"\s+", " ", text).strip()

words = text.split()
print(f"Total de palavras: {len(words):,}")

# ── 3. Vocab ──────────────────────────────────────────────────────────────────
vocab      = ["[PAD]", "[UNK]"] + sorted(set(words))
word2idx   = {w: i for i, w in enumerate(vocab)}
idx2word   = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Vocab size: {vocab_size:,}")

# ── 4. Stream de tokens ───────────────────────────────────────────────────────
all_tokens = [word2idx.get(w, word2idx["[UNK]"]) for w in words]
all_tokens = torch.tensor(all_tokens, dtype=torch.long)

print(f"Total de tokens: {len(all_tokens):,}")

SEQ_LEN = 32

class TextDataset(Dataset):
  def __init__(self, tokens, seq_len):
    self.tokens = tokens
    self.seq_len = seq_len

  def __len__(self):
    return len(self.tokens) - self.seq_len

  def __getitem__(self, idx):
    x = self.tokens[idx         : idx + self.seq_len]
    y = self.tokens[idx + 1     : idx + self.seq_len + 1]
    return x, y

dataset    = TextDataset(all_tokens, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Total de amostras: {len(dataset):,}")
print(f"Batches por epoch: {len(dataloader):,}")

def generate(model, prompt, max_new_tokens=5):
    model.eval()
    tokens = [word2idx.get(w, word2idx["[UNK]"]) for w in prompt.lower().split()]
    tokens = torch.tensor([tokens])

    result = prompt.split()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp        = tokens[:, -SEQ_LEN:]
            logits     = model(inp)
            probs      = F.softmax(logits[:, -1, :], dim=-1)
            next_tok   = torch.multinomial(probs, num_samples=1)
            result.append(idx2word[next_tok.item()])
            tokens     = torch.cat([tokens, next_tok], dim=1)

    return " ".join(result)

if __name__ == "__main__":

    
    model = TransformerModel(vocab_size=vocab_size, d_model=64, num_heads=4, d_ff=256, num_layers=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 2000
    totalLoss = []

    for epoch in range(epochs):
      model.train()

      for batch in dataloader:
        optimizer.zero_grad()

        X, y = batch
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()

    if epoch % 101 == 0:
        totalLoss.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        

    output = generate(model, "o rei")
    print(output)

    # Plotar perda
    plt.plot(totalLoss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss ao longo do treinamento")
    plt.show()
