import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class InputEmbedding(nn.Module):
    def __init__(self, 
                 d_model:int, 
                 vocab_size:int) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

class PositionalEncoding(nn.Module):
    def __init__(self, 
                 d_model:int, 
                 seq_len:int, 
                 dropout:float) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len,  dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self,
                 features: int, 
                 eps:float = 10**-6) -> None:
        
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, 
                 d_model:int, 
                 d_ff:int, 
                 dropout:float) -> None:
        
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 hidden: int, 
                 dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        assert d_model % hidden == 0, "d_model must be divisible by hidden"

        self.d_k = d_model // hidden
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = F.softmax(attention_score, dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value), attention_score
    
    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.hidden, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.hidden, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.hidden, self.d_k).transpose(1,2)

        x, self.attention_score = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], x.shape[1], self.hidden * self.d_k)
        
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__ (self,
                  features:int, 
                  dropout:None) -> None:
        
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):

    def __init__ (self, 
                  features:int,
                  self_attention_block: MultiHeadAttentionBlock, 
                  feed_forward_block: FeedForwardBlock, 
                  dropout: float) -> None:
        
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__ (self, 
                  features:int,
                  layers:nn.ModuleList) -> None:
        
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__ (self, 
                  features:int,
                  self_attention_block: MultiHeadAttentionBlock, 
                  cross_attention_block: MultiHeadAttentionBlock, 
                  feed_forward_block: FeedForwardBlock, 
                  dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])
    
    def forward(self, x, encder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encder_output, encder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__ (self, 
                  features:int,
                  layers:nn.ModuleList) -> None:
        
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, 
                 d_model:int, 
                 vocab_size:int) -> None:
        
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, 
                 encoder:Encoder,
                 decoder:Decoder,
                 src_embedding:InputEmbedding,
                 tgt_embedding:InputEmbedding,
                 src_positional_encoding:PositionalEncoding,
                 tgt_positional_encoding:PositionalEncoding,
                 projection_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_positional_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_positional_encoding(tgt)
        return self.decoder(tgt, encder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size:int,
                      tgt_vocab_size:int,
                      src_seq_len:int,
                      tgt_seq_len:int,
                      d_model:int=512,
                      N:int = 6,
                      hidden:int = 8,
                      d_ff:int = 2048,
                      dropout:float = 0.1) -> Transformer:
    
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)

    src_positional_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_positional_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, hidden, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderLayer(d_model, 
                                     encoder_self_attention_block, 
                                     feed_forward_block, 
                                     dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, hidden, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, hidden, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderLayer(d_model, 
                                     decoder_self_attention_block, 
                                     decoder_cross_attention_block, 
                                     feed_forward_block, 
                                     dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, 
                              decoder, 
                              src_embedding, 
                              tgt_embedding, 
                              src_positional_encoding, 
                              tgt_positional_encoding, 
                              projection)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
    








