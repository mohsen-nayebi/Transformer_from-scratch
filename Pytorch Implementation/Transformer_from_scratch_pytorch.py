# importing necessary packages 
import torch
from torch import nn, optim
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


# specifying gpu or cpu use
device='cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())


# a function to  give the model some information about the relative position of the tokens in the sentence.
def positional_encoding(seq_len, d_model, device='cpu'):
  pe = np.zeros((seq_len, d_model))
  for pos in range(seq_len):
    for i in range(0, d_model, 2):
      pe[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
      pe[pos, i+1] = math.cos(pos / (10000 ** (i/d_model)))
  return torch.from_numpy(pe[np.newaxis, :]).to(device)


#optional positional encoding visualisation:
#The 256-dimensional positonal encoding for a sentence with the maximum lenght of 1024

n, d = 1024, 256
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

pos_encoding = torch.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = torch.permute(pos_encoding, (2, 1, 0))
pos_encoding = torch.reshape(pos_encoding, (d, n))

plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()



# Also we can implement positional encoding in form of a class
class PositionalEncoding(nn.Module): 
  def __init__(self, seq_len, d_model, device):
    super().__init__()
    self.seq_len = seq_len
    self.d_model= seq_len

    pe = np.zeros((self.seq_len, self.d_model))
    for pos in range(self.seq_len):
      for i in range(0, self.d_model, 2):
        pe[pos, i] = math.sin(pos / (10000 ** (i/self.d_model)))
        pe[pos, i+1] = math.cos(pos / (10000 ** (i/self.d_model)))


  def forward(self, x):
    N, seq_len = x.shape
    return torch.from_numpy(self.pe[np.newaxis, :]).to(device)





class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, heads):
    super().__init__()

    self.d_model = d_model  
    self.heads = heads                  # number of heads 
    self.head_dim = d_model // heads    # dimension of each head after splitting attention layer


    assert (self.head_dim * self.heads == self.d_model), "model dimension needs to be div by number of heads"

    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.fc_out = nn.Linear(self.head_dim*heads, d_model, bias=False)

  def forward(self, v, k, q, mask):
    
    batch_size = q.shape[0] #num of training exmples in a batch (batch_size) 
    value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]   # inpute/target sequence lengths


    v = v.reshape(batch_size, value_len, self.heads, self.head_dim)
    k = k.reshape(batch_size, key_len, self.heads, self.head_dim)
    q = q.reshape(batch_size, query_len, self.heads, self.head_dim)


    values = self.values(v)
    keys = self.keys(k)
    queries = self.queries(q)

    energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
    #queries shape = (batch_size, query_len, heads, head_dim)
    #keys shape = (batch_size, key_len, heads, head_dim)
    #energy shape = (batch_size, heads, query_len, key_len)

    if mask is not None:
      energy = energy.masked_fill(mask==0, value = float("-1e-20"))


    attention_weights = torch.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
    out= torch.einsum("nhqk, nkhd->nqhd", [attention_weights, values]).reshape(batch_size, query_len, self.heads*self.head_dim)
    #attention_weights.shape = (batch_size, heads, query_len, key_len)
    #values.shape = (batch_size, value_len, heads, head_dim)
    #out.shape = (batch_size, query_len, d_model)

    out = self.fc_out(out)

    return out, attention_weights



class EncoderLayer(nn.Module):
  def __init__(self, d_model, heads, forward_expansion, dropout):
    super().__init__()
    self.mha = MultiHeadAttention(d_model, heads)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, d_model*forward_expansion),
        nn.ReLU(),
        nn.Linear(d_model*forward_expansion, d_model))
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x, mask):
    out1, _ = self.mha(x, x, x, mask)
    out2 = self.norm1(self.dropout1(out1+x))
    out3 = self.feed_forward(out2)
    out = self.norm2(self.dropout2(out3 + out2))
    return out


class DecoderLayer(nn.Module):
  def __init__(self, d_model, heads, forward_expansion, dropout):
    super().__init__()
    self.mha1 = MultiHeadAttention(d_model, heads)
    self.mha2 = MultiHeadAttention(d_model, heads)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, d_model*forward_expansion),
        nn.ReLU(),
        nn.Linear(d_model*forward_expansion, d_model))
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, x, enc_out, padding_mask, look_ahead_mask):
    attention, attention_weights_block1  = self.mha1(x, x, x, look_ahead_mask)
    q = self.norm1(self.dropout1(x + attention))
    out1, attention_weights_block2 = self.mha2(enc_out, enc_out, q, padding_mask)
    out2 = self.norm2(self.dropout2(out1+q))
    out3 = self.feed_forward(out2)
    out = self.norm3(self.dropout3(out3 + out2))
    return out, attention_weights_block1, attention_weights_block2




class Encoder(nn.Module):
  def __init__(self, input_vocab_size, d_model, num_layers, heads, max_length, device, forward_expansion, dropout):
    super().__init__()

    self.pos_encoding = positional_encoding(max_length, d_model, device)
    self.d_model = d_model
    self.device = device
    self.word_embedding = nn.Embedding(input_vocab_size, d_model)
    self.layers = nn.ModuleList(
                        [
                        EncoderLayer(d_model, heads, forward_expansion=forward_expansion, dropout=dropout)
                        for _ in range(num_layers)
                        ]
                        )
    self.dropout = nn.Dropout(dropout)
    

  def forward(self, x, padding_mask):
    N, seq_length = x.shape
    out = self.dropout((self.word_embedding(x) + self.pos_encoding[:, :seq_length, :])).type(torch.FloatTensor).to(self.device)
    for layer in self.layers:
      out = layer(out, padding_mask)
    return out





class Decoder(nn.Module):
  def __init__(self,trg_vocab_size, d_model, num_layers, heads, max_length, device, forward_expansion, dropout):
    super().__init__()

    self.pos_encoding = positional_encoding(max_length, d_model, device)
    self.trg_vocab_size = trg_vocab_size
    self.device = device
    self.word_embedding = nn.Embedding(trg_vocab_size, d_model)
    self.positional_embedding = nn.Embedding(max_length, d_model)
    self.layers = nn.ModuleList([
                       DecoderLayer(d_model, heads, forward_expansion, dropout)
                       for _ in range(num_layers)          
                                ])
    self.fc_out = nn.Linear(d_model, trg_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, enc_out, padding_mask, look_ahead_mask):
    N, seq_length = x.shape
    attention_weights = {}

    out = self.dropout(self.pos_encoding[:, :seq_length, :] + self.word_embedding(x)).type(torch.FloatTensor).to(self.device)

    for i, layer in enumerate(self.layers):
      out, att_block1, att_block2 = layer(out, enc_out, padding_mask, look_ahead_mask)
      attention_weights[f'decoder_layer{i+1}_block1'] = att_block1
      attention_weights[f'decoder_layer{i+1}_block2'] = att_block2

    out = self.fc_out(out)

    return out, attention_weights



class Transformer(nn.Module):
  def __init__(self, input_vocab_size, trg_vocab_size,
               d_model=512, num_layers=6, heads=8, forward_expansion=4, dropout=0.1, device="cuda", max_length=100):
    super().__init__()
    self.encoder = Encoder(input_vocab_size, d_model, num_layers, heads, max_length, device, forward_expansion, dropout)
    self.decoder = Decoder(trg_vocab_size, d_model, num_layers, heads, max_length, device, forward_expansion, dropout)
    self.device = device 
    
  def make_padding_mask(self, src):
    padding_mask = (src != 0).unsqueeze(1).unsqueeze(2) #shape = (N, 1, 1, src_len)
    return padding_mask.to(device)

  def make_look_ahead_mask(self, trg):
    N, trg_len = trg.shape
    look_ahead_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(N, 1, trg_len, trg_len)
    return look_ahead_mask.to(device)
  
  def forward(self, input, target):
    padding_mask = self.make_padding_mask(input)
    look_ahead_mask = self.make_look_ahead_mask(target)
    enc_out = self.encoder(input, padding_mask)
    out = self.decoder(target, enc_out, padding_mask, look_ahead_mask)
    return out




#testing the transformer architecture
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    src_input = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_vocab_size = 10
    trg_vocab_size = 10
    
    model = Transformer(src_vocab_size, trg_vocab_size, device=device).to(device)
    out, _ = model(src_input, trg[:, :-1])
    print(out.shape)
