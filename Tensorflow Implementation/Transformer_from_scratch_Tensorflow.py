# importing necessary packages 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import math



# a function to  give the model some information about the relative position of the tokens in the sentence.
def positional_encoding(position, d_model):

  pe = np.zeros((position, d_model))
  for pos in range(position):
    for i in range(0, d_model, 2):
      pe[pos, i] = math.sin(pos / 10000**(i/d_model))
      pe[pos, i+1] = math.cos(pos / 10000**(i/d_model))

  return pe[np.newaxis, :]

n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)


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
    return tf.constant(self.pe[np.newaxis, :])




class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, heads, d_model):
    super().__init__()

    self.head_dim = d_model // heads
    self.heads = heads                 # number of heads
    self.d_model = d_model             # dimension of each head after splitting attention layer

    assert (self.head_dim * self.heads == self.d_model), "d_model needs to be divisible by number of heads"

    self.wk = tf.keras.layers.Dense(self.head_dim)
    self.wq = tf.keras.layers.Dense(self.head_dim)
    self.wv = tf.keras.layers.Dense(self.head_dim)
    self.fc = tf.keras.layers.Dense(d_model)

  
  def call(self, v, k, q, mask):

    batch_size = q.shape[0]                                              #num of training exmples in a batch (batch_size) 
    value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]   #inpute/target sequence lengths

    v = tf.reshape(v, (batch_size, value_len, self.heads, self.head_dim))
    k = tf.reshape(k, (batch_size, key_len, self.heads, self.head_dim))
    q = tf.reshape(q, (batch_size, query_len, self.heads, self.head_dim))

    key = self.wk(k)
    value = self.wv(v)
    query = self.wq(q)

    matmul_qk = tf.einsum("nqhd, nkhd->nhqk", q, k)
    #queries shape = (batch_size, query_len, heads, head_dim)
    #keys shape = (batch_size, key_len, heads, head_dim)
    #energy shape = (batch_size, heads, query_len, key_len)

    if mask is not None:
      matmul_qk += (mask * -1e9)

    attention = tf.nn.softmax(matmul_qk / tf.math.sqrt(tf.cast(self.d_model, tf.float32)), axis=3)
    out = tf.einsum("nhqk, nkhd->nqhd", attention, v)
    out = tf.reshape(out, (batch_size, query_len, self.d_model))
    #attention.shape = (batch_size, heads, query_len, key_len)
    #values.shape = (batch_size, value_len, heads, head_dim)
    #out.shape = (batch_size, query_len, d_model)

    out = self.fc(out)

    return out, attention




class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, heads, dropout=0.1, expansion=4):
    super().__init__()
    
    self.mha = MultiHeadAttention(heads, d_model)
    self.dropout1 = tf.keras.layers.Dropout(dropout)
    self.dropout2 = tf.keras.layers.Dropout(dropout)
    self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.ffn = tf.keras.Sequential([
      tf.keras.layers.Dense(expansion * d_model, activation='relu'),  
      tf.keras.layers.Dense(d_model) 
  ])

  def call(self, x, training, mask):
    out1, _ = self.mha(x, x, x, mask)
    out2 = self.dropout1(self.norm1(out1 + x), training=training)
    out3 = self.ffn(out2)
    out = self.dropout2(self.norm2(out3 + out2), training=training)

    return out




class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, heads, dropout=0.1, expansion=4):
    super().__init__()
    
    self.mha1 = MultiHeadAttention(heads, d_model)
    self.mha2 = MultiHeadAttention(heads, d_model)
    self.dropout1 = tf.keras.layers.Dropout(dropout)
    self.dropout2 = tf.keras.layers.Dropout(dropout)
    self.dropout3 = tf.keras.layers.Dropout(dropout)
    self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.ffn = tf.keras.Sequential([
      tf.keras.layers.Dense(expansion * d_model, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

  def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
    out1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
    out2 = self.dropout1(self.norm1(out1 + x), training=training)
    out3, attn_weights_block2 = self.mha2(enc_out, enc_out, out2, padding_mask)
    out4 = self.dropout2(self.norm1(out3 + out2), training=training)
    out5 = self.ffn(out4)
    out6 = self.dropout3(self.norm3(out5 + out4), training=training)

    return out6, attn_weights_block1, attn_weights_block2





class Encoder(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, d_model, heads, layers=8, dropout=0.1, expansion=4, max_length=100):
    super().__init__()
    self.input_vocab_size = input_vocab_size
    self.d_model = d_model
    self.layers = layers
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(max_length, d_model)
    self.enc_layers = [EncoderLayer(d_model, heads, dropout=0.1, expansion=4) for _ in range(layers)]
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, x, training, mask):

    batch_size, seq_len = tf.shape(x)
    x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.dropout((x + self.pos_encoding[:, :seq_len, :]), training=training)
    for layer in self.enc_layers:
      x = layer(x, training, mask)
    
    return x





class Decoder(tf.keras.layers.Layer):
  def __init__(self, trg_vocab_size, d_model, heads, layers=8, dropout=0.1, expansion=4, max_length=100):
    super().__init__()

    self.trg_vocab_size = trg_vocab_size
    self.d_model = d_model
    self.layers = layers
    self.embedding = tf.keras.layers.Embedding(trg_vocab_size, d_model)
    self.pos_encoding = positional_encoding(max_length, d_model)
    self.dec_layers = [DecoderLayer(d_model, heads, dropout=0.1, expansion=4) for _ in range(layers)]
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, x, enc_out, training, look_ahead_mask, padding_mask):

    batch_size, seq_len = tf.shape(x)
    #attention_weights = {}

    x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.dropout((x + self.pos_encoding[:, :seq_len, :]), training=training)

    for layer in self.dec_layers:
      x, block1, block2 = layer(x, enc_out, training, look_ahead_mask, padding_mask)
      #attention_weights[f'decoder_layer{i+1}_block1'] = block1
      #attention_weights[f'decoder_layer{i+1}_block2'] = block2

    
    
    return x




class Transformer(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, trg_vocab_size, d_model, heads, layers=8, dropout=0.1, expansion=4, max_length=100):
    super().__init__()

    self.encoder = Encoder(input_vocab_size, d_model, heads, layers=layers, dropout=dropout, expansion=expansion, max_length=max_length)
    self.decoder = Decoder(trg_vocab_size, d_model, heads, layers=layers, dropout=dropout, expansion=expansion, max_length=max_length)
    self.final_layer = tf.keras.layers.Dense(trg_vocab_size)
  
  def creat_padding_mask(self, seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

  def create_look_ahead_mask(self, size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

  def call(self, input, targets, training):

    size = targets.shape[1]
    padding_mask = self.creat_padding_mask(input)
    padding_mask_trg = self.creat_padding_mask(targets)
    look_ahead_mask = self.create_look_ahead_mask(size)
    look_ahead_mask = tf.maximum(padding_mask_trg, look_ahead_mask)

    enc_out = self.encoder(input, training, padding_mask)
    dec_out = self.decoder(targets, enc_out, training, look_ahead_mask, padding_mask)

    out = self.final_layer(dec_out)

    return out
