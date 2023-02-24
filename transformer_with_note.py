import os
import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt

# init parameters
UNK = 0  # unknow word-id
PAD = 1  # padding word-id
BATCH_SIZE = 64   

DEBUG = True    # Debug / Learning Purposes. 
# DEBUG = False # Build the model, better with GPU CUDA enabled.

if DEBUG:        
    EPOCHS  = 2   
    LAYERS  = 3   
    H_NUM   = 8     # (Header Number) 这就是 Multi-Headers 里面的Header数量，必须被 D_MODEL 整除。
    D_MODEL = 128   # Embedding空间维度, PositionalEncoding用的维度也是这个
    D_FF    = 256      
    DROPOUT = 0.1   
    MAX_LENGTH = 60 # 翻译出来句子的最大长度
    TRAIN_FILE = 'data/nmt/en-cn/train_mini.txt'   
    DEV_FILE   = 'data/nmt/en-cn/dev_mini.txt'   
    SAVE_FILE  = 'save/models/model.pt'   
else:
    EPOCHS  = 20  
    LAYERS  = 6    
    H_NUM   = 8    
    D_MODEL = 256    
    D_FF    = 1024   
    DROPOUT = 0.1    
    MAX_LENGTH = 60  
    TRAIN_FILE = 'data/nmt/en-cn/train.txt'   
    DEV_FILE   = 'data/nmt/en-cn/dev.txt'   
    SAVE_FILE  = 'save/models/large_model.pt'  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seq_padding(X, padding=0):
    """
    add padding to a batch data 
    按本batch中最长的句子补足
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])    

class PrepareData:
    def __init__(self, train_file, dev_file):
        # 01. Read the data and tokenize
        self.train_en, self.train_cn = self.load_data(train_file)
        self.dev_en, self.dev_cn     = self.load_data(dev_file)

        # 02. build dictionary: English and Chinese
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # 03. word to id by dictionary 
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn     = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        # 04. batch + padding + mask
        self.train_data = self.splitBatch(self.train_en, self.train_cn, BATCH_SIZE)
        self.dev_data   = self.splitBatch(self.dev_en, self.dev_cn, BATCH_SIZE)

    def load_data(self, path):
        """
        Read English and Chinese Data 
        tokenize the sentence and add start/end marks(Begin of Sentence; End of Sentence)
        en = [['BOS', 'i', 'love', 'you', 'EOS'], 
              ['BOS', 'me', 'too', 'EOS'], ...]
        cn = [['BOS', '我', '爱', '你', 'EOS'], 
              ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
                a = " ".join([w for w in line[1]])
                cn.append(["BOS"] + word_tokenize(a) + ["EOS"])
        return en, cn
    
    def build_dict(self, sentences, max_words = 50000):
        """
        sentences: list of word list 
        build dictonary as {key(word): value(id)}
        """
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        # inverted index: {key(id): value(word)}
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        """
        convert input/output word lists to id lists. 
        Use input word list length to sort, reduce padding.
        """
        length = len(en)
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        def len_argsort(seq):
            """
            get sorted index w.r.t length.
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort: # update index
            sorted_index = len_argsort(out_en_ids) # English
            out_en_ids = [out_en_ids[id] for id in sorted_index]
            out_cn_ids = [out_cn_ids[id] for id in sorted_index]
        return out_en_ids, out_cn_ids

    def splitBatch(self, en, cn, batch_size, shuffle=True):
        """
        get data into batches
        """
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        
        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]  
            batch_cn = [cn[index] for index in batch_index]
            # paddings: batch, batch_size, batch_MaxLength
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn)) 
            #!!! 'Batch' Class is called here but defined in later section.
        return batches

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab): 
        """
            vocab=1435，词库大小
            d_model: embedding的维数，越大模型capacity越大，测试128，正式256。
        """
        super(Embeddings, self).__init__() 
        self.lut = nn.Embedding(vocab, d_model) # (1435, 128)
        self.d_model = d_model

    def forward(self, x):
        # return x's embedding vector（times math.sqrt(d_model)）
        # x值属于[0, vocab]，返回 (N, squence_size, d_model)
        return self.lut(x) * math.sqrt(self.d_model) # 最后乘以sqrt的原因Paper是参考

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
                                  
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))
        pe_pos   = torch.mul(position, div_term)
        pe[:, 0::2] = torch.sin(pe_pos) # 偶数间隔用sin
        pe[:, 1::2] = torch.cos(pe_pos) # 奇数间隔用cos
        pe = pe.unsqueeze(0) 
                                  
        self.register_buffer('pe', pe) # pe

    def forward(self, x):
        #  build pe w.r.t to the max_length
        # 位置编码就这么活生生的加到编码维度上了？可能因为编码是学出来的，也不固定，位置影响编码靠加法也许行。
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
        return self.dropout(x)        

# pe = PositionalEncoding(32, 0, 100)  # d_model, dropout-ratio, max_len
# positional_encoding = pe.forward(Variable(torch.zeros(1, 100, 32).cuda())).cpu() # sequence length, d_model
# plt.figure(figsize=(10,10))
# sns.heatmap(positional_encoding.squeeze()) # 100x32 matrix
# plt.title("Sinusoidal Function")
# plt.xlabel("hidden dimension")
# plt.ylabel("sequence length")
# None        

def attention(query, key, value, mask=None, dropout=None): # query/key/value: [N, 8, W(单词个数), 16(128(编码维度)//8(注意力隐空间维度)=16)] given x=[N, W(单词个数), 128(编码维度)]
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) 

    # [N, 8, W, 16] * [N, 8, 16, W] => [N, 8, W, W]
    # scores结果是自己word的query和句子其它全部word的key乘法结果。
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # [N, 8, W(单词个数), W(单词个数)]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1): # h: 8
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0 # check the h number
        self.d_k = d_model // h # d_k: 16
        self.h = h
        # 4 linear layers: WQ WK WV and final linear mapping WO  
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None # 无用
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_query, x_key, x_value, mask=None): # Encode时: query(x) == key(x) == value(x)，Decoder第二个layer时: query(x), memory, memory
        # 进来的时候 query/key/value 都是 x: [N, W(单词个数), 128]

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = x_query.size(0) # get batch size
        # 1) Do all the linear projections in batch from d_model => h x d_k (把d_model(128)分成了 h(8)*d_k(16))
        # parttion into h sections，switch 2,3 axis for computation. 
        # 不可能拿一样的值(x)去当Q/K/V，这里用线性投射一下x到不同的值(Q/K/V)。
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) # transpose后x变成 [N, 8, W, 16]
                             for l, x in zip(self.linears, (x_query, x_key, x_value))] # [N, 8, W, 16]
        # 2) Apply attention on all the projected vectors in batch.
        # 基本是： x = (query*key) * value, attn = softmax(query*key)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout) # TODO: self.attn 无用，应该可以去掉
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) # "把d_model分成了 h*d_k" 这里又合回去了，变成128维

        # illustrated-transformer文章说这里可以把concate的x多除来的维度用linear缩小回去，但本实现通过view并且d*d_k==d_model就不用改变维度了
        # paper说法其实也是本实现这样保持维度的: "Due to the reduced dimension of each head, the total computational cost 
        # is similar to that of single-head attention with full dimensionality"
        return self.linears[-1](x) # final linear layer    

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        # convert words id to long format.  
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()
        self.src = src
        # get the padding postion binary mask
        # change the matrix shape to  1×seq.length
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            # decoder input from target 
            self.trg = trg[:, :-1]
            # decoder target from trg 
            self.trg_y = trg[:, 1:]
            # add attention mask to decoder input  
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # check decoder output padding number
            self.ntokens = (self.trg_y != pad).data.sum()
    
    # Mask 
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask # subsequent_mask is defined in 'decoder' section.        

class LayerNorm(nn.Module):
    """
    BN是取batch内的全部样本对同一通道进行归一化，而LN是取同一样本自己内部做channel*feature这个面的归一化。
    参考：https://zhuanlan.zhihu.com/p/74516930    
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # rows
        std = x.std(-1, keepdim=True)
        x_zscore = (x - mean)/ torch.sqrt(std ** 2 + self.eps) 
        return self.a_2*x_zscore+self.b_2         

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    SublayerConnection: connect Multi-Head Attention and Feed Forward Layers 

    内部自带resnet, LN, dropout
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer): # sublayer: lambda包裹着 MultiHeadedAttention 或 feed-forward
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))        

class PositionwiseFeedForward(nn.Module):
    """
    Paper 3.3: Position-wise Feed-Forward Networks
    三明治，两个FC中间夹着一个dropout(paper说的是ReLU! 这里实现错了？)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = self.w_1(x)
        # h2 = self.dropout(h1)
        h2 = torch.relu(h1) # 用relu后current best loss: 0.809 减少到 0.334
        return self.w_2(h2)        

def clones(module, N):
    """
    "Produce N identical layers."
    Use deepcopy the weight are indenpendent.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])        

class Encoder(nn.Module): # XXX Encoder
    "Core encoder is a stack of N layers (blocks)"
    """
     # ODD: df  
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)    

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout): # dropout: float
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn # MultiHeadedAttention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2) # 内部有两层，每层最后都是一个LayerNormal + Residual + Dropout(都在SublayerConnection里面实现)
        self.size = size # d_model

    def forward(self, x, mask):
        # X-embedding to Multi-head-Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))    # multi-attention + 全家桶(LayerNormal + Residual + Dropout)
        # X-embedding to feed-forwad nn
        return self.sublayer[1](x, self.feed_forward)                       # feed_forward(PositionwiseFeedForward: FC+Dropout+FC) + 全家桶(LayerNormal + Residual + Dropout)

class Decoder(nn.Module): # XXX: Decoder
    def __init__(self, layer, N):
        "Generic N layer decoder with masking."
        super(Decoder, self).__init__()
        self.layers = clones(layer, N) # layer: DecoderLayer
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Repeat decoder N times
        Decoderlayer get a input attention mask (src) 
        and a output attention mask (tgt) + subsequent mask 
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask) # Pred时src_mask是逐渐覆盖的，Train时除了最后padding外都是True
        return self.norm(x)        

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn  # MultiHeadedAttention
        self.src_attn = src_attn    # MultiHeadedAttention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3) # 比Encoder多了一个Layer

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory # encoder output embedding

        # 只有Pred时需要对self_attn遮罩，所以第一个子层是Masked Multi-Header Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))    # Decoder第一层： 对self_attn用的是x自己。这一层是 Masked Multi-Header Attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))     # Decoder第二层： 对src_attn用的是来自Encoder输出的memory。这一层是Unmasked Multi-Header Attention
        # Context-Attention：q=decoder hidden，k,v from encoder hidden
        return self.sublayer[2](x, self.feed_forward)                           # Decoder第三层： feed-forward

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0        

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        """
        c = copy.deepcopy
        model = Transformer(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
            nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
            Generator(d_model, tgt_vocab)).to(DEVICE)
        )
        """
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # [Embeddings, PositionalEncoding](dropout=0.1)
        self.tgt_embed = tgt_embed
        self.generator = generator 

    def encode(self, src, src_mask):
        # src: [N, length of words]
        embed_out = self.src_embed(src) # [N, sequence_length, 128(d_model)]
        return self.encoder(embed_out, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        # encoder output will be the decoder's memory for decoding
        en_out = self.encode(src, src_mask)
        return self.decode(en_out, src_mask, tgt, tgt_mask)  # Transformer这个Model最后没用generator，是后面放到SimpleLossCompute时使用的。

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode: d_model to vocab mapping
        self.proj = nn.Linear(d_model, vocab) # 把embedding维度升级的目标单词数量维度

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)        

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h = 8, dropout=0.1):
    """
    N: Encoder和Decoder的主layer(EncoderLayer, DecoderLayer)层数，论文写的是6。
    d_model: max length of words.
    h: attention层Q/K/V矩阵的列数，也就是Q/K/V vector的长度
    """
    c = copy.deepcopy
    #  Attention 
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    #  FeedForward 
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    #  Positional Encoding
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    #  Transformer
    # Paper说"In our model, we share the same weight matrix between the two embedding layers and the pre-softmax
    # linear transformation"，但这里src, tgr没有共享embedding，单词数量维度也不一样没法共享啊。
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE), # N: 把EncoderLayer复制(deepclone)N份。
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    # Paper title: Understanding the difficulty of training deep feedforward neural networks Xavier
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)        


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    """
    公式： y_ls = (1 - a) * y_hot + a / K
    解决 Over-confident 问题。
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        size: 目标词库最大word数量
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum') # 2020 update
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone() # TODO: replace tensor.data with tensor.detach().clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        # 因为x, true_dist都是prob了，CrossEntropyLoss 要求target是个code，只能用KLDivLoss计算Loss
        # 如果不是为了LS应该可以改成CrossEntropyLoss
        return self.criterion(x, Variable(true_dist, requires_grad=False)) 


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x) # FC -> F.log_softmax
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()        

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
# We used factor=2, warmup-step = 4000
def get_std_opt(model): # 这个函数没用上，训练的时候重新直接给了一套值。
    " Paper: 5.3 Optimizer"
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))        

# Three settings of the lrate hyperparameters.
# 没用上，训练的时候重新直接给了一套值。
opts = [NoamOpt(512, 1, 4000, None), 
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]            

def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    for i , batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch {:d} Batch: {:d} Loss: {:.4f} Tokens per Sec: {:.2f}s".format(epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens        

def train(data, model, criterion, optimizer):
    """
    Train and Save the model.
    """
    # init loss as a large value
    best_dev_loss = 1e5
    
    for epoch in range(EPOCHS):
        # Train model 
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        model.eval()

        # validate model on dev dataset
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: {:.2f}'.format(dev_loss))
        
        # save the model with best-dev-loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), SAVE_FILE) # SAVE_FILE = 'save/model.pt'
            
        print(f">>>>> current best loss: {best_dev_loss}")    

# Step 1: Data Preprocessing
data = PrepareData(TRAIN_FILE, DEV_FILE)
src_vocab = len(data.en_word_dict)
tgt_vocab = len(data.cn_word_dict)
print(f"src_vocab {src_vocab}")
print(f"tgt_vocab {tgt_vocab}")

# Step 2: Init model
model = make_model(
                    src_vocab, 
                    tgt_vocab, 
                    LAYERS, 
                    D_MODEL, 
                    D_FF,
                    H_NUM,
                    DROPOUT
                )

# Step 3: Training model
print(">>>>>>> start train")
train_start = time.time()
criterion = LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= 0.0) # smoothing为啥用0，这LS就没用了
optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))

train(data, model, criterion, optimizer)
print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")        

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Translate src with model
    """
    # decode the src 
    # Pred时Encoder没iterater，只有Decoder需要。
    memory = model.encode(src, src_mask) # 产生的memory传给Decoder三个子层中间的那一层(Unmasked Multi-Header Attention)当参数
    
    # init 1×1 tensor as prediction，fill in ('BOS')id, type: (LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    #  run 遍历输出的长度下标
    for i in range(max_len-1):
        # decode one by one
        out = model.decode(memory, 
                           src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data))) # [1, W(逐渐变长), 128], out和ys里面单词数量每次都是一样，因为ys第一个是BOS，out实际内容多出下一个字。
        #  out to log_softmax 
        prob = model.generator(out[:, -1]) # [1, 1446], 每次只出一个字
        #  get the max-prob id
        _, next_word = torch.max(prob, dim = 1) # 这里用了max，没用multinormal概率draw sample.
        next_word = next_word.data[0]
        #  concatnate with early predictions
        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        # 为啥不检查碰到3(EOS)就break？应该是因为不知道EOS的编码？
        if next_word == 3: # "EOS"
            break
    return ys

def evaluate(data, model):
    """
    Make prediction with trained model, and print results.
    """
    with torch.no_grad():
        #  pick some random sentences from dev data.         
        for i in np.random.randint(len(data.dev_en), size=10):
            # Print English sentence
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)
            
            # Print Target Chinese sentence
            cn_sent =  " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))
            
            # conver English to tensor  
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            src = src.unsqueeze(0)
            # set attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # apply model to decode, make prediction
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
            # save all in the translation list 
            translation = []
            # convert id to Chinese, skip 'BOS' 0.
            # 遍历翻译输出字符的下标（注意：跳过开始符"BOS"的索引 0）
            for j in range(1, out.size(1)):
                sym = data.cn_index_dict[out[0, j].item()]
                if sym != 'EOS':
                    translation.append(sym)
                else:
                    break
            print("translation: {}".format(" ".join(translation)))        

# Predition
model.load_state_dict(torch.load(SAVE_FILE))
print(">>>>>>> start evaluate")
evaluate_start  = time.time()
evaluate(data, model)         
print(f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")