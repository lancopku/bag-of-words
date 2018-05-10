'''
 @Date  : 2017/12/18
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models


class rnn_encoder(nn.Module):

    def __init__(self, config, embedding=None):
        super(rnn_encoder, self).__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        self.hidden_size = config.hidden_size
        self.config = config
        if config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size)
        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)
            if config.attemb:
                self.rnnpos = nn.GRU(input_size=self.config.emb_size, hidden_size=self.config.hidden_size, num_layers=1, dropout=self.config.dropout, bidirectional=False)

    def forward(self, inputs, lengths):
        if self.config.resRNN:
            embs = pack(self.embedding(inputs), lengths)
            inputs = unpack(embs)[0]
            embeds = inputs
        else:
            embs = pack(self.embedding(inputs), lengths)
            embeds = unpack(embs)[0]
            if self.config.attemb:
                outputs, state = self.rnnpos(embs)
                outputs = unpack(outputs)[0]
                self.attention.init_context(context = outputs)
                out_attn = []
                for i, emb in enumerate(embeds.split(1)):
                    output, attn = self.attention(emb.squeeze(0), embeds)
                    out_attn.append(output)
                embs = torch.stack(out_attn)
            outputs, state = self.rnn(embeds)
            if self.config.attemb:
                outputs = outputs
            else:
                outputs = outputs
            if self.config.bidirectional:
                outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        if self.config.selfatt:
            self.attention.init_context(context=outputs)
            out_attn = []
            for i, out in enumerate(outputs.split(1)): 
                output, weights = self.attention(out.squeeze(0), embeds[i])
                out_attn.append(output)
            outputs = torch.stack(out_attn)

        if self.config.cell == 'gru':
            state = state[:self.config.dec_num_layers]
        else:
            state = (state[0][:self.config.dec_num_layers], state[1][:self.config.dec_num_layers])

        return outputs, state, embeds



class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None, use_attention=True):
        super(rnn_decoder, self).__init__()
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.emb_size)

        input_size = config.emb_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)

        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong':
            self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size)

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, input, state, embeds, memory):
        embs = self.embedding(input)
        output, state = self.rnn(embs, state)
        if self.attention is not None:
            if self.config.attention == 'luong_gate':
                output, attn_weights, memory = self.attention(output, embeds, memory, hops=self.config.hops)
            else:
                output, attn_weights = self.attention(output, embs)
        else:
            attn_weights = None
        output = self.dropout(output)
        output = self.compute_score(output)

        return output, state, attn_weights, memory

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores




class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1
