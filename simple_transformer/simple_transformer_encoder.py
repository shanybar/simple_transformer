import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_embed, d_output,len_vocab):
        super(SelfAttention, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len_vocab,embedding_dim=d_embed)
        self.d_embed = d_embed
        self.w_key = nn.Linear(d_embed, d_embed)
        self.w_value= nn.Linear(d_embed,d_embed)
        self.w_query = nn.Linear(d_embed,d_embed)
        self.output_layer = nn.Linear(d_embed, d_output)


    def dot_product_attention(self, k, v, q):
        unnorm_scores = torch.mm(q,k.t())
        # try also with matmul
        atten_scores = nn.functional.softmax(unnorm_scores,dim=1)
        atten_result = torch.mm(atten_scores, v)

        return atten_result


    def scaled_dot_product_attention(self, k, v, q):
        unnorm_scores = torch.mm(q, k.t())/ math.sqrt(self.d_embed)
        # try also with matmul
        atten_scores = nn.functional.softmax(unnorm_scores, dim=1)
        atten_result = torch.mm(atten_scores, v)

        return atten_result

    def forward(self, input):
        embeds = self.embedding(input)

        keys = self.w_key(embeds)
        values = self.w_value(embeds)
        queries = self.w_query(embeds)

        # atten_rep = self.dot_product_attention(keys,values, queries)
        atten_rep = self.scaled_dot_product_attention(keys, values, queries)

        atten_rep = torch.mean(atten_rep,dim=0,keepdim=True)

        pred = self.output_layer(atten_rep)

        return pred