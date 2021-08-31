import math
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, embed_dim, hid_dim,vocab_len):
        super(EncoderDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.vocab_len = vocab_len
        self.char_embeddings = nn.Embedding(vocab_len,embed_dim)
        self.position_embeddings = nn.Embedding(20, embed_dim)

        self.encoder_w_key = nn.Linear(embed_dim, hid_dim)
        self.encoder_w_value = nn.Linear(embed_dim, hid_dim)
        self.encoder_w_query = nn.Linear(embed_dim, hid_dim)

        self.decoder_w_key = nn.Linear(embed_dim, hid_dim)
        self.decoder_w_value = nn.Linear(embed_dim, hid_dim)
        self.decoder_w_query = nn.Linear(embed_dim, hid_dim)

        self.cross_attention_w_key = nn.Linear(embed_dim, hid_dim)
        self.cross_attention_w_value = nn.Linear(embed_dim, hid_dim)
        self.cross_attention_w_query = nn.Linear(embed_dim, hid_dim)

        self.out = nn.Linear(hid_dim, vocab_len)

    def unmasked_scaled_dot_product(self, key, value, query):
        unnorm_scores = torch.mm(query, key.t())/math.sqrt(self.hid_dim)
        norm_scores = nn.functional.softmax(unnorm_scores, dim=1) # N * N
        atten_result = torch.mm(norm_scores, value)  # N * hid_dim

        return atten_result


    def masked_scaled_dot_product(self, d_key, d_value, d_query):
        max_len, _ = d_query.size()
        mask = torch.tril(torch.ones(max_len, max_len))
        scores = torch.mm(d_query, d_key.t())
        scores = scores.masked_fill(mask == 0, -1e9)
        norm_scores = nn.functional.softmax(scores, dim=1)
        atten_results = torch.mm(norm_scores, d_value)

        return atten_results

    def forward(self, encoder_inp, decoder_inp):
        encoder_embed = self.char_embeddings(encoder_inp)
        decoder_embed = self.char_embeddings(decoder_inp)

        e_positions = torch.arange(0, encoder_inp.size()[0])
        e_pos_embs = self.position_embeddings(e_positions)

        d_positions = torch.arange(0, decoder_inp.size()[0])
        d_pos_embs = self.position_embeddings(d_positions)

        encoder_embed = encoder_embed + e_pos_embs
        e_query = self.encoder_w_query(encoder_embed)
        e_key = self.encoder_w_key(encoder_embed)
        e_value = self.encoder_w_value(encoder_embed)

        decoder_embed = decoder_embed + d_pos_embs
        d_query = self.decoder_w_query(decoder_embed)
        d_key = self.decoder_w_key(decoder_embed)
        d_value = self.decoder_w_value(decoder_embed)

        encoder_states = self.unmasked_scaled_dot_product(e_key, e_value, e_query)
        decoder_states = self.masked_scaled_dot_product(d_key, d_value, d_query)

        cross_query = self.cross_attention_w_query(decoder_states)
        cross_key = self.cross_attention_w_key(encoder_states)
        cross_value = self.cross_attention_w_value(encoder_states)

        cross_attention_states = self.unmasked_scaled_dot_product(cross_key,cross_value, cross_query)

        final_rep = cross_attention_states + decoder_states

        # final_rep = final_rep.transpose(0, 1).contiguous().view(-1, self.d_hid)
        decoder_preds = self.out(final_rep)
        decoder_preds = torch.nn.functional.log_softmax(decoder_preds, dim=1)


        return decoder_preds





