import os
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, device, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        padding = stride
        
        self.device = device

        # patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * (int((configs.seq_len - patch_len) / stride + 2) + 1)

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

        self.lm_model_name = configs.lm_model
        os.environ['TOKENIZERS_PARALLELISM'] = 'True'
        if self.lm_model_name == 'sentence':
            self.lm_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        elif self.lm_model_name == 'minilm6':
            self.lm_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif self.lm_model_name == 'minilm12':
            self.lm_model = SentenceTransformer('all-MiniLM-L12-v2')
        elif self.lm_model_name == 'msmarco':
            self.lm_model = SentenceTransformer('msmarco-MiniLM-L-12-v3')
        elif self.lm_model_name == 'mpnet':
            self.lm_model = SentenceTransformer('all-mpnet-base-v2')
        elif self.lm_model_name == 'distilroberta':
            self.lm_model = SentenceTransformer('all-distilroberta-v1')
        
        elif self.lm_model_name == 'deberta':
            self.lm_model = AutoModel.from_pretrained('microsoft/deberta-base')
        elif self.lm_model_name == 'bert':
            self.lm_model = AutoModel.from_pretrained("google-bert/bert-base-cased")
        elif self.lm_model_name == 'roberta':
            self.lm_model = AutoModel.from_pretrained("roberta-base")
        elif self.lm_model_name == 'distilbert':
            self.lm_model = AutoModel.from_pretrained("distilbert-base-uncased")
            
        if configs.lm_model in ['mpnet', 'distilroberta', 'deberta', 'bert', 'roberta', 'distilbert']:
            self.layer_text = nn.Linear(768, configs.d_model)
        else:
            self.layer_text = nn.Linear(384, configs.d_model)
            
    def forward(self, x_enc_time, x_enc_text):
        # Normalization from Non-stationary Transformer
        means = x_enc_time.mean(1, keepdim=True).detach()
        x_enc_time = x_enc_time - means
        stdev = torch.sqrt(
            torch.var(x_enc_time, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_time /= stdev

        # do patching and embedding
        x_enc_time = x_enc_time.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc_time)
        
        if self.lm_model_name in ['deberta', 'bert', 'roberta', 'distilbert']:
            outputs = self.lm_model(input_ids=x_enc_text['input_ids'].to(self.device), 
                                attention_mask=x_enc_text['attention_mask'].to(self.device),
                                output_hidden_states=True)
        
            emb_text = outputs['hidden_states'][-1]
            emb_text = emb_text.permute(1, 0, 2)[0]
            emb_text = self.layer_text(emb_text)
            emb_text = emb_text.unsqueeze(1).repeat(1, self.enc_in, 1)
            emb_text = self.dropout(emb_text)
            emb_text = emb_text.view(-1, emb_text.shape[-1]).unsqueeze(1)
        else:
            emb_text = torch.from_numpy(self.lm_model.encode(x_enc_text)).to(self.device)
            emb_text = self.layer_text(emb_text)
            emb_text = emb_text.unsqueeze(1).repeat(1, self.enc_in, 1)
            emb_text = self.dropout(emb_text)
            emb_text = emb_text.view(-1, emb_text.shape[-1]).unsqueeze(1)

        enc_out = torch.cat((enc_out, emb_text), 1)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output_time = self.flatten(enc_out)
        output_time = self.dropout(output_time)
        emb = output_time.reshape(output_time.shape[0], -1)
        output_time = self.projection(emb)
        
        return output_time, emb

