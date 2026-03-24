from model.SeqContextEncoder import SeqContextEncoder
from model_utils.BERT_SeqRec import BertModel
from model_utils.BERT_SeqRec import BertConfig
import torch


class MAGNN(SeqContextEncoder):

    def __init__(self, config, graph):
        super().__init__(config, graph)

    def network_param_init(self, config):
        bert_config = BertConfig(config['item_num'], config)
        self.seq_module = BertModel(bert_config, use_outer_embed=True)

    def seq_modelling(self, hist_item_ids, user_embed=None, rectify=True):
        # [bs, seq_len, hidden_size]
        bert_context = self.seq_module(hist_item_ids, outer_embed=self.obtain_item_embeds(), unidirectional=False)
        # [bs, hidden_size]
        if rectify:
            return bert_context[:, -1, :].squeeze(1), bert_context[:, -2, :].squeeze(1)
        else:
            return bert_context[:, -1, :].squeeze(1)

    def obtain_item_embeds(self):
        items_emb = self.item_embeddings.weight
        all_emb = items_emb
        # [numItem, hidden_size]
        embs = [all_emb]
        g_droped = self.graph
        for layer in range(self.config['n_layers']):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        # [numItem, hidden_size, n_layers + 1]
        embs = torch.stack(embs, dim=2)
        # [numItem, hidden_size]
        light_out = torch.mean(embs, dim=2)
        return light_out