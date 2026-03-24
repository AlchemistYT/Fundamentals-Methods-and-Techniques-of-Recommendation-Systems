import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import torch

from model_utils.Trainer import Trainer
from data_utils.GraphDataGenerator import GraphDataCollector
from data_utils.RankingEvaluator import print_dict
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    config = {
        # data settings
        'thread_num': 4,
        'dataset': 'yelp',  # electronics
        'eval_neg_num': 'full',
        'train_neg_num': 250,
        'target_neg_num': 8,
        'input_len': 5,
        'threshold': 1,
        # training settings
        'rec_model': 'GRU4Rec',
        'train_type': 'train',  # train / eval
        'co-train': False,
        'filter-with-loss': False,
        'save_epochs': [90],
        'epoch_num': 100,
        'learning_rate': 0.001,   # 0.01 for yelp and beauty
        'train_batch_size': 8000,
        'test_batch_size': 512,
        'drop_ratio': 0.1,
        # graph settings
        'n_layers': 2,
        'next_hop_num': 1,
        # prob settings
        'entropy_threshold': 1.0,
        'sample_num': 4,
        'sample_loss_weight': 0.001,
        # BERT settings
        'decay_factor': 0.9,
        'hidden_size': 64,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'intermediate_size': 128,
        'hidden_act': "gelu",
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'type_vocab_size': 1,
        'initializer_range': 0.1,
        'weight_decay': 0.01,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'max_seq_len': 5,
        'add_user': True,
        'contrast': False,
        'contrast_lmd': 0.1,
        'corr_epoch': 0,
        'corr_threshold': 1,
        'delete_target_ratio': 0.25,  # beta; yelp 0.3,
        'delete_input_ratio': 0.99,  # beta'; yelp 0.2,
        'replace_ratio': 0.6,        # yelp 0.7,
        'rectify_input': True,
        'rectify_target': True,
        'replace_target': True,
        'self_ensemble': True,
        'temperature': 0.01,
        'weighted_loss': False,
        # caser setting
        'n_h': 16,
        'n_v': 4,
        # GRU4Rec config
        'gru_layer_num': 2,
        'candidate_size': 2,
        'rho': 0.99,
        'load_data': True,
        'new_data': False,
    }

random.seed(123)


def main():
    # './datasets/electronics/seq/', './datasets/sports/seq/', './datasets/ml2k/seq/'
    # 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
    # 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    # for rec_model in ['BERT4Rec', 'MAGNN', 'FMLPRec', 'SASRec', 'GRU4Rec', 'Caser', 'FPMC']:
    for dataset in ['ml1m', 'game', 'cd', 'kindle']:  # 'ml1m', 'game', 'cd', 'kindle' 'cd', 'electronics', 'beauty', 'ml100k', 'yelp', 'sports', 'qk-article', 'qk-video'
        for rec_model in ['SASRec']:
            for delete_input_ratio in [0.1]:  # 0.1, 0.2, 0.3, 0.4, 0.5
                for delete_target_ratio in [0.2]:
                    for replace_ratio_add in [0.1]:  # 0.1, 0.2, 0.3, 0.4
                        for rho in [0.1]:
                            config['delete_input_ratio'] = delete_input_ratio
                            config['delete_target_ratio'] = delete_target_ratio
                            config['replace_ratio'] = delete_target_ratio + replace_ratio_add
                            config['dataset'] = dataset
                            config['rec_model'] = rec_model
                            config['rho'] = rho
                            data_model = GraphDataCollector(config)
                            print_dict(config, 'config')
                            trainer = Trainer(config, data_model, save_dir='./datasets/' + config['dataset'] + '/seq/')
                            trainer.train()
                            data_model = None
                            trainer = None
                            # for corr_epoch in [0, 1, 3, 5, 7, 10]:
                            #     for corr_ratio_upper in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
                            # config['corr_epoch'] = corr_epoch
                            # config['corr_ratio_upper'] = corr_ratio_upper

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()



