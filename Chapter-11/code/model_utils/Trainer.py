from data_utils.RankingEvaluator import RankingEvaluator
import torch
from torch import optim
import time
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.BirDRec import BirDRec
from model.SASRec import SASRec
from model.Caser import BERD_Caser
from model.GRU4Rec import BERD_GRU4Rec
from model.FPMC import FPMC
from model.FMLPRec import FMLPRec
from model.MAGNN import MAGNN
from model.BERT4Rec import BERT4Rec
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

class Trainer:
    def __init__(self, config, data_model, save_dir):

        self.config = config
        self.save_dir = save_dir
        self.train_type = config['train_type']
        self.rec_model = config['rec_model']
        self.load_data = self.config['load_data']
        self.model_save_dir = './datasets/' + self.config['dataset'] + '/model/'
        self.model_save_path = self.model_save_dir + self.rec_model + str(self.config['sample_loss_weight']) + '-'
        self.data_analysis_dir = './datasets/' + self.config['dataset'] + '/analysis/'
        self.save_epochs = self.config['save_epochs']

        if self.train_type == 'analysis' and self.load_data:
            self.data_analysis_new()
        else:
            if self.train_type == 'analysis':
                new_dataset, new_dataloader, hc_lowfeq_itempair_instances, hc_highfeq_itempair_instances, instance_idx_2_bin_idx = data_model.generate_train_dataloader_unidirect()
                test_loader = None
            else:
                new_dataset, new_dataloader, hc_lowfeq_itempair_instances, hc_highfeq_itempair_instances, instance_idx_2_bin_idx = data_model.generate_train_dataloader_unidirect()
                test_loader = data_model.generate_test_dataloader_unidirect()

            self.instance_idx_2_bin_idx = instance_idx_2_bin_idx
            self.train_dataset = new_dataset
            self.train_loader = new_dataloader
            self.hc_lowfeq_itempair_instances = hc_lowfeq_itempair_instances
            self.hc_highfeq_itempair_instances = hc_highfeq_itempair_instances

            self._evaluator = RankingEvaluator(test_loader)
            self.train_size = len(self.train_loader.dataset)

            graph = None
            if self.config['rec_model'] == 'MAGNN':
                graph = data_model.getSparseGraph()
            seq_encoder = self.getSeqEncoder(graph)

            rec_model = BirDRec(config, seq_encoder)

            if self.train_type == 'train':
                if rec_model is not None:
                    self._model = rec_model
                    self._device = config['device']
                    self._model.double().to(self._device)
                    self._optimizer = _get_optimizer(
                        self._model.forward_encoder, learning_rate=config['learning_rate'],
                        weight_decay=config['weight_decay'])
                    self.scheduler = ReduceLROnPlateau(self._optimizer, 'max', patience=10,
                                                       factor=config['decay_factor'])

                self.forget_rates = self.build_forget_rates()

            else:
                self._device = config['device']
                self._model = rec_model

    def getSeqEncoder(self, graph):
        if self.config['rec_model'] == 'MAGNN':
            return MAGNN(self.config, graph)
        elif self.config['rec_model'] == 'SASRec':
            return SASRec(self.config, graph)
        elif self.config['rec_model'] == 'GRU4Rec':
            return BERD_GRU4Rec(self.config, graph)
        elif self.config['rec_model'] == 'Caser':
            return BERD_Caser(self.config, graph)
        elif self.config['rec_model'] == 'FPMC':
            return FPMC(self.config, graph)
        elif self.config['rec_model'] == 'FMLPRec':
            return FMLPRec(self.config, graph)
        elif self.config['rec_model'] == 'BERT4Rec':
            return BERT4Rec(self.config, graph)

    def build_forget_rates(self):
        forget_rates = np.ones(self.config['epoch_num']) * 0.06
        forget_rates[:20] = np.linspace(0, 0.06, 20)
        return forget_rates

    def save_model(self, epoch_num):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        corr_str = ''
        if self.config['rectify_input']:
            corr_str += 'rectify_input'
        if self.config['rectify_target']:
            corr_str += 'rectify_target'
        if self.config['new_data']:
            corr_str += 'new_data'
        save_path = self.model_save_path + str(epoch_num) + corr_str + '-model.pkl'
        torch.save(self._model.forward_encoder, save_path)
        print(f'model saved at {save_path}')

    def save_a_set(self, my_set, file_name):
        # Save the set to a text file
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        with open(f'{self.model_save_dir}{file_name}.txt', 'w') as file:
            for item in my_set:
                file.write(str(item) + '\n')

    def save_instance_sets(self, unreliable_inputs, unreliable_targets, epoch):
        if self.config['new_data']:
            self.save_a_set(unreliable_inputs, f'new_unreliable_inputs_epoch{epoch}')
            self.save_a_set(unreliable_targets, f'new_unreliable_targets_epoch{epoch}')
        else:
            self.save_a_set(unreliable_inputs, f'unreliable_inputs_epoch{epoch}')
            self.save_a_set(unreliable_targets, f'unreliable_targets_epoch{epoch}')

    def load_model(self, epoch_num):
        corr_str = ''
        if self.config['rectify_input']:
            corr_str += 'rectify_input'
        if self.config['rectify_target']:
            corr_str += 'rectify_target'
        if self.config['new_data']:
            corr_str += 'new_data'
        load_path = self.model_save_path + str(epoch_num) + corr_str + '-model.pkl'
        print(f'loading model from {load_path}')

        if torch.cuda.is_available():
            model = torch.load(load_path)
        else:
            model = torch.load(load_path, map_location=torch.device('cpu'))

        return model

    def train_one_batch(self, batch, epoch_num, confidence_bins, reliable_bins):
        self._model.forward_encoder.train()
        self._optimizer.zero_grad()

        # [bs], [bs], [1]
        loss, sample_idices, rectified_input_ratio, rectified_target_ratio, update_pool_indices, instances_unreliable_inputs, instances_unreliable_targets, reliable_confidence, unreliable_confidence, reliable_mask = self._model(batch, epoch_num)
        overall_loss = loss.sum()
        overall_loss.backward()

        self._optimizer.step()

        if self._model.moving_avg_model is None:
            pass
        else:
            self._model.moving_avg_model.update_params(self._model.forward_encoder)
            self._model.moving_avg_model.apply_shadow()

        if update_pool_indices is None:
            pass
        else:
            self.train_dataset.update_replacement_candidate_pool(sample_idices, update_pool_indices)

        reliable_mask = torch.squeeze(reliable_mask, 1)
        sample_idices = torch.squeeze(sample_idices, 1)
        confidence_scores = torch.where(reliable_mask > 0, reliable_confidence, unreliable_confidence)

        for instance_idx, confidence_score in zip(sample_idices.tolist(), confidence_scores.tolist()):
            bin_idx = self.instance_idx_2_bin_idx[instance_idx]
            confidence_bins[bin_idx].append(confidence_score)

        for instance_idx, reliable_bool in zip(sample_idices.tolist(), reliable_mask.tolist()):
            bin_idx = self.instance_idx_2_bin_idx[instance_idx]
            if reliable_bool > 0:
                reliable_bins[bin_idx].append(1)
            else:
                reliable_bins[bin_idx].append(0)

        return overall_loss, rectified_input_ratio, rectified_target_ratio, instances_unreliable_inputs, instances_unreliable_targets

    def train(self):
        if self.train_type == 'train':
            print('=' * 60, '\n', 'Start Training', '\n', '=' * 60, sep='')
            keep_train = True
            if self.config['dataset'] == 'ml1m':
                max_freq = 180
                freq_step = 30
            else:
                max_freq = 30
                freq_step = 5
            num_bins = int(max_freq / freq_step) + 1
            for epoch in range(self.config['epoch_num']):
                confidence_bins = [[] for _ in range(num_bins)]
                reliable_bins = [[] for _ in range(num_bins)]
                start_train = time.time()
                loss_iter = 0
                rectified_input_ratios = 0
                rectified_target_ratios = 0
                unreliable_inputs = set()
                unreliable_targets = set()
                for batch in self.train_loader:
                    loss, rectified_input_ratio, rectified_target_ratio, instances_unreliable_inputs, instances_unreliable_targets = self.train_one_batch(batch, epoch, confidence_bins, reliable_bins)
                    unreliable_inputs = unreliable_inputs | set(instances_unreliable_inputs.cpu().numpy().flatten())
                    unreliable_targets = unreliable_targets | set(instances_unreliable_targets.cpu().numpy().flatten())
                    loss_iter += loss.item()
                    rectified_input_ratios += rectified_input_ratio
                    rectified_target_ratios += rectified_target_ratio
                len_train_loader = len(self.train_loader)
                detected_unreliable_instances = unreliable_inputs | unreliable_targets

                print(f'################## epoch {epoch} ###########################')
                print(
                    f"loss: {round(loss_iter / len(self.train_loader), 4)}, len_train_loader:{len_train_loader}, train time: {time.time() - start_train}")
                print(f"rectified_input_ratio: {rectified_input_ratios / len_train_loader}")
                print(f"rectified_target_ratio: {rectified_target_ratios / len_train_loader}")
                start_eval = time.time()
                keep_train = self.evaluate(epoch)
                print(f"eval time: {time.time() - start_eval}")
                print('#################### data analysis ######################')
                # self.analyze_unreliable_detection(detected_unreliable_instances)
                self.print_confidence(confidence_bins)
                self.print_reliable_ratio(reliable_bins)

                if epoch in self.save_epochs:
                    self.save_instance_sets(unreliable_inputs, unreliable_targets, epoch)
                    self.save_model(epoch)
                if not keep_train:
                    break
        elif self.train_type == 'eval':
            for epoch in self.save_epochs:
                self._model.set_encoder(self.load_model(epoch))
                self._model.double().to(self._device)
                self._evaluator.evaluate(model=self._model, train_iter=epoch)
        elif self.train_type == 'analysis':
            if self.load_data:
                pass
            else:
                self.data_analysis()

    def print_confidence(self, confidence_bins):
        print(f'Confidence of Different Groups')
        for i, bin in enumerate(confidence_bins):
            confidence_values = np.array(bin)
            print(f'Group_{i}: mean: {confidence_values.mean()}, min: {confidence_values.min()}, max: {confidence_values.max()}, var: {confidence_values.var()}')

    def print_reliable_ratio(self, reliable_bins):
        print(f'Ratio of Instances Classified as Reliable across Different Groups')
        for i, bin in enumerate(reliable_bins):
            reliable_values = np.array(bin)
            print(f'Group_{i}: reliable ratio: {reliable_values.sum() / reliable_values.size}')


    def data_analysis_new(self):

        prob_gap = np.load(self.data_analysis_dir + 'prob_gap.npy')

        self.draw_fig_new(prob_gap)

    def data_analysis(self):
        load_data = self.config['load_data']
        # collect raw data
        pos_score, neg_scores = None, None
        if load_data:
            pos_score = np.load(self.data_analysis_dir + 'pos_score.npy')
            neg_scores = np.load(self.data_analysis_dir + 'neg_scores.npy')
        else:
            epoch = self.save_epochs[-1]
            self._model.set_encoder(self.load_model(epoch))
            self._model.double().to(self._device)
            for batch in self.train_loader:
                pos_score, neg_scores = self._model.data_analysis(batch)

                if not os.path.exists(self.data_analysis_dir):
                    os.makedirs(self.data_analysis_dir)

                np.save(self.data_analysis_dir + 'pos_score', pos_score)
                np.save(self.data_analysis_dir + 'neg_scores', neg_scores)
                break

        if self.config['dataset'] == 'ml1m':
            bias, max_sum_num, mid_sum_num = 6.9, 10, 400
        if self.config['dataset'] == 'beauty':
            bias, max_sum_num, mid_sum_num = 0, 140, 490
        if self.config['dataset'] == 'yelp':
            bias, max_sum_num, mid_sum_num = 1.55, 110, 490
        if self.config['dataset'] == 'qk-video':
            bias, max_sum_num, mid_sum_num = 6.6, 10, 60
        all_pred_scores = softmax(np.concatenate([(pos_score + bias), neg_scores], axis=1), axis=1)  # ml1m
        # [bs, analysis_neg_num + 1]
        all_pred_scores.sort(axis=1)
        np.save(self.data_analysis_dir + 'all_pred_scores', all_pred_scores)
        self.draw_fig(max_sum_num, mid_sum_num, all_pred_scores)


    def draw_fig_new(self, prob_gap):
        # generate data points
        alpha = 0.05
        step = 0.005
        x = []
        y = []
        while alpha < 0.9:
            alpha += step
            prob_less_than_alpha = len(prob_gap[prob_gap < alpha]) / prob_gap.shape[0]
            if prob_less_than_alpha < 1e-6:
                continue
            x.append(np.log2(alpha))
            y.append(np.log2(prob_less_than_alpha + 1e-8))
        x = np.array(x)
        y = np.array(y)
        res = stats.linregress(x, y)

        estimated_c = round(np.exp2(res.intercept), 4)
        estimated_lambda = round(res.slope, 4)

        font = {
                'weight': 'normal',
                'size': 20,
                }
        sns.set_style("whitegrid")
        blue, = sns.color_palette("muted", 1)

        plt.plot(x, y, 'o', color='lightblue', label=r'Observed (log($\alpha$), log($F_{\alpha}$))', markeredgecolor=blue, markeredgewidth=0.5)
        plt.plot(x, res.intercept + res.slope * x, 'indianred', linewidth=4, alpha=0.8, label=r'Regression Line: $\lambda$log($\alpha$)+log$(C)$')

        x_position = (x.max() - x.min()) * 0.5 + x.min()
        y_position = (y.max() - y.min()) * 0.02 + y.min()
        t = r'Estimated $C$: {:.4f}'.format(estimated_c) + '\n' + r'Estimated $\lambda$: {:.4f}'.format(
            estimated_lambda)

        # plt.figure(figsize=(4, 3), dpi=150)
        plt.text(x=x_position, y=y_position, s=t, bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), fontsize=16)
        # plt.text(x=0.9, y=1.4, s=r'Estimated $\lambda$: {:.4f}'.format(estimated_lambda),
        #          bbox=dict(facecolor='grey', alpha=0.1))
        plt.xlabel(r'log($\alpha$)', fontdict=font)
        plt.ylabel(r'log($F_{\alpha}$)', fontdict=font)

        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.xticks(np.arange(-4, 1, step=1))

        ax = plt.gca()
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)

        plt.grid(linestyle='--', linewidth=1)

        plt.legend(handlelength=0.5, fontsize=13.5)

        plt.savefig(self.data_analysis_dir + self.config['dataset'] + f'-{self.rec_model}.png', dpi=300)
        plt.show()

    def draw_fig(self, max_sum_num, mid_sum_num, all_pred_scores):
        max_value = np.sum(all_pred_scores[:, -max_sum_num:], axis=1, keepdims=False)
        # [bs, 1]
        # beauty 240,
        mid_value = np.sum(all_pred_scores[:, 0:mid_sum_num], axis=1, keepdims=False)
        # [bs, 1]
        prob_gap = max_value - mid_value
        # [bs]
        np.save(self.data_analysis_dir + 'prob_gap', prob_gap)

        # generate data points
        alpha = 0.05
        step = 0.005
        x = []
        y = []
        while alpha < 0.9:
            alpha += step
            prob_less_than_alpha = len(prob_gap[prob_gap < alpha]) / prob_gap.shape[0]
            if prob_less_than_alpha < 1e-6:
                continue
            x.append(np.log2(alpha))
            y.append(np.log2(prob_less_than_alpha + 1e-8))
        x = np.array(x)
        y = np.array(y)
        res = stats.linregress(x, y)

        estimated_c = round(np.exp2(res.intercept), 4)
        estimated_lambda = round(res.slope, 4)

        font = {
                'weight': 'normal',
                'size': 20,
                }
        sns.set_style("whitegrid")
        blue, = sns.color_palette("muted", 1)

        plt.plot(x, y, 'o', color='lightblue', label=r'Observed (log($\alpha$), log($F_{\alpha}$))', markeredgecolor=blue, markeredgewidth=0.5)
        plt.plot(x, res.intercept + res.slope * x, 'indianred', linewidth=4, alpha=0.8, label=r'Regression Line: $\lambda$log($\alpha$)+log$(C)$')

        x_position = (x.max() - x.min()) * 0.5 + x.min()
        y_position = (y.max() - y.min()) * 0.02 + y.min()
        t = r'Estimated $C$: {:.4f}'.format(estimated_c) + '\n' + r'Estimated $\lambda$: {:.4f}'.format(
            estimated_lambda)

        # plt.figure(figsize=(4, 3), dpi=150)
        plt.text(x=x_position, y=y_position, s=t, bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), fontsize=16)
        # plt.text(x=0.9, y=1.4, s=r'Estimated $\lambda$: {:.4f}'.format(estimated_lambda),
        #          bbox=dict(facecolor='grey', alpha=0.1))
        plt.xlabel(r'log($\alpha$)', fontdict=font)
        plt.ylabel(r'log($F_{\alpha}$)', fontdict=font)

        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.xticks(np.arange(-4, 1, step=1))

        ax = plt.gca()
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)

        plt.grid(linestyle='--', linewidth=1)

        plt.legend(handlelength=0.5, fontsize=13.5)

        plt.savefig(self.data_analysis_dir + f'{max_sum_num}={mid_sum_num}-{self.rec_model}.png', dpi=300)
        plt.show()

    def evaluate(self, iter):
        self._model.eval()
        keep_train, ndcg10 = self._evaluator.evaluate(model=self._model, train_iter=iter, eval_model=0)
        self.scheduler.step(ndcg10)
        if self._model.moving_avg_model is None:
            pass
        else:
            print('------------EMA Eval-------------')
            keep_train, ndcg10 = self._evaluator.evaluate(model=self._model, train_iter=iter, eval_model=1)
        return keep_train

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    def analyze_unreliable_detection(self, detected_unreliable_instances):
        hc_lowfeq_itempair_instances = self.hc_lowfeq_itempair_instances
        hc_highfeq_itempair_instances = self.hc_highfeq_itempair_instances
        train_size = self.train_size
        print(f'model unreliable ratio: {len(detected_unreliable_instances) / train_size}')
        print(f'(hc_lowfeq_itempair_instances & unreliable) / (hc_lowfeq_itempair_instances): {len(hc_lowfeq_itempair_instances & detected_unreliable_instances) / len(hc_lowfeq_itempair_instances)}')
        print(f'(hc_lowfeq_itempair_instances & unreliable) / (unreliable): {len(hc_lowfeq_itempair_instances & detected_unreliable_instances) / len(detected_unreliable_instances)}')
        print(f'(hc_highfeq_itempair_instances & unreliable) / (hc_highfeq_itempair_instances): {len(hc_highfeq_itempair_instances & detected_unreliable_instances) / len(hc_highfeq_itempair_instances)}')
        print(f'(hc_highfeq_itempair_instances & unreliable) / (unreliable): {len(hc_highfeq_itempair_instances & detected_unreliable_instances) / len(detected_unreliable_instances)}')




def _get_optimizer(model, learning_rate, weight_decay=0.01):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optim.Adam(optimizer_grouped_parameters, lr=learning_rate)


def set2str(input_set):
    set_str = ''
    set_len = len(input_set)
    for i, item in enumerate(input_set):
        if i < set_len - 1:
            set_str += str(item) + ','
        else:
            set_str += str(item)
    return set_str




