import random
random.seed(0)
import time

import multiprocessing as mp
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from collections import Counter
import copy
import matplotlib.pyplot as plt


def Most_Common(lst, topN=2):
    data = Counter(lst)
    list_of_item_with_frequency = data.most_common(topN)
    items = [list_of_item_with_frequency[i][0] for i in range(len(list_of_item_with_frequency))]
    return items

def list2str(lst):
    str_elements = [str(element) for element in lst]
    return ','.join(str_elements)


class SeqDataCollector(object):

    def __init__(self, config):
        print('#' * 10 + ' DataInfo ' + '#' * 10)
        self.config = config
        self.input_len = config['input_len']
        self.device = config['device']
        self.data_path = './datasets/' + config['dataset'] + '/seq/'
        self.train_neg_num = config['train_neg_num']
        random.seed(123)
        np.random.seed(123)
        self.userSet = set()
        self.itemSet = set()
        self.itemIdx2Str = self.load_itemIdx2Str()
        self.userIdx2sequence = {}
        self.item2succeeding_items = {}
        self.userItemSetTrain = {}
        self.item2Users = {}
        self.itemIdx2InstanceIdx_input = {}
        self.itemIdx2InstanceIdx_target = {}
        self.item_freq = {}
        self.valid_users = set()
        self.valid_items = set()
        self.cpu_num = min(mp.cpu_count(), config['thread_num'])
        print(f'cpu num: {self.cpu_num}')

        self.numUser = 0
        self.numItem = 0
        self.item_item_content = None
        self.item_item_interaction = None
        # self.item_dist = [0 for _ in range(self.numItem)]
        # self.user_dist = [0 for _ in range(self.numUser)]

        build_succeeding = self.load_item_succeeding()
        self.load_seq_data(build_succeeding=build_succeeding)

        config['user_num'] = self.numUser
        config['item_num'] = self.numItem
        print(f'numUser:{self.numUser}')
        print(f'numItem:{self.numItem}')
        self.eval_neg_num = config['eval_neg_num']
        self.train_batch_size = config['train_batch_size']
        self.test_batch_size = config['test_batch_size']
        self.train_size = 0
        self.valid_size = 0
        self.test_size = 0


    def load_seq_data(self, build_succeeding=True):
        full_seq_count = self.load_file('seq.dat')
        self.numUser = len(self.userSet)
        self.numItem = len(self.itemSet)

        if build_succeeding:
            for user, item_seq in self.userIdx2sequence.items():
                train_item_seq = item_seq[0:-1]
                for i, item in enumerate(train_item_seq):
                    if item not in self.item2succeeding_items:
                        self.item2succeeding_items[item] = []
                    if i + 1 < len(train_item_seq) - 1:
                        succeeding_item = train_item_seq[i + 1]
                        self.item2succeeding_items[item].append(succeeding_item)
            for item, succeeding_items in self.item2succeeding_items.items():
                commonest_items = Most_Common(succeeding_items, self.config['candidate_size'])
                self.item2succeeding_items[item] = commonest_items
            self.save_item_succeeding(self.item2succeeding_items)

        for user, items in self.userIdx2sequence.items():
            self.userItemSetTrain[user] = set(items[0:-1])
            for item in items[0:-1]:
                if item not in self.item2Users:
                    self.item2Users[item] = set()
                self.item2Users[item].add(user)

    def load_itemIdx2Str(self):
        file_path = self.data_path + '/i_idx2str.dat'
        i_idx2str = {}
        if os.path.exists(file_path):
            print('reading i_idx2str.dat')
            with open(file_path) as fin:
                for line in fin:
                    splited_line = line.strip().split('::')
                    itemIdx, itemStr = int(splited_line[0]), splited_line[1]
                    i_idx2str[itemIdx] = itemStr
        i_idx2str[0] = 'pad'
        return i_idx2str

    def load_item_succeeding(self):
        candidate_size = self.config['candidate_size']
        file_path = self.data_path + f'/item_succeeding{candidate_size}.dat'
        if os.path.exists(file_path):
            print(f'reading {file_path}')
            with open(file_path) as fin:
                for line in fin:
                    splited_line = line.strip().split(':')
                    str_succeeding_items = splited_line[1].strip().split(',')
                    item = int(splited_line[0])
                    succeeding_items = []
                    for str_item in str_succeeding_items:
                        if str_item == '':
                            continue
                        succeeding_items.append(int(str_item))
                    self.item2succeeding_items[item] = succeeding_items
            print(f'{file_path} loaded')
            return False
        else:
            return True

    def save_item_succeeding(self, item2succeedings):
        candidate_size = self.config['candidate_size']
        file_path = self.data_path + f'/item_succeeding{candidate_size}.dat'
        output_lines = []
        for item, succeeding_items in item2succeedings.items():
            output_line = f'{item}:' + list2str(succeeding_items) + '\n'
            output_lines.append(output_line)
        with open(file_path, 'w') as fout:
            fout.writelines(output_lines)
        print(f'{file_path} saved')

    def save_false_P_N(self, outputStr, fileName):
        file_path = self.data_path + f'/{fileName}.dat'
        with open(file_path, 'w') as fout:
            fout.writelines(outputStr)
        print(f'{file_path} saved')

    def slide_window(self, itemList, window_size, candidate_size=2):
        """
        Input a sequence [1, 2, 3, 4, 5] with window size 3
        Return [0, 1, 2], [1, 2, 3],  [2, 3, 4],  [3, 4, 5]
        with   [0, 1, 1], [1, 1, 1],  [1, 1, 1],  [1, 1, 1]
        """
        new_item_list = [0] * (window_size - 2) + itemList
        num_seq = len(itemList) - 1
        for startIdx in range(num_seq):
            endIdx = startIdx + window_size
            item_sub_seq = new_item_list[startIdx:endIdx]

            mask = [1] * window_size
            for i, item in enumerate(item_sub_seq):
                if item == 0:
                    mask[i] = 0

            succeeding_items = []
            candidate_end_idx = min(endIdx + candidate_size, len(new_item_list))
            succeeding_items.extend(new_item_list[endIdx: candidate_end_idx])
            # if len(succeeding_items) < candidate_size:
            #     second_last = item_sub_seq[-2]
            #     succeeding_items.extend(self.item2succeeding_items[second_last])
            if len(succeeding_items) < candidate_size:
                sample_num = candidate_size - len(succeeding_items)
                sampled_items = np.random.randint(low=1, high=self.numItem, size=sample_num)
                succeeding_items.extend(sampled_items)
            if len(succeeding_items) > candidate_size:
                succeeding_items = succeeding_items[0:candidate_size]

            assert len(succeeding_items) == candidate_size

            yield item_sub_seq, mask, succeeding_items[0: candidate_size]

    def generate_train_dataloader_unidirect(self):
        input_len = self.input_len
        print('generating train samples')
        start = time.time()
        train_users = []
        train_hist_items = []
        train_masks = []
        train_targets = []
        train_succeedings = []
        sub_seq_len = input_len + 1
        abandon_count = 0

        instance_idx = 0
        target2instance_ids = [[] for _ in range(self.numItem)]
        for user, item_full_seq in self.userIdx2sequence.items():
            item_train_seq = item_full_seq[0:-1]
            for sub_seq, mask, succeeding_items in self.slide_window(item_train_seq, sub_seq_len):
                input_seq = sub_seq[0: input_len]
                input_mask = mask[0: input_len]
                target = sub_seq[input_len]
                assert len(sub_seq) == len(mask) == sub_seq_len
                # append lists
                train_users.append(user)
                train_hist_items.append(input_seq)
                train_masks.append(input_mask)
                train_targets.append(target)
                train_succeedings.append(succeeding_items)
                #
                target2instance_ids[target].append(instance_idx)
                instance_idx += 1

                self.valid_users.add(user)
                self.valid_items.add(target)
                for item in sub_seq:
                    if item != 0:
                        self.valid_items.add(item)
        self.train_size = len(train_users)
        assert self.train_size == instance_idx
        print(f"train_size: {self.train_size}, time: {(time.time() - start)}")
        print(f'abandoned {abandon_count}({round(abandon_count / self.train_size, 4)}) samples')
        print(f"valid user num: {len(self.valid_users)}")
        print(f"valid item num: {len(self.valid_items)}")

        # new_train_users, new_train_hist_items, new_train_targets, new_train_masks, hc_lowfeq_itempair_instances, hc_highfeq_itempair_instances = self.manipulate_training_instances(train_users, train_hist_items, train_targets, train_masks, train_succeedings)

        # new_train_users = train_users
        # new_train_hist_items = train_hist_items
        # new_train_targets = train_targets
        # new_train_masks = train_masks

        dataset = UnidirectTrainDataset(self.config, train_users, train_hist_items,
                                        train_masks, train_targets, self.userItemSetTrain, target2instance_ids,
                                        self.valid_items, max_item_idx=self.numItem - 1, sample_neg_num=self.train_neg_num,
                                        train_succeedings=train_succeedings)
        dataloader = DataLoader(dataset, shuffle=True, num_workers=self.cpu_num, batch_size=self.train_batch_size)
        instance_idx_2_bin_idx = self.cal_item_cooccurrence(train_hist_items, train_targets)

        # new_dataset = UnidirectTrainDataset(self.config, new_train_users, new_train_hist_items,
        #                                 new_train_masks, new_train_targets, self.userItemSetTrain, target2instance_ids,
        #                                 self.valid_items, max_item_idx=self.numItem - 1, sample_neg_num=self.train_neg_num,
        #                                 train_succeedings=train_succeedings)

        print(f"new_train_size: {len(train_users)}, time: {(time.time() - start)}")

        # new_dataloader = DataLoader(new_dataset, shuffle=True, num_workers=self.cpu_num, batch_size=self.train_batch_size)

        return dataset, dataloader, None, None, instance_idx_2_bin_idx

    def cal_item_cooccurrence(self, train_hist_items, train_targets):
        instance_occur_feq = []
        item_sim_matrix = np.zeros([self.numItem, self.numItem]) + 1e-10
        print('building item_sim_matrix')
        for instance_idx in range(len(train_hist_items)):
            input_items = train_hist_items[instance_idx]
            target_item = train_targets[instance_idx]
            for in_item in input_items:
                if in_item == 0:
                    continue
                item_sim_matrix[in_item, target_item] += 1

        for instance_idx in range(len(train_hist_items)):
            input_items = train_hist_items[instance_idx]
            target_item = train_targets[instance_idx]
            co_occur_feqs = []
            for in_item in input_items:
                if in_item == 0:
                    continue
                co_occur_feqs.append(item_sim_matrix[in_item, target_item])
            co_occur_feqs = np.array(co_occur_feqs)
            instance_occur_feq.append(co_occur_feqs.max())

        assert len(instance_occur_feq) == len(train_hist_items)

        instance_occur_feq = np.array(instance_occur_feq)
        if self.config['dataset'] == 'ml1m':
            max_freq = 180
            freq_step = 30
        else:
            max_freq = 30
            freq_step = 5

        num_bins = int(max_freq / freq_step)
        instance_idx_2_bin_idx = self.allocate_to_bins_by_range(input_list=instance_occur_feq, max_value=max_freq, num_bins=num_bins)

        # Create a histogram
        fig, ax1 = plt.subplots()
        bins = range(0, max_freq, freq_step)
        ax1.hist(instance_occur_feq, bins=bins, color='blue', alpha=0.7, density=True, label='Histogram')

        ax2 = ax1.twinx()
        line_data = np.linspace(min(bins), max(bins), 100)
        line_chart = ax2.plot(line_data, 1 / (np.sqrt(2 * np.pi)) * np.exp(-0.5 * line_data ** 2), color='red',
                              label='Line Chart')

        # plt.hist(instance_occur_feq, bins=bins, edgecolor="k")
        # plt.xticks(bins)
        # plt.xlabel('Item Co-occurrence Frequency')
        # plt.ylabel('Number of Instances')
        # plt.title(self.config['dataset'])
        # plt.savefig(f'{self.data_path}/item_pair_cooccur_dist')

        return instance_idx_2_bin_idx

    def allocate_to_bins_by_range(self, input_list, max_value=180, num_bins=6):
        # Determine the value ranges for each bin
        value_ranges = [(i * (max_value / num_bins), (i + 1) * (max_value / num_bins))
                        for i in range(num_bins)]
        value_ranges.append((max_value, 10000000))

        # Initialize bins as empty lists
        bins = [[] for _ in range(num_bins + 1)]
        instance_idx_2_bin_idx = {}

        # Allocate keys to bins based on value ranges
        for instance_idx, freq in enumerate(input_list):
            for i, (start, end) in enumerate(value_ranges):
                if start <= freq < end:
                    bins[i].append(instance_idx)
                    instance_idx_2_bin_idx[instance_idx] = i
                    break
        assert len(instance_idx_2_bin_idx) == len(input_list)
        return instance_idx_2_bin_idx

    def convert_itemIndices2str(self, items:list):
        outputStr = ''
        for i, itemIdx in enumerate(items):
            itemStr = self.itemIdx2Str[itemIdx]
            outputStr += f'[{itemStr}]'
            if i != len(items) - 1:
                outputStr += ','
        return outputStr

    def manipulate_training_instances(self, train_users, train_hist_items, train_targets, train_masks, train_succeedings):
        new_train_users = copy.deepcopy(train_users)
        new_train_hist_items = copy.deepcopy(train_hist_items)
        new_train_targets = copy.deepcopy(train_targets)
        new_train_masks = copy.deepcopy(train_masks)
        input_len = len(new_train_hist_items[0])

        for item in range(self.numItem):
            if item not in self.itemIdx2InstanceIdx_input:
                self.itemIdx2InstanceIdx_input[item] = set()
            if item not in self.itemIdx2InstanceIdx_target:
                self.itemIdx2InstanceIdx_target[item] = set()
        print('building itemIdx2Instance maps')
        for instance_idx in range(len(train_hist_items)):
            input_items = train_hist_items[instance_idx]
            target_item = train_targets[instance_idx]
            self.itemIdx2InstanceIdx_target[target_item].add(instance_idx)
            for in_item in input_items:
                if in_item == 0:
                    continue
                self.itemIdx2InstanceIdx_input[in_item].add(instance_idx)
        item_sim_matrix = np.zeros([self.numItem, self.numItem]) + 1e-10
        print('building item_sim_matrix')
        for instance_idx in range(len(train_hist_items)):
            input_items = train_hist_items[instance_idx]
            target_item = train_targets[instance_idx]
            for in_item in input_items:
                if in_item == 0:
                    continue
                item_sim_matrix[in_item, target_item] += 1

        for i in range(self.numItem):
            item_sim_matrix[i, i] = 1e-10

        data = item_sim_matrix
        # Flatten the original array and get the indices that would sort it
        flat_data = data.flatten()

        sorted_indices = np.argsort(flat_data)
        print('obtained sorted matrix indices')

        # Get the indices of the top 300 values
        top_indices = sorted_indices[-300:]
        bottom_indices = sorted_indices[0: 1000000]

        # Reshape the top_50_indices to 2D indices in the original array
        top_indices_2d = np.unravel_index(top_indices, data.shape)
        bottom_indices_2d = np.unravel_index(bottom_indices, data.shape)

        # Get the top 50 values from the original array
        top_values = data[top_indices_2d]
        bottom_values = data[bottom_indices_2d]

        # Create a list of tuples containing (value, indices) pairs
        top_with_indices = [(value, (row, col)) for value, row, col in
                               zip(top_values.flatten(), *top_indices_2d)]
        bottom_with_indices = [(value, (row, col)) for value, row, col in
                               zip(bottom_values.flatten(), *bottom_indices_2d)]

        # Sort the list of tuples by value in descending order
        top_with_indices.sort(reverse=True, key=lambda x: x[0])
        bottom_with_indices.sort(reverse=False, key=lambda x: x[0])

        # Print the top 50 values with their indices in the original array
        print('Top item-item pairs')
        instances_to_modify_list = []  # instance_index, input, target
        instances_false_negative_print = []
        hc_lowfeq_itempair_instances = set()
        for matrix_idx, (value, (row_item, col_item)) in enumerate(top_with_indices, start=1):
            instances_to_modify_candidates_input = self.itemIdx2InstanceIdx_input[row_item]
            instances_to_modify_candidates_target = self.itemIdx2InstanceIdx_target[col_item]
            instances_to_modify = instances_to_modify_candidates_input & instances_to_modify_candidates_target
            for i, instance_idx in enumerate(instances_to_modify):
                if i == 0:
                    print(
                        f"Rank {matrix_idx}: Value={value}, Index=({row_item}, {col_item}) Str({self.itemIdx2Str[row_item]}, {self.itemIdx2Str[col_item]})")
                    hc_lowfeq_itempair_instances.add(instance_idx)
                    user = train_users[instance_idx]
                    user_seq_len = len(self.userIdx2sequence[user])
                    if user_seq_len > 10:
                        historical_items = self.userIdx2sequence[user][0:10]
                    else:
                        historical_items = self.userIdx2sequence[user]

                    input_items = train_hist_items[instance_idx]
                    target_item = train_targets[instance_idx]

                    historical_items_str = self.convert_itemIndices2str(historical_items)
                    input_items_str = self.convert_itemIndices2str(input_items)
                    target_item_str = self.convert_itemIndices2str([target_item])

                    instances_false_negative_print.append(f'{instance_idx}:::{historical_items_str}:::{input_items_str}:::{target_item_str}\n')
                    continue

                if i != 0 and self.config['new_data']:
                    user = train_users[instance_idx]
                    instances_to_modify_list.append((instance_idx, row_item, col_item))
                    replace_candidates = self.userIdx2sequence[user]
                    new_input = random.choice(replace_candidates)
                    while new_input == row_item:
                        new_input = random.choice(replace_candidates)
                    new_target = random.choice(replace_candidates)
                    while new_target == col_item:
                        new_target = random.choice(replace_candidates)
                    new_train_hist_items[instance_idx] = [new_input] * input_len
                    new_train_targets[instance_idx] = new_target

        print(f'{len(instances_to_modify_list)} instances are modified')
        self.save_false_P_N(outputStr=instances_false_negative_print, fileName='false_negatives')


        print('Bottom item-item pairs')
        instances_false_positive_print = []
        instances_to_add_list = []
        current_instance_idx = len(train_hist_items)
        used_rows = set()
        hc_highfeq_itempair_instances_high = set()
        added_pair_num = 0
        if self.config['new_data']:
            threshold = 1000
        else:
            threshold = 1
        for matrix_idx, (value, (row_item, col_item)) in enumerate(bottom_with_indices, start=1):
            if row_item in used_rows:
                continue
            if row_item == col_item:
                continue
            if row_item == 0 or col_item == 0:
                continue
            else:
                used_rows.add(row_item)
                print(
                    f"Rank {matrix_idx}: Value={value}, Index=({row_item}, {col_item}) Str({self.itemIdx2Str[row_item]}, {self.itemIdx2Str[col_item]})")
                added_pair_num += 1
                repeat = 0
                while repeat < threshold:
                    user = random.choice(list(self.item2Users[row_item]))
                    instances_to_add_list.append((current_instance_idx, user, row_item, col_item))
                    current_instance_idx += 1
                    repeat += 1
                    new_input_items = random.choices(self.userIdx2sequence[user], k=input_len - 1) + [row_item]
                    user_seq_len = len(self.userIdx2sequence[user])
                    if user_seq_len > 10:
                        historical_items = self.userIdx2sequence[user][0:10]
                    else:
                        historical_items = self.userIdx2sequence[user]
                    historical_items_str = self.convert_itemIndices2str(historical_items)
                    new_train_hist_items.append(new_input_items)
                    new_train_users.append(user)
                    new_train_targets.append(col_item)
                    new_train_masks.append([1] * input_len)
                    train_succeedings.append(np.random.randint(low=1, high=self.numItem - 1, size=2))
                    added_instance_idx = len(new_train_users) - 1
                    hc_highfeq_itempair_instances_high.add(added_instance_idx)

                    input_items_str = self.convert_itemIndices2str(new_input_items)
                    target_item_str = self.convert_itemIndices2str([col_item])
                    instances_false_positive_print.append(f'{added_instance_idx}:::{historical_items_str}:::{input_items_str}:::{target_item_str}\n')

                if added_pair_num == 100:
                    break

        print(f'{len(instances_to_add_list)} instances are added')

        hc_highfeq_itempair_instances = set(range(len(train_users), len(new_train_users)))
        self.save_false_P_N(instances_false_positive_print, 'false_positives')

        return new_train_users, new_train_hist_items, new_train_targets, new_train_masks, hc_lowfeq_itempair_instances, hc_highfeq_itempair_instances


    def convert_file_to_int_set(self, file):
        int_set = set()
        with open(file) as fin:
            for line in fin:
                int_set.add(int(float(line.strip())))
        return int_set

    def generate_valid_dataloader_unidirect(self):
        input_len = self.input_len
        print('generating valid samples')
        start = time.time()
        valid_users = []
        valid_hist_items = []
        valid_masks = []
        valid_targets = []
        abandon_count = 0
        for user, item_full_seq in self.userIdx2sequence.items():
            target_item = item_full_seq[-2]
            if user not in self.valid_users or target_item not in self.valid_items:
                abandon_count += 1
                continue
            valid_users.append(user)
            raw_input = item_full_seq[0:-2]
            raw_input_len = len(raw_input)
            if raw_input_len >= input_len:
                input = raw_input[-input_len:]
                mask = [1] * input_len
            else:  # raw_input_len < input_len
                input = [0] * (input_len - raw_input_len) + raw_input
                mask = [0] * (input_len - raw_input_len) + [1] * raw_input_len
            assert len(input) == len(mask) == input_len
            assert item_full_seq[-3] == input[-1]

            valid_hist_items.append(input)
            valid_masks.append(mask)
            valid_targets.append(target_item)

        dataset = UnidirectTrainDataset(valid_users, valid_hist_items,
                                        valid_masks, valid_targets, self.userItemSetTrain,
                                        max_item_idx=self.numItem - 1, sample_neg_num=self.train_neg_num)
        dataloader = DataLoader(dataset, shuffle=True,
                                num_workers=self.cpu_num, batch_size=self.train_batch_size)
        self.valid_size = len(valid_users)
        print(f"valid_size: {self.valid_size}, time: {(time.time() - start)}")
        print(f'abandoned {abandon_count}({round(abandon_count / self.valid_size, 4)}) samples')
        return dataloader

    def generate_test_dataloader_unidirect(self):
        input_len = self.input_len
        print('generating test samples')
        start = time.time()
        test_users = []
        test_hist_items = []
        test_masks = []
        test_targets = []
        pad_indices = []
        abandon_count = 0
        for user, item_full_seq in self.userIdx2sequence.items():
            if len(item_full_seq) < 2:
                continue
            test_users.append(user)
            raw_input = item_full_seq[0:-1]
            raw_input_len = len(raw_input)
            if raw_input_len >= input_len:
                input = raw_input[-input_len:]
                mask = [1] * input_len
            else: # raw_input_len < input_len
                input = [0] * (input_len - raw_input_len) + raw_input
                mask = [0] * (input_len - raw_input_len) + [1] * raw_input_len
            assert len(input) == len(mask) == input_len
            assert item_full_seq[-2] == input[-1]

            test_hist_items.append(input)
            test_masks.append(mask)

            sampled_negs = []

            if self.eval_neg_num == 'full':
                valid_neg_ids = self.valid_items - self.userItemSetTrain[user]
                pad_length = self.numItem - len(valid_neg_ids)
                valid_length = len(valid_neg_ids) + 1
                padded_valid_neg_ids = list(valid_neg_ids) + [0] * pad_length
                pad_indices.append([0] * valid_length + [-10^8] * pad_length)
                sampled_negs = padded_valid_neg_ids
            # else:
            #     num_item_to_rank = self.eval_neg_num + 1  # negative ones plus the positive one
            #     while len(sampled_negs) < num_item_to_rank:
            #         sampled_neg_cands = np.random.choice(self.numItem, self.eval_neg_num, False)
            #         valid_neg_ids = [x for x in sampled_neg_cands if x not in self.userItemSetTrain[user]]
            #         sampled_negs.extend(valid_neg_ids[:])
            #     sampled_negs = sampled_negs[:self.eval_neg_num]
            #     pad_indices.append([0] * num_item_to_rank)

            test_targets.append([item_full_seq[-1]] + list(sampled_negs))

        dataset = UnidirectTestDataset(test_users, test_hist_items, test_masks, test_targets, pad_indices)
        dataloader = DataLoader(dataset, shuffle=False,
                                num_workers=self.cpu_num, batch_size=self.test_batch_size)
        self.test_size = len(test_users)
        print(f"test_size: {self.test_size}, time: {(time.time() - start)}")
        print(f'abandoned {abandon_count}({round(abandon_count / self.test_size, 4)}) samples')
        return dataloader, test_hist_items, test_targets

    def load_file(self, file_name):
        file_path = self.data_path + '/' + file_name
        line_count = 0
        if os.path.exists(file_path):
            print(f'reading {file_name}')
            with open(file_path) as fin:
                for line in fin:
                    splited_line = line.strip().split('::')
                    userIdx, itemIdx = int(splited_line[0]), int(splited_line[1])
                    self.userSet.add(userIdx)
                    self.itemSet.add(itemIdx)
                    if userIdx not in self.userIdx2sequence:
                        self.userIdx2sequence[userIdx] = []
                    self.userIdx2sequence[userIdx].append(itemIdx)
                    line_count += 1
        self.itemSet.add(0)  # 0 is the item for padding
        return line_count

    def save_file(self, file_name):
        file_path = self.data_path + '/' + file_name
        line_count = 0
        if os.path.exists(file_path):
            print(f'reading {file_name}')
            with open(file_path) as fin:
                for line in fin:
                    splited_line = line.strip().split(' ')
                    user, item = splited_line[0], splited_line[1]
                    if user not in self.user2Idx:
                        userIdx = len(self.user2Idx)
                        self.user2Idx[user] = userIdx
                    if item not in self.item2Idx:
                        itemIdx = len(self.item2Idx)
                        self.item2Idx[item] = itemIdx
                    userIdx = self.user2Idx[user]
                    itemIdx = self.item2Idx[item]
                    if userIdx not in self.userIdx2sequence:
                        self.userIdx2sequence[userIdx] = []
                    self.userIdx2sequence[userIdx].append(itemIdx)
                    line_count += 1
        return line_count


class UnidirectTrainDataset(torch.utils.data.Dataset):

    def __init__(self, config, train_users, train_hist_items,
                 train_masks, train_targets, userItemSet, target2instance_ids, valid_items, max_item_idx, sample_neg_num, train_succeedings, aug_num=3):
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        train_masks = [bs, seq_len]
        train_targets = [bs]
        clean_mask = [bs], denoting whether this instance is clean or not
        """
        assert len(train_users) == len(train_hist_items) == len(train_masks) == len(train_targets)
        self.config = config
        self.train_users = train_users
        self.train_hist_items = train_hist_items
        self.train_targets = train_targets
        # self.clean_mask = clean_mask
        self.train_size = len(train_users)
        self.userItemSet = userItemSet
        self.max_item_idx = max_item_idx
        self.sample_neg_num = sample_neg_num + 1  # additional one for negative input sample
        self.target_neg_num = sample_neg_num
        self.aug_num = aug_num
        self.input_len = len(self.train_hist_items[0])
        # self.item_dist = item_dist
        # self.numItem = len(item_dist)
        self.target2instance_ids = target2instance_ids
        self.train_succeedings = train_succeedings
        self.replacement_candidate_pool = [[] for i in range(len(self.train_succeedings))]
        self.valid_items = valid_items
        self.init_replacement_candidate_pool()

    def init_replacement_candidate_pool(self):
        print('begin to initialize the replacement candidate pool')
        start = time.time()
        pool_size = self.config['train_neg_num'] / 2
        for i, succeeding_items in enumerate(self.train_succeedings):
            sample_size = int(pool_size - len(succeeding_items))
            sampled_items = np.random.randint(low=1, high=self.max_item_idx, size=sample_size)
            self.replacement_candidate_pool[i] = np.append(succeeding_items, sampled_items)
        print(f'replacement candidate pool initialized successfully!, time: {time.time()-start}')

    def update_replacement_candidate_pool(self, indices, updated_candidates):
        pool_size = self.config['train_neg_num']
        for i in range(len(indices)):
            instance_idx = indices[i]
            new_candidates = updated_candidates[i, :]
            # sample_size = pool_size - len(old_candidates)
            # new_candidates = np.random.randint(low=1, high=self.max_item_idx, size=sample_size)
            self.replacement_candidate_pool[instance_idx] = new_candidates

    def __getitem__(self, index):
        userIdx = self.train_users[index]
        # interacted_items = self.userItemSet[userIdx]
        # final_negs = np.random.randint(low=1, high=self.numItem, size=self.sample_neg_num)

        # final_negs = []
        # while len(final_negs) < self.sample_neg_num:
        #     # [bs, num_neg]
        #     # sampled_negs = np.random.choice(self.numItem, self.neg_num, False, self.item_dist)
        #     sampled_negs = np.random.randint(low=1, high=self.numItem, size=self.sample_neg_num)
        #     valid_negs = [x for x in sampled_negs if x not in interacted_items]
        #     final_negs.extend(valid_negs[:])

        # final_negs_targets = final_negs[:self.target_neg_num]
        # target_for_neg_input = final_negs[-1]

        target = self.train_targets[index]
        # random_pos_instance_idx = random.choice(self.target2instance_ids[target])
        # random_neg_instance_idx = random.choice(self.target2instance_ids[target_for_neg_input])

        return userIdx, \
               torch.tensor(self.train_hist_items[index]), \
               target, \
               torch.tensor(self.replacement_candidate_pool[index]), \
               index

    def __len__(self):
        return self.train_size


class UnidirectTestDataset(torch.utils.data.Dataset):

    def __init__(self, test_users, test_hist_items, test_masks, test_targets, pad_indices=None):
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        masks = [bs, seq_len]
        target_ids = [bs, pred_num]
        """
        self.test_users = test_users
        self.test_hist_items = test_hist_items
        self.test_targets = test_targets
        self.test_size = len(test_users)
        self.pad_indices = pad_indices

    def __getitem__(self, index):
        return self.test_users[index], \
               torch.tensor(self.test_hist_items[index]), \
               torch.tensor(self.test_targets[index]), \
               torch.tensor(self.pad_indices[index])

    def __len__(self):
        return self.test_size
