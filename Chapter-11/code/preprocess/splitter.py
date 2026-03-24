import os
import random
from operator import itemgetter

# input: ratings.dat
# output: full.dat, train.dat, valid.dat, test.dat, i_idx2id.dat, u_idx2id.dat, log.dat
# output: full.dat: [userIdx, itemIdx, rating, time]

def pre_process(seprate, dir_path, user_len_thre, item_len_thre):

    wholeDataSet = []
    valid_items = set()
    userId_set = set()
    itemId_set = set()
    with open(dir_path + '/ratings.dat') as dataFile:
        for line in dataFile:
            record = line.strip().split(seprate)
            userId, itemId, rating, time = record
            userId_set.add(userId)
            itemId_set.add(itemId)
            wholeDataSet.append([userId, itemId, rating, float(time)])
    print("raw userId num:" + str(len(userId_set)))
    print("raw itemId num:" + str(len(itemId_set)))
    print('numRating: ' + str(len(wholeDataSet)))
    split_UserTimeLOO(wholeDataSet, dir_path, user_len_thre, item_len_thre)
    split_triples(dir_path)

def read_maping(idx2Id_file):
    id2Idx_dict = {}
    with open(idx2Id_file) as fin:
        for line in fin:
            entry_idx, entry_id = line.strip().split('\t')
            id2Idx_dict[entry_id] = entry_idx
    return id2Idx_dict

def split_UserTimeLOO(wholeDataSet, dir_path, user_len_thre, item_len_thre):

    # random.seed(123)
    # collect all user-items mappings
    user_items = {}
    item_users = {}
    # collect all user item - rating review timeStamp mappings
    userItemToRatingTimeStamp = {}
    for record in wholeDataSet:
        userId, itemId, rating, timeStamp = record
        timeStamp = float(timeStamp)
        user_items.setdefault(userId, [])
        item_users.setdefault(itemId, [])
        user_items[userId].append([itemId, timeStamp])
        item_users[itemId].append([userId, timeStamp])
        userItemToRatingTimeStamp[userId, itemId] = [rating, timeStamp]

    # split the data
    wholeSet_toWrite = []
    userId2Idx = {}
    itemId2Idx = {}
    userCount = 0
    itemCount = 1  # item idx starts from 1, 0 is reserved for padding

    for userId in user_items:
        items = sorted(user_items[userId], key=itemgetter(1))
        new_items = []
        for item in items:
            if item not in new_items:
                new_items.append(item)
        # get and sort items
        if len(new_items) >= user_len_thre:
            # split first n items into train set, except the last 2
            # items for trainSet
            for itemId, timeStamp in items:
                if len(item_users[itemId]) >= item_len_thre:
                    if userId not in userId2Idx:
                        userId2Idx[userId] = userCount
                        userCount += 1
                    userIdx = userId2Idx[userId]
                    if itemId not in itemId2Idx:
                        itemId2Idx[itemId] = itemCount
                        itemCount += 1
                    itemIdx = itemId2Idx[itemId]
                    rating, timeStamp = userItemToRatingTimeStamp[userId, itemId]
                    wholeSet_toWrite.append([userIdx, itemIdx, rating, timeStamp])

    write_data(wholeSet_toWrite, userId2Idx, itemId2Idx, dir_path)

    print("processed userId num:" + str(len(userId2Idx)))
    print("processed itemId num:" + str(len(itemId2Idx)))
    print('processed numRating: ' + str(len(wholeSet_toWrite)))

def build_item_id_to_str(data_dir):
    item_id2str = {}
    data_path = data_dir + '/itemId2Str.dat'
    if os.path.exists(data_path):
        with open(data_path) as fin:
            lineCount = 0
            for line in fin:
                records = line.strip().split('::')
                if len(records) > 1:
                    itemId, item_str = records[0], records[1]
                    lineCount += 1
                    item_id2str[itemId] = item_str
    return item_id2str

def write_data(wholeSet, userId2Idx, itemId2Idx, dir_path):
    userIdx2Id_write = []
    itemIdx2Id_write = []
    full_to_write = []
    itemIdx2str_write = []
    itemId2str = build_item_id_to_str(dir_path)

    for userId in userId2Idx:
        userIdx = userId2Idx[userId]
        userIdx2Id_write.append(str(userIdx) + '\t' + str(userId) + '\n')

    for itemId in itemId2Idx:
        itemIdx = itemId2Idx[itemId]
        if itemId in itemId2str:
            item_str = itemId2str[itemId]
            itemIdx2str_write.append(str(itemIdx) + '::' + str(item_str) + '\n')
        itemIdx2Id_write.append(str(itemIdx) + '::' + str(itemId) + '\n')

    for record in wholeSet:
        userIdx, itemIdx, rating, time = record[0], record[1], record[2], record[3]
        full_to_write.append(f'{userIdx}::{itemIdx}::{rating}::{time}\n')

    fullOutputPath = dir_path + '/seq'
    if not os.path.exists(fullOutputPath):
        os.makedirs(fullOutputPath)

    with open(dir_path + '/seq/seq.dat', 'w') as fullFile:
        fullFile.writelines(full_to_write)

    with open(dir_path + '/u_idx2id.dat', 'w') as userMapFile:
        userMapFile.writelines(userIdx2Id_write)

    with open(dir_path + '/i_idx2id.dat', 'w') as itemMapFile:
        itemMapFile.writelines(itemIdx2Id_write)

    with open(dir_path + '/seq/i_idx2str.dat', 'w') as itemMapFile:
        itemMapFile.writelines(itemIdx2str_write)

    with open(dir_path + '/rec_log.dat', 'w') as logfile:
        logfile.writelines(f'numUser:{len(userId2Idx)}\n'
                           f'numItem:{len(itemId2Idx)}\n'
                           f'trainSize:{len(wholeSet)}\n')


def print_map(map_to_print, print_path):

    print_lines = []
    for itemId, itemIdx in sorted(map_to_print.items(), key=lambda item: item[1]):
        print_lines.append(f'{itemIdx}\t{itemId}\n')

    with open(print_path, 'w', encoding='utf-8') as fout:
        fout.writelines(print_lines)


def print_triples(triples, ent_map, rel_map, print_path):
    print_lines = []
    for triple in triples:
        head, rel, tail = triple
        headIdx = ent_map[head]
        tailIdx = ent_map[tail]
        relIdx = rel_map[rel]
        print_lines.append(f'{headIdx} {tailIdx} {relIdx}\n')

    with open(print_path, 'w', encoding='utf-8') as fout:
        fout.writelines(print_lines)


def split_triples(data_dir):
    """
        input: i_idx2id.dat, triplets.dat: [head_str, rel_str, tail_str]
        output: train.dat, valid.dat, test.dat [head_idx, rel_idx, tail_idx]
    """
    ent_id2idx = {}
    rel_id2idx = {}
    head_set = set()
    with open(data_dir + '/i_idx2id.dat', encoding='utf-8') as fin:
        for line in fin:
            records = line.strip().split('::')
            item_idx, item_id = int(records[0]), records[1]
            ent_id2idx[item_id] = item_idx

    triples = []
    ent_count = len(ent_id2idx) + 1
    rel_count = 0
    with open(data_dir + '/triples.dat', encoding='utf-8') as fin:
        for line in fin:
            if len(line.strip().split('\t')) != 3:
                # print(line)
                continue
            ent_head, rel, ent_tail = line.strip().split('\t')
            head_set.add(ent_head)
            triples.append([ent_head, rel, ent_tail])
            if ent_head not in ent_id2idx:
                ent_id2idx[ent_head] = ent_count
                ent_count += 1
            if ent_tail not in ent_id2idx:
                ent_id2idx[ent_tail] = ent_count
                ent_count += 1
            if rel not in rel_id2idx:
                rel_id2idx[rel] = rel_count
                rel_count += 1

    print('entity_num: ' + str(len(ent_id2idx)))
    print('entity_except_item: ' + str(len(ent_id2idx) - len(head_set)))
    print('rel_num: ' + str(len(rel_id2idx)))

    random.seed = 1
    random.shuffle(triples)
    data_size = len(triples)
    print('whole data size: ' + str(data_size))
    trainSize = int(data_size * 0.8)
    validSize = int(data_size * 0.1)

    trainSet = triples[0: trainSize]
    validSet = triples[trainSize: trainSize + validSize]
    testSet = triples[trainSize + validSize:]

    print('raw_trainSize: ' + str(len(trainSet)))
    print('raw_validSize: ' + str(len(validSet)))
    print('raw_testSize: ' + str(len(testSet)))

    ent_in_train = set()
    rel_in_train = set()
    for triple in trainSet:
        ent_head, rel, ent_tail = triple
        ent_in_train.add(ent_head)
        ent_in_train.add(ent_tail)
        rel_in_train.add(rel)

    for triple in validSet:
        ent_head, rel, ent_tail = triple
        if ent_head not in ent_in_train or ent_tail not in ent_in_train or rel not in rel_in_train:
            validSet.remove(triple)
            trainSet.append(triple)
            # print('valid to train: ', triple)
    for triple in testSet:
        ent_head, rel, ent_tail = triple
        if ent_head not in ent_in_train or ent_tail not in ent_in_train or rel not in rel_in_train:
            testSet.remove(triple)
            trainSet.append(triple)
            # print('test to train: ', triple)

    print('trainSize: ' + str(len(trainSet)))
    print('validSize: ' + str(len(validSet)))
    print('testSize: ' + str(len(testSet)))

    if not os.path.exists(data_dir + '/kg/'):
        os.makedirs(data_dir + '/kg/')

    print_map(ent_id2idx, data_dir + '/kg/e_idx2id.dat')
    print_map(rel_id2idx, data_dir + '/kg/r_idx2id.dat')
    print_triples(trainSet, ent_id2idx, rel_id2idx, data_dir + '/kg/train.dat')
    print_triples(validSet, ent_id2idx, rel_id2idx, data_dir + '/kg/valid.dat')
    print_triples(testSet, ent_id2idx, rel_id2idx, data_dir + '/kg/test.dat')
    with open(data_dir + '/kg/kg_log.dat', 'w') as logfile:
        logfile.writelines(f'numEntity:{len(ent_id2idx)}\n'
                           f'numRelation:{len(rel_id2idx)}\n'
                           f'trainSize:{len(trainSet)}\n'
                           f'validSize:{len(validSet)}\n'
                           f'testSize:{len(testSet)}')

if __name__ == '__main__':
    random.seed(123)
    # electronics : 20, 7; CD: 15, 7; Steam: 12, 7
    # pre_process('::', '../../datasets/cd/', user_len=15, item_len=7)
    # pre_process('::', '../../datasets/ml1m/') kindle cd electronics
    # game 10, 10; others: 20, 25
    # pre_process('::', '../datasets/game/', 10, 10)
    # pre_process('::', '../datasets/kindle/', 20, 25)
    # pre_process('::', '../datasets/game/', 10, 10)
    pre_process('::', '../datasets/ml1m/', 1, 1)
    # pre_process('::', '../datasets/movie/', 20, 25)

