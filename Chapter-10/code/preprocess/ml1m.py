# convert the ml2k side information into kg triples ()
import random


def update_dict(item, my_dict):
    if item in my_dict:
        my_dict[item] += 1
    else:
        my_dict[item] = 1


def collect_triples(data_dir):
    """
        input: i_idx2id.dat, raw/XXX.dat,
        output: triplets.dat: [head_str, rel_str, tail_str]
    """
    # read item mapping
    ent_frequency = {}
    rel_frequency = {}
    rating_itemId_set = set()
    with open(data_dir + '/i_idx2id.dat', encoding='utf-8') as fin:
        for line in fin:
            itemIdx, itemId = line.strip().split('::')
            rating_itemId_set.add(itemId)

    # read tag mapping:
    tagId2str = {}
    with open(data_dir + '/raw/tags.dat', encoding='utf-8') as fin:
        lineCount = 0
        for line in fin:
            if lineCount == 0:
                lineCount += 1
                continue
            records = line.strip().split('\t')
            tag_id, tag_str = records[0], records[1]
            tagId2str[tag_id] = tag_str

    triples = []  # [head_str, rel_str, tail_str]
    item_with_addition = set()
    with open(data_dir + '/raw/movie_actors.dat', encoding='windows-1252') as fin:
        lineCount = 0
        for line in fin:
            if lineCount == 0:
                lineCount += 1
                continue
            records = line.strip().split('\t')
            itemId, actor_str, ranking = records[0], 'ent_actor_' + records[1], int(records[3])
            if itemId in rating_itemId_set and ranking <= 2:
                update_dict(actor_str, ent_frequency)
                update_dict('rel_actor', rel_frequency)
                triples.append([itemId, 'rel_actor', actor_str])
                item_with_addition.add(itemId)

    country_set = set()
    with open(data_dir + '/raw/movie_countries.dat', encoding='windows-1252') as fin:
        lineCount = 0
        for line in fin:
            if lineCount == 0:
                lineCount += 1
                continue
            records = line.strip().split('\t')
            if len(records) > 1:
                itemId, country_str = records[0], 'ent_country_' + records[1]
                if itemId in rating_itemId_set:
                    update_dict(country_str, ent_frequency)
                    update_dict('rel_country', rel_frequency)
                    triples.append([itemId, 'rel_country', country_str])
                    country_set.add(country_str)
                    item_with_addition.add(itemId)

    director_set = set()
    with open(data_dir + '/raw/movie_directors.dat', encoding='windows-1252') as fin:
        lineCount = 0
        for line in fin:
            if lineCount == 0:
                lineCount += 1
                continue
            records = line.strip().split('\t')
            if len(records) > 1:
                itemId, director_str = records[0], 'ent_director_' + records[1]
                if itemId in rating_itemId_set:
                    update_dict(director_str, ent_frequency)
                    update_dict('rel_director', rel_frequency)
                    triples.append([itemId, 'rel_director', director_str])
                    director_set.add(director_str)
                    item_with_addition.add(itemId)

    genre_set = set()
    with open(data_dir + '/raw/movie_genres.dat', encoding='windows-1252') as fin:
        lineCount = 0
        for line in fin:
            if lineCount == 0:
                lineCount += 1
                continue
            records = line.strip().split('\t')
            if len(records) > 1:
                itemId, genre_str = records[0], 'ent_genre_' + records[1]
                lineCount += 1
                if itemId in rating_itemId_set:
                    update_dict(genre_str, ent_frequency)
                    update_dict('rel_genre', rel_frequency)
                    triples.append([itemId, 'rel_genre', genre_str])
                    genre_set.add(genre_str)
                    item_with_addition.add(itemId)

    tag_set = set()
    with open(data_dir + '/raw/movie_tags.dat', encoding='windows-1252') as fin:
        lineCount = 0
        for line in fin:
            if lineCount == 0:
                lineCount += 1
                continue
            records = line.strip().split('\t')
            if len(records) > 1:
                itemId, tag_id = records[0], records[1]
                lineCount += 1
                if itemId in rating_itemId_set:
                    tag_str = 'ent_tag_' + tagId2str[tag_id]
                    update_dict('rel_tag', rel_frequency)
                    update_dict(tag_str, ent_frequency)
                    triples.append([itemId, 'rel_tag', tag_str])
                    tag_set.add(tag_str)
                    item_with_addition.add(itemId)

    print('rating item num: ' + str(len(rating_itemId_set)))
    print('item with additional info: ' + str(len(item_with_addition)))

    # print valid items
    valid_item_print = []
    for itemId in rating_itemId_set:
        valid_item_print.append(itemId + '\n')

    triple_write = []
    for triple in triples:
        head, rel, tail = triple
        triple_write.append(head + '\t' + rel + '\t' + tail + '\n')
        # head, rel, tail = triple
        # if rel is 'rel_director' and ent_frequency[tail] < 2:
        #     print(f"delete director: {tail}")
        # elif rel in ['rel_actor', 'rel_country', 'rel_genre'] and ent_frequency[tail] < 2:
        #     continue
        #     # print("delet %s: " % (rel) + head + '#%d\t' % (ent_frequency[head]) + rel + '#%d\t' % (
        #     # rel_frequency[rel]) + tail + '#%d' % (ent_frequency[tail]))
        # elif rel is 'rel_tag' and ent_frequency[tail] < 3:
        #     continue
        #     # print("delet tag: " + head + '#%d\t' % (ent_frequency[head]) + rel + '#%d\t' % (
        #     # rel_frequency[rel]) + tail + '#%d' % (ent_frequency[tail]))
        # elif rel_frequency[rel] <= 5:
        #     print(f"delete rel: {tail}")
        # else:
        #     triple_write.append(head + '\t' + rel + '\t' + tail + '\n')

    with open(data_dir + '/raw/triples.dat', 'w', encoding='utf-8') as fout:
        fout.writelines(triple_write)


def split_triples(data_dir):
    """
    input: triples.dat
    output: e_map.dat, r_map.dat, train.dat, valid.dat, test.dat
    """
    ent_map = {}
    rel_map = {}

    triples = []
    ent_count = 0
    rel_count = 0
    with open(data_dir + '/raw/triples.dat', encoding='utf-8') as fin:
        for line in fin:
            if len(line.strip().split('\t')) is not 3:
                print(line)
                continue
            ent_head, rel, ent_tail = line.strip().split('\t')
            triples.append([ent_head, rel, ent_tail])
            if ent_head not in ent_map:
                ent_map[ent_head] = ent_count
                ent_count += 1
            if ent_tail not in ent_map:
                ent_map[ent_tail] = ent_count
                ent_count += 1
            if rel not in rel_map:
                rel_map[rel] = rel_count
                rel_count += 1

    print('entity_num: ' + str(len(ent_map)))
    print('rel_num: ' + str(len(rel_map)))

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
            print('valid to train: ', triple)

    for triple in testSet:
        ent_head, rel, ent_tail = triple
        if ent_head not in ent_in_train or ent_tail not in ent_in_train or rel not in rel_in_train:
            testSet.remove(triple)
            trainSet.append(triple)
            print('test to train: ', triple)

    print('trainSize: ' + str(len(trainSet)))
    print('validSize: ' + str(len(validSet)))
    print('testSize: ' + str(len(testSet)))

    print_map(ent_map, data_dir + '/kg/e_map.dat')
    print_map(rel_map, data_dir + '/kg/r_map.dat')
    print_triples(trainSet, ent_map, rel_map, data_dir + '/kg/train.dat')
    print_triples(validSet, ent_map, rel_map, data_dir + '/kg/valid.dat')
    print_triples(testSet, ent_map, rel_map, data_dir + '/kg/test.dat')


def print_map(map_to_print, print_path):
    print_lines = []
    for itemId, itemIdx in sorted(map_to_print.items(), key=lambda item: item[1]):
        print_lines.append(str(itemIdx) + '\t' + itemId + '\n')

    with open(print_path, 'w', encoding='utf-8') as fout:
        fout.writelines(print_lines)


def print_triples(triples, ent_map, rel_map, print_path):
    print_lines = []
    for triple in triples:
        head, rel, tail = triple
        headIdx = ent_map[head]
        tailIdx = ent_map[tail]
        relIdx = rel_map[rel]
        print_lines.append(str(headIdx) + '\t' + str(tailIdx) + '\t' + str(relIdx) + '\n')

    with open(print_path, 'w', encoding='utf-8') as fout:
        fout.writelines(print_lines)


def convert_rating_data(dir_path):
    pass


if __name__ == '__main__':
    collect_triples('../datasets/ml1m/')
    # split_triples('../../datasets/cd/')
    # split_triples('../../datasets/electronics/')
    # split_triples('../../datasets/sports/')