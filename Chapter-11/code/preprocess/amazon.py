import json

from ml1m import update_dict
from ml1m import split_triples

def generate_ratings_and_itemDetails(dir_path):
    item_attributes = read_meta(dir_path + '/meta.json')
    valid_items = set()  # items that are in the meta and rating simultaneously
    user_set = set()

    # generate rating data
    out_rating_lines = []
    with open(dir_path + '/raw_ratings.json') as fin:
        for line in fin:
            json_data = json.loads(line)
            # get properties
            itemId = str(json_data['asin'])
            if itemId in item_attributes:
                valid_items.add(itemId)
                userId = str(json_data['reviewerID'])
                user_set.add(userId)
                timeStamp = str(json_data['unixReviewTime'])
                rating = str(json_data['overall'])
                out_rating_lines.append(userId + '::' + itemId + '::' + rating + '::' + timeStamp + '\n')
    with open(dir_path + '/ratings.dat', 'w', encoding='utf-8') as fout:
        fout.writelines(out_rating_lines)
    print(f'num_ratings: {len(out_rating_lines)}')
    print(f'num_users: {len(valid_items)}')
    print(f'num_items: {len(user_set)}')

    # generate itemId2Str
    itemId2Str = []
    for itemId, attributes in item_attributes.items():
        if len(attributes) <= 0:
            continue
        if itemId in valid_items:
            itemId2Str.append(itemId + '::' + json.dumps(attributes) + '\n')
    with open(dir_path + '/itemId2Str.dat', 'w', encoding='utf-8') as fout:
        fout.writelines(itemId2Str)

    collect_triples(dir_path)

def read_meta(file_path):
    item_attributes = {}
    with open(file_path) as fin:
        for line in fin:
            json_data = json.loads(line)
            attributes = {'title':[], 'category': [], 'brand': [], 'description': [], 'rank': []}
            # get properties
            itemId = str(json_data['asin'])
            if 'title' in json_data:
                attributes['title'] = json_data['title']
            if 'category' in json_data:
                attributes['category'] = json_data['category']
            if 'brand' in json_data:
                attributes['brand'] = json_data['brand']
            if 'description' in json_data:
                attributes['description'] = json_data['description']
            if 'rank' in json_data:
                attributes['rank'] = json_data['rank']
            item_attributes[itemId] = attributes
    return item_attributes

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

    triples = []
    item_attributes = read_meta(data_dir + '/meta.json')
    for itemId in item_attributes:
        if itemId in rating_itemId_set:
            content = item_attributes[itemId]
            for attribute_type in content:
                attribute_values = content[attribute_type]
                for attribute_value in attribute_values:
                    update_dict(attribute_value, ent_frequency)
                    update_dict(attribute_type, rel_frequency)
                    triples.append([itemId, attribute_type, attribute_value])

    print('rating item num: ' + str(len(rating_itemId_set)))

    # print valid items
    valid_item_print = []
    for itemId in rating_itemId_set:
        valid_item_print.append(itemId + '\n')

    triple_write = []
    for triple in triples:
        head, rel, tail = triple
        if ent_frequency[tail] < 2:
            print(f"delete {rel} : {tail}")
        else:
            triple_write.append(head + '\t' + rel + '\t' + tail + '\n')
    with open(data_dir + '/triples.dat', 'w', encoding='utf-8') as fout:
        fout.writelines(triple_write)

if __name__ == '__main__':
    # read_raw_review_data('../../datasets/beauty/') electronics, cd
    generate_ratings_and_itemDetails('../datasets/movie/')
    generate_ratings_and_itemDetails('../datasets/kindle/')
    generate_ratings_and_itemDetails('../datasets/cd/')
    generate_ratings_and_itemDetails('../datasets/game/')


