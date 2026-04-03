import numpy as np


def str_list_to_int(str_list):
    return [int(item) for item in str_list]
def str_list_to_float(str_list):
    return [float(item) for item in str_list]
def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges

def read_edges(filename):
    graph={}
    nodes=set()
    edges=read_edges_from_file(filename)

    for edge in edges:
        nodes.add(edge[0])
        if graph.get(edge[0]) is None:
            graph[edge[0]]=[]
        graph[edge[0]].append(edge[1])

def read_embeddings(filename,n_emb):
    with open(filename,"r") as f:
        lines=f.readlines()[1:]
        embedding_matrix=np.random.rand(2625,n_emb)
        for line in lines:
            emd=line.split()
            embedding_matrix[int(emd[0]), : ]=str_list_to_float(emd[1:])
        return embedding_matrix
def read_test_edges(test_filename):
    graph={}
    nodes=set()
    user_list=[]
    test_edges=read_edges_from_file(test_filename) if test_filename != "" else []
    for edge in test_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]]=[]
        if graph.get(edge[1]) is None:
            graph[edge[1]]=[]
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
        if edge[0] not in user_list:
            user_list.append(edge[0])
    return len(user_list),user_list,nodes,graph
def get_batch_data(x_real,y_real,x_fake,y_fake):
    user=[]
    item=[]
    label=[]
    train_size=len(x_real)
    for i in range(train_size):
        user.append(x_real[i])
        user.append(x_fake[i])
        item.append(y_real[i])
        item.append(y_fake[i])
        label.append(1.)
        label.append(0.)
def user_pos_train(u):
    user_pos_train = {}
    with open("train_pos.txt") as fin:
        for line in fin:
            line = line.split()
            uid = int(line[0])
            iid = int(line[1])
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]
    return user_pos_train[u]
