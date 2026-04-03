import tensorflow as tf
import numpy as np
import generator
import models
import pickle
import G_u as Gu
import xlwt
emb=20
workbook=xlwt.Workbook(encoding='utf-8')
booksheet=workbook.add_sheet('data', cell_overwrite_ok=True)
all_items = set(range(1683))
def trainGAN(userCount,itemCount,user_pos_train,useGPU,hp,user_pos_test):
    itemCount=1683
    with tf.Graph().as_default():
        param1=None
        param2=None
        popular_score=pickle.load(open("100k_popular_score.pkl", 'rb'), encoding='iso-8859-1')
        positive_items=tf.placeholder(tf.int32)
        negative_items=tf.placeholder(tf.int32)
        user=tf.placeholder(tf.int32)
        D_real,D_fake,D_real_fenbu,D_fake_fenbu,positive_item_embeddings,negative_item_embeddings,u_embedding,positive_bias,negative_bias,d_emb=models.DIS(positive_items,negative_items,user,itemCount,userCount,emb,hp,param=param2,_reuse=False)
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real,labels=tf.ones_like(D_real))+hp['reg_D'] * (tf.nn.l2_loss(u_embedding) + tf.nn.l2_loss(positive_item_embeddings) + tf.nn.l2_loss(positive_bias))
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))+hp['reg_D'] * (tf.nn.l2_loss(u_embedding) + tf.nn.l2_loss(negative_item_embeddings) + tf.nn.l2_loss(negative_bias))
        d_loss_fenbu = tf.exp(tf.reduce_sum(softmax(D_real_fenbu[0])*tf.log(softmax(D_fake_fenbu[0]))))

        #G
        G = generator.GEN(user,itemCount, userCount, emb, hp,param=param1,_reuse=False)
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        all_logits = tf.reduce_sum(tf.multiply(G.u_embedding, G.item_embeddings), 1) + G.item_bias
        score = tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(all_logits, [1, -1])), [-1]),negative_items)
        g_loss_g = -tf.reduce_mean(tf.log(score) * D_fake)+hp['reg_G'] * (tf.nn.l2_loss(G.u_embedding) + tf.nn.l2_loss(tf.nn.embedding_lookup(G.item_embeddings,negative_items)) + tf.nn.l2_loss(tf.gather(G.item_bias,negative_items)))

        #G_u
        fake_hunhe_items = tf.placeholder(tf.int32)
        G_u = Gu.GEN(fake_hunhe_items,itemCount, userCount, emb, hp, param=param1)
        g_u_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen_u')
        all_logits_u = tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(G_u.user_embeddings,G_u.fake_user), G_u.item_embeddings), 1) + G_u.item_bias
        score_u_ = tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(all_logits_u, [1, -1])), [-1]),fake_hunhe_items)
        score_u=tf.clip_by_value(score_u_,0.001,100)


        #D_u
        D_u_real,D_u_fake,D_u_real_fenbu,D_u_fake_fenbu,fake_user_embedding,user_embedding,fake_hunhe_items_embedding,fake_hunhe_items_bias=models.DIS_u(G_u.fake_user,user,fake_hunhe_items,itemCount, userCount, emb, hp,param=param2)
        d_u_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis_u')
        d_u_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_u_real, labels=tf.ones_like(D_u_real))+hp['reg_D_u']*(tf.nn.l2_loss(user_embedding)+tf.nn.l2_loss(fake_hunhe_items_embedding)+tf.nn.l2_loss(fake_hunhe_items_bias))
        d_u_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_u_fake, labels=tf.zeros_like(D_u_fake))+hp['reg_D_u']*(tf.nn.l2_loss(fake_user_embedding)+tf.nn.l2_loss(fake_hunhe_items_embedding)+tf.nn.l2_loss(fake_hunhe_items_bias))
        d_u_loss_fenbu=tf.reduce_sum(softmax(D_u_real_fenbu[0])*tf.log(softmax(D_u_fake_fenbu[0])))


        #G_u_loss
        g_u_loss_g=-tf.reduce_mean(tf.log(score_u)*D_u_fake)+hp['reg_G_u']*(tf.nn.l2_loss(fake_user_embedding)+tf.nn.l2_loss(fake_hunhe_items_embedding))

        cycle_loss_fenbu = -tf.reduce_sum(softmax(tf.nn.embedding_lookup(G.user_embeddings,user)[0]) * tf.log(softmax(tf.nn.embedding_lookup(G.user_embeddings,G_u.fake_user)[0])))

        d_loss = d_loss_real + d_loss_fake + d_loss_fenbu
        g_loss = g_loss_g+cycle_loss_fenbu
        d_u_loss = d_u_loss_real + d_u_loss_fake
        g_u_loss=g_u_loss_g

        D_optimizer,learning_rate1 = make_optimizer_d(d_loss,d_vars,hp,name='Adam_D')
        G_optimizer,learning_rate2=make_optimizer(g_loss,g_vars,hp,name='Adam_G')
        D_u_optimizer,learning_rate3=make_optimizer_d(d_u_loss,d_u_vars,hp,name='Adam_D_u')
        G_u_optimizer,learning_rate4=make_optimizer(g_u_loss,g_u_vars,hp,name='Adam_G_u')

        with tf.control_dependencies([G_optimizer, D_optimizer, G_u_optimizer, D_u_optimizer]):
            optimizers=tf.no_op(name='optimizers')
        with tf.control_dependencies([D_optimizer,D_u_optimizer]):
            optimizers_d= tf.no_op(name='optimizers_d')

        if useGPU == True:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        dis_log = open('dis_set.txt', 'w')
        gen_log = open('gen_cycle.txt', 'w')
        for epoch_1 in range(1000):
            best=0
            for epoch in range(10000):
                loss_d=0
                loss_g=0
                loss_d_u=0
                loss_g_u=0
                for u in user_pos_train:
                    sample_lambda=0.2
                    pos=user_pos_train[u]
                    ratings=sess.run(G.all_rating,feed_dict={user:[u]})
                    exp_rating=np.exp(ratings)
                    prob=exp_rating/np.sum(exp_rating)
                    pn=(1-sample_lambda)*prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    popular_score_pos=[popular_score[i] for i in pos]
                    popular_score_pos=np.exp(popular_score_pos)/np.sum(np.exp(popular_score_pos))
                    real_items=np.random.choice(pos,max(int(len(pos)*0.2),1),p=popular_score_pos)
                    real_items = list(set(pos).difference(set(real_items)))
                    fake_items=np.random.choice(np.arange(itemCount),len(real_items),p=pn)
                    for d in range(5):

                        _, dLoss, dULoss= sess.run(
                            [optimizers_d, d_loss,d_u_loss],
                            feed_dict={positive_items: real_items, negative_items: fake_items, user: [u],
                                    fake_hunhe_items: fake_items})
                    for p in range(1):
                        _,dLoss,gLoss,dULoss,gULoss,a,b,embedd= sess.run(
                            [optimizers,d_loss,g_loss,d_u_loss,g_u_loss,G_u.fake_user,cycle_loss_fenbu,d_emb],
                            feed_dict={positive_items:real_items,negative_items:fake_items,user:[u],fake_hunhe_items:fake_items})
                    loss_d=loss_d+np.sum(dLoss)
                    loss_g=loss_g+np.sum(gLoss)
                    loss_d_u=loss_d_u+np.sum(dULoss)
                    loss_g_u=loss_g_u+np.sum(gULoss)
                print("[%d] cost_d:%.4f,cost_g:%.4f,cost_gu:%.4f,cost_du:%.4f," % (epoch + 1,loss_d,loss_g,loss_g_u,loss_d_u))
                result=np.array([0.]*6)
                test_users = list(user_pos_test.keys())
                test_user_num = len(test_users)
                index=0
                batch_size=128
                while True:
                    if index>=test_user_num:
                        break
                    user_batch=test_users[index:index+batch_size]
                    index+=batch_size
                    user_batch_rating=sess.run(G.all_rating_1,{user:user_batch})
                    for i in range(len(user_batch)):
                        batch_result=simple_test_one_user(user_batch_rating[i],user_batch[i],user_pos_test,user_pos_train)
                        result+=batch_result

                result=list(result/test_user_num)
                print(result)
                if result[0]>best:
                    best=result[0]
                    best_epoch=epoch
                    param_g=sess.run([[G.user_embeddings,G.item_embeddings,G.item_bias]])
                    pickle.dump(param_g, open("g_100k_1.pkl", 'wb'))
                    pickle.dump(embedd, open("d_100k_1.pkl", 'wb'))
                    print('best:',best_epoch)
                buf = '\t'.join([str(x) for x in result])
                gen_log.write(str(epoch) + '\t' + buf + '\n')
                gen_log.flush()

        sess.close()


def simple_test_one_user(rating,u,user_pos_test,user_pos_train):
    if u in user_pos_train:
        test_items=list(all_items-set(user_pos_train[u]))
    else:
        test_items = list(all_items)
    item_score=[]
    for i in test_items:
        item_score.append((i,rating[i]))
    item_score=sorted(item_score,key=lambda x:x[1])
    item_score.reverse()
    item_sort=[x[0] for x in item_score]
    r=[]
    #pdb.set_trace()
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)
    p_3=np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])

    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])
def dcg_at_k(r, k):

    r = np.asfarray(r)[:k]

    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):

    dcg_max = dcg_at_k(sorted(r, reverse=True), k)

    if not dcg_max:

        return 0.

    return dcg_at_k(r, k) / dcg_max

def softmax(x):
    return tf.exp(x)/tf.reduce_sum(tf.exp(x),axis=0)

def make_optimizer(loss,variables,hp,name='Adam'):
    global_step=tf.Variable(0,trainable=False)
    starter_learning_rate=hp['starter_learning_rate']
    end_learning_rate=0.0
    start_decay_step=100000
    decay_steps = 100000
    beta1 = hp['beta1']
    learning_rate = (
        tf.where(
            tf.greater_equal(global_step, start_decay_step),
            tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                      decay_steps, end_learning_rate,
                                      power=1.0),
            starter_learning_rate
        )
    )
    lr=0.00008
    learning_step = (
        #tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
        tf.train.AdamOptimizer(lr,name=name)
            .minimize(loss, global_step=global_step, var_list=variables)
    )
    return learning_step,learning_rate
def make_optimizer_d(loss,variables,hp,name='Adam'):
    global_step=tf.Variable(0,trainable=False)
    starter_learning_rate=hp['starter_learning_rate']
    end_learning_rate=0.0
    start_decay_step=100000
    decay_steps = 100000
    beta1 = hp['beta1']
    learning_rate = (
        tf.where(
            tf.greater_equal(global_step, start_decay_step),
            tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                      decay_steps, end_learning_rate,
                                      power=1.0),
            starter_learning_rate
        )
    )
    lr=0.0008
    learning_step = (
        #tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
        tf.train.AdamOptimizer(lr,name=name)
            .minimize(loss, global_step=global_step, var_list=variables)
    )
    return learning_step,learning_rate