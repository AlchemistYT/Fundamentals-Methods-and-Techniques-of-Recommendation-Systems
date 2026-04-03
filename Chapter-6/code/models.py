

import tensorflow as tf

def DIS_u(fake_user,user,fake_hunhe_items,itemCount,userCount,emb,hp,param=None,_reuse=False):
    with tf.variable_scope("dis_u") as scope:
        if _reuse == True:
            scope.reuse_variables()
        init_random=tf.random_normal_initializer(mean=0.0,stddev=1.0,seed=None,dtype=tf.float32)
        init_truncated=tf.truncated_normal_initializer(mean=0.0,stddev=1.0,seed=None,dtype=tf.float32)
        if param==None:
            user_embeddings = tf.Variable(tf.random_uniform([userCount, emb], minval=-0.05, maxval=0.05, dtype=tf.float32))
            item_embeddings = tf.Variable(tf.random_uniform([itemCount, emb], minval=-0.05, maxval=0.05, dtype=tf.float32))
            item_bias=tf.Variable(tf.zeros([itemCount]))
        else:
            user_embeddings = tf.Variable(tf.cast(param[0], tf.float32))
            item_embeddings = tf.Variable(tf.cast(param[1], tf.float32))
            item_bias = tf.Variable(tf.cast(param[2], tf.float32))

        fake_user_embedding=tf.nn.embedding_lookup(user_embeddings,fake_user)
        user_embedding=tf.nn.embedding_lookup(user_embeddings,user)
        fake_hunhe_items_embedding=tf.nn.embedding_lookup(item_embeddings,fake_hunhe_items)
        fake_hunhe_items_bias=tf.gather(item_bias,fake_hunhe_items)

        reward_logits_fake = tf.reduce_sum(tf.multiply(fake_user_embedding, fake_hunhe_items_embedding), 1) + fake_hunhe_items_bias
        d_u_fake = 2 * (tf.sigmoid(reward_logits_fake) - 0.5)
        reward_logits_real = tf.reduce_sum(tf.multiply(user_embedding, fake_hunhe_items_embedding),1) + fake_hunhe_items_bias
        d_u_real=2 * (tf.sigmoid(reward_logits_real) - 0.5)
        d_u_real_fenbu=user_embedding
        d_u_fake_fenbu=fake_user_embedding
    return d_u_real,d_u_fake,d_u_real_fenbu,d_u_fake_fenbu,fake_user_embedding,user_embedding,fake_hunhe_items_embedding,fake_hunhe_items_bias

def DIS(positive_items,negative_items,user,itemCount,userCount,emb,hp,param=None,_reuse=False):
    with tf.variable_scope("dis") as scope:
        if _reuse == True:
            scope.reuse_variables()
        if param==None:
            user_embeddings = tf.Variable(tf.random_uniform([userCount, emb], minval=-0.05, maxval=0.05,dtype=tf.float32))
            item_embeddings = tf.Variable(tf.random_uniform([itemCount, emb], minval=-0.05, maxval=0.05,dtype=tf.float32))
            item_bias=tf.Variable(tf.zeros([itemCount]))
        else:
            user_embeddings = tf.Variable(tf.cast(param[0], tf.float32))
            item_embeddings = tf.Variable(tf.cast(param[1], tf.float32))
            item_bias = tf.Variable(tf.cast(param[2], tf.float32))
        d_params=[user_embeddings,item_embeddings,item_bias]
        u_embedding=tf.nn.embedding_lookup(user_embeddings,user)
        positive_items_embedding=tf.nn.embedding_lookup(item_embeddings,positive_items)
        negative_items_embedding=tf.nn.embedding_lookup(item_embeddings,negative_items)
        positive_bias=tf.gather(item_bias,positive_items)
        negative_bias=tf.gather(item_bias,negative_items)

        reward_logits_fake = tf.reduce_sum(tf.multiply(u_embedding,negative_items_embedding),1) + negative_bias
        d_fake = 2 * (tf.sigmoid(reward_logits_fake) - 0.5)
        reward_logits_real=tf.reduce_sum(tf.multiply(u_embedding,positive_items_embedding),1) + positive_bias
        d_real = 2 * (tf.sigmoid(reward_logits_real) - 0.5)
        d_real_fenbu=tf.reshape(positive_items_embedding,[1,-1])
        d_fake_fenbu=tf.reshape(negative_items_embedding,[1,-1])

    return d_real,d_fake,d_real_fenbu,d_fake_fenbu,positive_items_embedding,negative_items_embedding,u_embedding,positive_bias,negative_bias,d_params