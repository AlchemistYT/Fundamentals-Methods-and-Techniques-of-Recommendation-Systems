import tensorflow as tf
class GEN():
    def __init__(self,user,itemNum, userNum, emb_dim, hp, param=None,_reuse=False):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.param = param
        self.g_params = []
        self.user=user
        with tf.variable_scope('gen') as scope:
            if _reuse == True:
                scope.reuse_variables()
            init_random=tf.random_normal_initializer(mean=0.0,stddev=1.0,seed=None,dtype=tf.float32)
            init_truncated=tf.truncated_normal_initializer(mean=0.0,stddev=1.0,seed=None,dtype=tf.float32)
            init_uniform=tf.random_uniform_initializer(minval=0,maxval=1,seed=None,dtype=tf.float32)

            if self.param == None:
                self.user_embeddings=tf.get_variable('user_embeddings',shape=[self.userNum,self.emb_dim],initializer=init_uniform)
                self.item_embeddings=tf.get_variable('item_embeddings',shape=[self.itemNum,self.emb_dim],initializer=init_uniform)

                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(tf.cast(self.param[0], tf.float32))
                self.item_embeddings = tf.Variable(tf.cast(self.param[1], tf.float32))
                self.item_bias = tf.Variable(tf.cast(self.param[2], tf.float32))
            self.g_params = [self.user_embeddings, self.item_embeddings, self.item_bias]
            self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.user)
            self.all_rating = tf.reduce_sum(tf.multiply(self.u_embedding,self.item_embeddings),1)+self.item_bias
            self.all_rating_1=tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias
            self.all_rating_2=tf.clip_by_value(self.all_rating_1,0.0001,10)

            self.ratings=tf.matmul(self.user_embeddings,self.item_embeddings,transpose_a=False,transpose_b=True)+self.item_bias
