import random
from collections import defaultdict

def getHyperParams(benchmark):
    hyperParams = {}
    if benchmark == "ML100K":
        hyperParams['epochs'] = random.choice(
        [750, 800, 850, 900, 950, 1000])  # quite sensitive to this value, hence trying multiple values


        hyperParams['hiddenDim_D'] = 200
        hyperParams['reg_G'] = 0.001
        hyperParams['reg_G_u'] = 0.001
        hyperParams['reg_D'] = 0.001
        hyperParams['reg_D_u'] = 0.001

        hyperParams['lr_G'] = 0.001
        hyperParams['lr_G_u'] = 0.001
        hyperParams['lr_cycle'] = 0.000005
        hyperParams['lr_D_u'] = 0.001
        hyperParams['lr_D'] = 0.001

        hyperParams['batchSize_G'] = 32
        hyperParams['batchSize_D'] = 64
        #adam,sgd,rms
        hyperParams['opt_G'] = 'rms'
        hyperParams['opt_D'] = 'rms'
        hyperParams['opt_D_u'] = 'rms'
        hyperParams['opt_G_u'] = 'rms'
        hyperParams['opt_cycle'] = 'adam'

        hyperParams['hiddenLayer_D'] = 1
        hyperParams['step_G'] = 1
        hyperParams['step_D'] = 1
        hyperParams['sample_num']=380
        hyperParams['starter_learning_rate'] = 0
        hyperParams['beta1'] = 0.5

    if benchmark == "pinterest-20":
        hyperParams['epochs'] = random.choice(
        [750, 800, 850, 900, 950, 1000])  # quite sensitive to this value, hence trying multiple values

        hyperParams['sample_num'] = 150
        #sample=500时，按tf.multinomial采样要优于topk

        #D
        hyperParams['hiddenDim_D'] = 100
        hyperParams['hiddenLayer_D'] = 1
        hyperParams['step_D'] = 1
        hyperParams['opt_D'] = 'adam'
        hyperParams['reg_D'] = 0.001
        hyperParams['lr_D'] = 0.00001


        #G
        hyperParams['step_G'] = 1
        hyperParams['reg_G'] = 0.001
        hyperParams['lr_G'] = 0.00001
        hyperParams['opt_G'] = 'adam'

        #G_u
        hyperParams['reg_G_u'] = 0.001
        hyperParams['lr_G_u'] = 0.0001
        hyperParams['opt_G_u'] = 'rms'
        hyperParams['sample_user']=10

        #D_u
        hyperParams['reg_D_u'] = 0.001
        hyperParams['lr_D_u'] = 0.001
        hyperParams['opt_D_u'] = 'rms'
        #rms比adam好


        #cycle
        hyperParams['lr_cycle'] = 0.001
        hyperParams['opt_cycle'] = 'adam'
        hyperParams['lambda1']=1
        hyperParams['lambda2'] = 1
        hyperParams['starter_learning_rate']=0
        hyperParams['beta1']=0.5

        hyperParams['batchSize_G'] = 32
        hyperParams['batchSize_D'] = 64
    if benchmark == "ML1M":
        hyperParams['epochs'] = random.choice(
        [750, 800, 850, 900, 950, 1000])  # quite sensitive to this value, hence trying multiple values

        hyperParams['sample_num'] = 150
        #sample=500时，按tf.multinomial采样要优于topk

        #D
        hyperParams['hiddenDim_D'] = 100
        hyperParams['hiddenLayer_D'] = 1
        hyperParams['step_D'] = 1
        hyperParams['opt_D'] = 'adam'
        hyperParams['reg_D'] = 0.001
        hyperParams['lr_D'] = 0.00001


        #G
        hyperParams['step_G'] = 1
        hyperParams['reg_G'] = 0.001
        hyperParams['lr_G'] = 0.00001
        hyperParams['opt_G'] = 'adam'

        #G_u
        hyperParams['reg_G_u'] = 0.001
        hyperParams['lr_G_u'] = 0.0001
        hyperParams['opt_G_u'] = 'rms'
        hyperParams['sample_user']=10

        #D_u
        hyperParams['reg_D_u'] = 0.001
        hyperParams['lr_D_u'] = 0.001
        hyperParams['opt_D_u'] = 'rms'
        #rms比adam好


        #cycle
        hyperParams['lr_cycle'] = 0.001
        hyperParams['opt_cycle'] = 'adam'
        hyperParams['lambda1']=1
        hyperParams['lambda2'] = 1
        hyperParams['starter_learning_rate']=0
        hyperParams['beta1']=0.5

        hyperParams['batchSize_G'] = 32
        hyperParams['batchSize_D'] = 64
    return hyperParams

