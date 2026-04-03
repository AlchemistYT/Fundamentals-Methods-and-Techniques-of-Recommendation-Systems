import data
import parameters
import trainer as trainer
path="datasets/"
benchmark="ML100K"
hyperParams = parameters.getHyperParams(benchmark)
user_pos_train,userCount,itemCount=data.loadTrainingData(benchmark,path)
testSet=data.loadTestData(benchmark,path)
useGPU=True
trainer.trainGAN(userCount,itemCount,user_pos_train,useGPU,hyperParams,testSet)


