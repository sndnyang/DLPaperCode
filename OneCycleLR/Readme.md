# OneCycleLR scheduler

《Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates》

Paper: https://arxiv.org/abs/1708.07120 

Conclusion:

1/10 - 1/5 training time, but 3%-5% accuracy drops -- constant learning rate with decaying may also achieve this.

Almost failed, Because the baseline is weak, their results are better than a very weak baseline. 


## ResNet56 for cifar10:

Setting from this repo: https://github.com/sgugger/Deep-Learning/blob/master/Cyclical%20LR%20and%20momentums.ipynb

LR range test

Conclusion:

The baseline of the paper is skeptical-- SGD lr=0.1 with batch size 64 can achieve 93.5% in 100 epochs/~10000 iterations( batch size=512 can only achieve 91.4% as they show in the paper). 

## ResNet18 and ResNet50 for imagenet

LR range test

and 

some results' log files

Conclusion:

The baseline of the paper is skeptical-- 

Pytorch official resnet50：76.2% within 100 epochs. Their paper uses 80 epochs < 70% accuracy.

I tried 12-20 epochs,  a 5% percents gap ~ 70.5%
