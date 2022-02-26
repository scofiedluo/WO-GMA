### 训练策略
* 先训练一定轮数的TPG分支，再训练一定轮数的OAR分支，但每轮的每个step都需要根据训好的TPG计算proposals，如此迭代进行。

* 先训练一定轮数的TPG分支，末尾轮数的TPG结果记录下来，同时该轮的dataloader的结果也记录下来，训练OAR的时候，不再重新使用dataloader，而是使用上一轮每个batch的结果以及相应的proposals，每个batch可重复利用多次




### idea
TPG 训练完后所有video_level 的feature平均作为lstm的hidden state 输入