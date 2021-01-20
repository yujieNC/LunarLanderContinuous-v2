# LunarLanderContinuous-v2-PARL-yj
本工程是百度强化学习7天打卡营的终极复现项目

AIstudio昵称：yujiekun77

二星环境-Box2D(LunarLanderContinuous-v2)

## 环境配置
numpy
gym
matplotlib
paddlepaddle-gpu(用CPU版也可以)
parl1.3.1
box2d-py


## 实现细节
LunarLanderContinuous-v2的目标就是使月球着陆器能稳稳当当地停在指定区域内。
与LunarLander-v2不同，LunarLanderContinuous-v2的state有8维，是连续的。Action有2维，数值范围都为(-1,1)，是连续的。其中，第一维控制主引擎，(-1,0)表示主引擎关闭，(0,1)表示功率由50%至100%。第二维控制左右引擎，(-1,-0.5)表示启动左引擎，(0.5,1)表示右引擎启动，(-0.5,0.5)表示左右引擎都不启动。
因此，本工程采用DDPG算法，使用的框架为paddlepaddle和PARL。

`model.py` ：Actor网络、Critic网络以及总模型文件

`agent.py`:智能体文件

`LunarLanderContinuous-v2_train.py`:模型训练，并实时绘制train_reward曲线(使用队列，保存最新的500组数据).每训练50个episode，测试10次(开启显示渲染)

`evaluate.py`:测试模型，运行100个episode，实时绘制得分散点图，并统计通关百分比(gym官网：得分大于200即认为通关)

### 训练策略
训练参数如下
`ACTOR_LR` = 0.0002   # Actor网络更新的 learning rate
`CRITIC_LR` = 0.001   # Critic网络更新的 learning rate
`GAMMA` = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
`TAU` = 0.001         # target_model 跟 model 同步参数 的 软更新参数
`MEMORY_SIZE` = 1000000   # replay memory的大小，越大越占用内存
`MEMORY_WARMUP_SIZE` = 10000      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
`BATCH_SIZE` = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
`TRAIN_TOTAL_EPISODE` = 1e4   # 总训练episode数
并在训练过程中，对于action加入标准差为0.5的高斯扰动
总体来说，进行比较谨慎的训练，直到train_reward有明显下降趋势时停止训练，中途保存测试效果好的多个模型

## 最终效果
从`/modeldir`中保存的模型得分图和模型效果统计表`/model/modelPerformance.ods`可以看出，本工程训练得到的模型，具有很高的通关率，并且平均得分也不错。其中，编号为2349和2949的模型均取得了100%的通关率。从可视化窗口可以看出，在2349模型指导下，Lander非常将谨慎，总体动作较慢，很谨慎地调整姿态。在2949模型的指导下，Lander非常激进，动作很快，以更少的动作快速着陆在指定位置，取得更高的得分。

## 其他文件
`/model`:保存模型参数和模型得分散点图

`/model/modelPerformance.ods`:各模型得分值和通关率统计

`output.gif`:模型`LanderDDPG_2949.ckpt`运行结果动图


