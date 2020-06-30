import numpy as np
import gym
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内

import matplotlib.pyplot as plt
import collections

from model import LunarLanderModel
from agent import Agent

from parl.algorithms import DDPG

ACTOR_LR = 0.0002   # Actor网络更新的 learning rate
CRITIC_LR = 0.001   # Critic网络更新的 learning rate

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数


# 评估 agent, 跑 test_times 个episode，总reward求平均
def evaluate(env, agent, test_times, render=False):
    eval_reward = []
    for i in range(test_times):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)
            action = np.clip(action, -1.0, 1.0)
            action = action_mapping(action, env.action_space.low[0],
                                    env.action_space.high[0])

            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1
            if render:
                env.render();
            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


if __name__ == '__main__':

    # 创建LunarLanderContinuous环境
    env = gym.make("LunarLanderContinuous-v2")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    model = LunarLanderModel(act_dim)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)

    # 启动训练
    modeldir = './model'
    modelname = '/LanderDDPG_2949.ckpt'
    agent.restore(modeldir + modelname)

    test_times = 100
    solved_standard = 200.0
    # 用于实时绘制train reward的队列
    evalreward_curvesize = test_times
    evalreward_curve = collections.deque(maxlen=evalreward_curvesize)
    episodebuf = collections.deque(maxlen=evalreward_curvesize)

    plt.figure()
    solved = 0
    for i in range(test_times):
        evaluate_reward = evaluate(env, agent, 1, render=True)
        evalreward_curve.append(evaluate_reward)
        episodebuf.append(i+1)
        if evaluate_reward > solved_standard:
            solved += 1

        # 实时绘制train reward曲线
        plt.clf()
        plt.plot(episodebuf, evalreward_curve, "*")
        plt.xlabel('episode')
        plt.ylabel('evalReward')
        plt.pause(0.001)

    plt.savefig(modeldir + modelname + ".png")
    plt.close()
    logger.info("test_reward_mean: {}".format(np.mean(np.array(evalreward_curve))))
    logger.info("solved percent: {}%".format(solved*100.0/test_times))




