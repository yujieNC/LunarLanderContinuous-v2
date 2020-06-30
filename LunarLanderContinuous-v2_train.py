import numpy as np
import gym
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
from parl.utils import ReplayMemory # 经验回放
import matplotlib.pyplot as plt
import collections

from model import LunarLanderModel
from agent import Agent

from parl.algorithms import DDPG

ACTOR_LR = 0.0002   # Actor网络更新的 learning rate
CRITIC_LR = 0.001   # Critic网络更新的 learning rate

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1000000   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 10000      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
# REWARD_SCALE = 1       # reward 的缩放因子
BATCH_SIZE = 256          # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_EPISODE = 1e4   # 总训练episode数


def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, 0.5), -1.0, 1.0)
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action, reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                    batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


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

    # parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
    rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)


    modeldir = './model'

    # 启动训练
    agent.restore(modeldir+'/LanderDDPG_2349.ckpt')

    last_test_reward = 0;
    episode_now = 0
    test_times = 5

    # 用于实时绘制train reward的队列
    trainreward_curvesize = 500
    trainreward_curve = collections.deque(maxlen=trainreward_curvesize)
    episodebuf = collections.deque(maxlen=trainreward_curvesize)

    plt.figure()
    while episode_now < TRAIN_TOTAL_EPISODE:
        train_reward, steps = run_episode(env, agent, rpm)
        episode_now += 1
        logger.info('episode: {} Reward: {}'.format(episode_now, train_reward)) # 打印训练reward

        trainreward_curve.append(train_reward)
        episodebuf.append(episode_now)

        # 实时绘制train reward曲线
        plt.clf()
        plt.plot(episodebuf, trainreward_curve)
        plt.xlabel('episode_train')
        plt.ylabel('trainReward')
        plt.pause(0.001)

        if (episode_now + 1) % 50 == 0:  # 每隔一定step数，评估一次模型
            evaluate_reward = evaluate(env, agent, test_times, render=True)
            logger.info('Steps {}, Test reward: {}'.format(episode_now, evaluate_reward))  # 打印评估的reward
            if evaluate_reward > last_test_reward:
                agent.save(modeldir + '/LanderDDPG_{}.ckpt'.format(episode_now))
                last_test_reward = evaluate_reward

    plt.close()

    agent.restore(modeldir + '/LanderDDPG_{}'.format(TRAIN_TOTAL_EPISODE))

