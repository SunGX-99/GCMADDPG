import argparse
import pickle
import numpy as np
import tensorflow as tf
import time
import tf_util as U
from GCmaddpg import MADDPGAgentTrainer
import tensorflow.compat.v1 as tf1
import tensorflow.contrib.layers as layers

# 使用 TensorFlow 1.x 兼容模式
tf1.disable_v2_behavior()


def gc_encoder(input, latent_dim, num_units=64):
    with tf1.variable_scope("gc_encoder"):
        out = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        latent_mean = layers.fully_connected(out, num_outputs=latent_dim, activation_fn=None)
        latent_log_std = layers.fully_connected(out, num_outputs=latent_dim, activation_fn=None)
        return latent_mean, latent_log_std

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # 环境
    parser.add_argument("--scenario", type=str, default="agent_obstacle", help="场景脚本名称")
    parser.add_argument("--max-episode-len", type=int, default=50, help="最大回合长度")
    parser.add_argument("--num-episodes", type=int, default=100000, help="回合数量")
    parser.add_argument("--num-adversaries", type=int, default=1000, help="对抗者数量")
    parser.add_argument("--good-policy", type=str, default="GC-maddpg", help="善意智能体的策略")
    parser.add_argument("--adv-policy", type=str, default="GC-maddpg", help="对抗者的策略")
    # 核心训练参数
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam优化器的学习率")
    parser.add_argument("--gamma", type=float, default=0.85, help="折扣因子")
    parser.add_argument("--batch-size", type=int, default=569, help="每次优化的回合数")
    parser.add_argument("--num-units", type=int, default=78, help="MLP中的单位数量")
    parser.add_argument("--latent-dim", type=int, default=16, help="GC中的潜在空间维度")  # 添加潜在维度参数
    # 检查点
    parser.add_argument("--exp-name", type=str, default=None, help="实验名称")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="保存训练状态和模型的目录")
    parser.add_argument("--save-rate", type=int, default=1000, help="每完成这么多回合保存一次模型")
    parser.add_argument("--load-dir", type=str, default="", help="加载训练状态和模型的目录")
    # 评估
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="用于基准测试的迭代次数")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="保存基准测试数据的目录")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="保存绘图数据的目录")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # 这个模型接受观察值作为输入，并返回所有动作的值
    with tf1.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from environment import MultiAgentEnv
    import scenario as scenarios

    # 从脚本加载场景
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # 创建世界
    world = scenario.make_world()
    # 创建多代理环境
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    latent_dim = arglist.latent_dim  # Use the latent dimension specified in the arguments

    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'GCmaddpg'),
            gc_encoder=gc_encoder, latent_dim=latent_dim))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'GCmaddpg'),
            gc_encoder=gc_encoder, latent_dim=latent_dim))
    return trainers

def train(arglist):
    from environment import MultiAgentEnv
    import scenario as simple_spread
    with U.single_threaded_session():
        # 创建环境
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # 创建代理训练器
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('使用的善意策略: {} 和对抗者策略: {}'.format(arglist.good_policy, arglist.adv_policy))

        # 初始化
        U.initialize()

        # 如果需要，加载之前的结果
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('加载之前的状态...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # 所有代理的奖励总和
        agent_rewards = [[0.0] for _ in range(env.n)]  # 单个代理的奖励
        final_ep_rewards = []  # 训练曲线的奖励总和
        final_ep_ag_rewards = []  # 训练曲线的代理奖励
        agent_info = [[[]]]  # 用于基准测试的占位符
        saver = tf1.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('开始迭代...')
        while True:
            # 获取动作
            action_n = []
            for agent, obs in zip(trainers, obs_n):
                latent_mean, latent_log_std = agent.gc_encoder(obs)
                latent_sample = latent_mean + tf.random_normal(tf.shape(latent_mean)) * tf.exp(latent_log_std)
                augmented_obs = np.concatenate([obs, latent_sample], axis=-1)
                action_n.append(agent.action(augmented_obs))

            # 环境步骤
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # 收集经验
            for i, agent in enumerate(trainers):
                latent_mean, latent_log_std = agent.gc_encoder(new_obs_n[i])
                latent_sample = latent_mean + tf.random_normal(tf.shape(latent_mean)) * tf.exp(latent_log_std)
                augmented_new_obs = np.concatenate([new_obs_n[i], latent_sample], axis=-1)
                agent.experience(obs_n[i], action_n[i], rew_n[i], augmented_new_obs, done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # 增加全局步骤计数器
            train_step += 1

            #


            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    round(time.time() - t_start, 3)))
                t_start = time.time()
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
                with open(arglist.plots_dir + arglist.exp_name + '_rewards.pkl', 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                with open(arglist.plots_dir + arglist.exp_name + '_agrewards.pkl', 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
