import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 1
        num_good_agents =5
        num_adversaries =5
        num_agents = num_adversaries + num_good_agents
        num_landmarks =1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.055 if agent.adversary else 0.0000000005
            agent.accel = 3.0 if agent.adversary else 0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.10
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):

        # random properties for agents
        for i, agent in enumerate(world.agents):

            if i == 0:
                agent.color = np.array([0.35, 0.35, 0.85])#深蓝
            elif i == 1:
                agent.color = np.array([0.85, 0.95, 0.25])#黄色
            elif i == 2:
                agent.color = np.array([0.85, 0.35, 0.25])#棕色
            elif i == 3:
                agent.color = np.array([0.15, 0.95, 0.85])#浅蓝色
            elif i == 4:
                agent.color = np.array([0.85, 0.35, 0.85])#粉红
            else:
                agent.color = np.array([1.00, 1.00, 1.00])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for i, landmark in enumerate(world.landmarks):

            landmark.state.p_pos = np.random.uniform(0.2, 0.5, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


        for i, agent in enumerate(world.agents):
                if i == 0:
                  agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
                  agent.state.p_vel = np.zeros(world.dim_p)
                  agent.state.c = np.zeros(world.dim_c)
                elif i == 1:
                  agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
                  agent.state.p_vel = np.zeros(world.dim_p)
                  agent.state.c = np.zeros(world.dim_c)
                elif i == 2:
                  agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
                  agent.state.p_vel = np.zeros(world.dim_p)
                  agent.state.c = np.zeros(world.dim_c)
                elif i == 3:
                  agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
                  agent.state.p_vel = np.zeros(world.dim_p)
                  agent.state.c = np.zeros(world.dim_c)
                elif i == 4:
                  agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
                  agent.state.p_vel = np.zeros(world.dim_p)
                  agent.state.c = np.zeros(world.dim_c)
                elif i == 5:
                  agent.state.p_pos = np.array([-0.002, 0.02])
                  agent.state.p_vel = np.zeros(world.dim_p)
                elif i == 6:
                  agent.state.p_pos = np.array([-0.25, 0.25])
                  agent.state.p_vel = np.zeros(world.dim_p)
                elif i == 7:
                  agent.state.p_pos = np.array([-0.5, 0.5])
                  agent.state.p_vel = np.zeros(world.dim_p)
                elif i == 8:
                  agent.state.p_pos = np.array([-0.25, -0.25])
                  agent.state.p_vel = np.zeros(world.dim_p)
                elif i == 9:
                  agent.state.p_pos = np.array([-0.5, -0.5])
                  agent.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        # shape = True
        # adversaries = self.adversaries(world)
        # if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
        #     # for adv in adversaries:
        #     #     rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        #     for adv in adversaries:
        #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in world.agents]
        #         rew -= min(dists)
        # if agent.collide:
        #     for a in adversaries:
        #         if self.is_collision(a, agent):
        #             rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        # return rew
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        i = 0
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for a in agents:
                dists = [np.sqrt(np.sum(np.square(adv.state.p_pos - a.state.p_pos))) for adv in adversaries]
                rew -= dists[i]
                i = i + 1
        if agent.collide:
            for l in world.landmarks:
                for adv in adversaries:
                    if self.is_collision(adv, l):
                        rew -= 1
        if agent.collide:
            for a in adversaries:
                for ad in adversaries:
                    if self.is_collision(ad, a):
                        rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
