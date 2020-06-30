import parl
from parl import layers


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        self.fc1 = layers.fc(size=128, act='relu')
        # self.fc2 = layers.fc(size=64, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        hid1= self.fc1(obs)
        # hid2 = self.fc2(hid1)
        means = self.fc3(hid1)
        return means


class CriticModel(parl.Model):
    def __init__(self):
        hid_size = 120

        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        concat = layers.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = layers.squeeze(Q, axes=[1])
        return Q


class LunarLanderModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

