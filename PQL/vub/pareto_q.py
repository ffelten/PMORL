from collections import namedtuple
from collections.abc import Iterable
import numpy as np
from pygmo import hypervolume
import tensorboardX as tb
import matplotlib.pyplot as plt
import pickle
plt.switch_backend('agg')
import datetime
import cv2


Log = namedtuple('Log', ['total_steps', 'episode', 'episode_step', 'reward'])


class Agent(object):

    def record_frame(self, episode, record_every, frames=None):
        if not episode % record_every:
            frame = self.env.render('rgb_array')
            w, h = frame.shape[:2]
            r = h / w
            w = np.minimum(w, 100)
            h = int(r * w)
            frame = cv2.resize(frame, (h, w))
            frame = np.expand_dims(frame, axis=0)
            if frames is None:
                return frame
            else:
                return np.append(frames, frame, axis=0)

    def train(self, episodes, max_steps=float('inf'), logdir='runs/'):
        self.writer = tb.SummaryWriter(logdir=logdir + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        record_every = 50

        total_steps = 0
        for e_i in range(1, episodes+1):
            e_step = 0
            e_reward = 0
            res = self.start(log=Log(total_steps, e_i, e_step, e_reward))
            frames = self.record_frame(e_i, record_every)
            while not res['terminal'] and e_step < max_steps:
                res = self.step(res, log=Log(total_steps, e_i, e_step, e_reward))
                frames = self.record_frame(e_i, record_every, frames)

                e_step += 1
                total_steps += 1
                e_reward += res['reward']
            log = Log(total_steps, e_i, e_step, e_reward)
            self.end(log=log, writer=self.writer)
            print(log)
            for k, v in log._asdict().items():
                if isinstance(v, Iterable):
                    for i, v_i in enumerate(v):
                        self.writer.add_scalar(k + '_{}'.format(i), v_i, e_i)
                else:
                    self.writer.add_scalar(k, v, e_i)
            if not e_i % record_every:
                frames = np.expand_dims(np.moveaxis(frames, -1, 1), 0)
                self.writer.add_video('episode_recording', frames, e_i)

    def start(self, log=None):
        raise NotImplementedError()

    def step(self, params, log=None):
        raise NotImplementedError()

    def end(self, log=None, writer=None):
        pass


class ParetoQ(Agent):

    def __init__(self, env, choose_action, ref_point, nO=2, gamma=1.):
        self.env = env
        self.choose_action = choose_action
        self.gamma = gamma

        self.ref_point = ref_point

        self.non_dominated = [[[np.zeros(nO)] for _ in range(env.nA)] for _ in range(env.nS)]
        self.avg_r = np.zeros((env.nS, env.nA, nO))
        self.n_visits = np.zeros((env.nS, env.nA))

    def start(self, log=None):
        self.epsilon = 1.
        state = self.env.reset()
        return {'observation': state,
                'terminal': False}

    def compute_q_set(self, s):
        q_set = []
        for a in range(self.env.nA):
            nd_sa = self.non_dominated[s][a]
            rew = self.avg_r[s, a]
            q_set.append([rew + self.gamma*nd for nd in nd_sa])
        return np.array(q_set)

    def update_non_dominated(self, s, a, s_n):
        q_set_n = self.compute_q_set(s_n)
        # update for all actions, flatten
        solutions = np.concatenate(q_set_n, axis=0)
        # append to current pareto front
        # solutions = np.concatenate([solutions, self.non_dominated[s][a]])

        # compute pareto front
        self.non_dominated[s][a] = get_non_dominated(solutions)

    def step(self, previous, log=None):
        state = previous['observation']
        q_set = self.compute_q_set(state)
        action = self.choose_action(state, q_set, self.epsilon)
        next_state, reward, terminal, _ = self.env.step(action)
        # update non-dominated set
        self.update_non_dominated(state, action, next_state)
        # update avg immediate reward
        self.n_visits[state, action] += 1
        self.avg_r[state, action] += (reward - self.avg_r[state, action]) / self.n_visits[state, action]

        self.epsilon *= 0.997
        return {'observation': next_state,
                'terminal': terminal,
                'reward': reward}

    def end(self, log=None, writer=None):

        if writer is not None:
            h_v = compute_hypervolume(self.compute_q_set(0), self.env.nA, self.ref_point)
            writer.add_scalar('hypervolume', np.amax(h_v), log.episode)
            fig = plt.figure()
            if self.avg_r.shape[2] == 3:
                ax = plt.axes(projection='3d')
            pareto = np.concatenate(self.non_dominated[0])
            if len(pareto):
                # number of objectives
                if pareto.shape[1] == 2:
                    plt.plot(pareto[:, 0], pareto[:, 1], 'o', label='estimated pareto-front')
                elif pareto.shape[1] == 3:
                    ax.plot3D(pareto[:, 0], pareto[:, 1], pareto[:, 2], 'o')

            plt.legend()
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = np.rollaxis(data, -1, 0)

            writer.add_image('pareto_front', data, log.episode)
            plt.close(fig)

    def evaluate(self, w=np.array([.5, .5])):

        p = self.start()
        state, done = p['observation'], p['terminal']
        norm = np.array([124, 19])
        i = np.ravel_multi_index((3, 5), self.env.shape)
        qi = self.compute_q_set(i)
        print([np.array(qi[a]) for a in range(4)])
        while not done:
            q = self.compute_q_set(state)
            q_s = [np.amax(np.sum(q[a]*w/norm, axis=1)) for a in range(self.env.nA)]
            action = np.random.choice(np.argwhere(q_s == np.amax(q_s)).flatten())
            print(q_s, action)
            state, reward, done, _ = self.env.step(action)

        print(w, reward)


def get_non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1

    return solutions[is_efficient]


def compute_hypervolume(q_set, nA, ref):
    q_values = np.zeros(nA)
    for i in range(nA):
        # pygmo uses hv minimization,
        # negate rewards to get costs
        points = np.array(q_set[i]) * -1.
        hv = hypervolume(points)
        # use negative ref-point for minimization
        q_values[i] = hv.compute(ref*-1)
    return q_values


def action_selection(state, q_set, epsilon, ref):
    q_values = compute_hypervolume(q_set, q_set.shape[0], ref)

    if np.random.rand() >= epsilon:
        return np.random.choice(np.argwhere(q_values == np.amax(q_values)).flatten())
    else:
        return np.random.choice(range(q_set.shape[0]))


if __name__ == '__main__':
    import gym
    import deep_sea_treasure
    from gym import wrappers

    env = gym.make('deep-sea-treasure-v0')
    ref_point = np.array([0, -25])
    agent = ParetoQ(env, lambda s, q, e: action_selection(s, q, e,  ref_point), ref_point, nO=2, gamma=1.)

    # env = gym.make('resource-gathering-v0')
    # ref_point = np.array([-1, -1, -2])
    # agent = ParetoQ(env, lambda s, q, e: action_selection(s, q, e, ref_point), ref_point, nO=3, gamma=0.96)

    logdir = 'runs/pareto-q/'
    agent.train(4000, logdir=logdir)
    print(agent.non_dominated[0])

    import pickle
    pickle.dump(agent.non_dominated, open('dst-pareto.pkl', 'rb'))