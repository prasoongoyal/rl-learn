import gym
import sys
import random
import numpy as np
from scipy.misc import imresize
from utils import *
from PIL import Image
from copy import deepcopy
import tensorflow as tf
from itertools import groupby
import pdb
import pickle
import torch
sys.path.insert(0, 'learn/')
from learn_model import LearnModel
from tasks import *

class GymEnvironment(object):
    def __init__(self, args, gamma):
        self.args = args
        self.env = gym.make(ENV_NAME)

        self.dims = (SCREEN_WIDTH, SCREEN_HEIGHT)

        self._screen = None
        self.reward = 0
        self.terminal = True

        self.reset()
        if self.args.lang_coeff > 0:
            self.setup_language_network()
            self.gamma = gamma

            # aggregates to compute Spearman correlation coefficients
            self.action_vectors_list = []
            self.rewards_list = []

    def reset(self):
        self.n_steps = 0
        self.action_vector = np.zeros(N_ACTIONS)
        self.potentials_list = []

    def new_game(self, from_random_game=False):
        self._screen = self.env.reset()
        self._step(0)
        self.initial_frame = None
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in xrange(random.randint(0, RANDOM_START - 1)):
            self._step(0)
        return self.screen, 0, 0, self.terminal

    def agent_pos(self):
        x, y = self.env.ale.getRAM()[42:44]
        return int(x), int(y)

    def skull_pos(self):
        return int(self.env.ale.getRAM()[47])

    def room(self):
        return int(self.env.ale.getRAM()[3])

    def has_key(self):
        return int(self.env.ale.getRAM()[101])

    def orb_collected(self):
        return int(self.env.ale.getRAM()[49])

    def save_state(self, filename):
            state = self.env.clone_full_state()
            np.save(filename, state)
            print ('File written : {}'.format(filename))

    def load_state(self, filename):
            state = np.load(filename)
            self.env.restore_full_state(state)
            self._step(0)

    def repeat_action(self, action, n=1):
        for _ in range(n):
            self._step(action)

    def inspect(self):
        screen = self.env.ale.getScreenRGB()
        img = Image.fromarray(screen.astype('uint8'))
        img.save('trajectory/'+str(self.n_steps)+'.png')
        if self.n_steps > 100:
            input('Done')

    def new_expt(self):
        if self.args.expt_id == 1:
            self.task = Task1(self)
        elif self.args.expt_id == 2:
            self.task = Task2(self)
        elif self.args.expt_id == 3:
            self.task = Task3(self)
        elif self.args.expt_id == 4:
            self.task = Task4(self)
        elif self.args.expt_id == 5:
            self.task = Task5(self)
        elif self.args.expt_id == 6:
            self.task = Task6(self)
        elif self.args.expt_id == 7:
            self.task = Task7(self)
        elif self.args.expt_id == 8:
            self.task = Task8(self)
        elif self.args.expt_id == 9:
            self.task = Task9(self)
        elif self.args.expt_id == 10:
            self.task = Task10(self)
        elif self.args.expt_id == 11:
            self.task = Task11(self)
        elif self.args.expt_id == 12:
            self.task = Task12(self)
        elif self.args.expt_id == 13:
            self.task = Task13(self)
        elif self.args.expt_id == 14:
            self.task = Task14(self)
        elif self.args.expt_id == 15:
            self.task = Task15(self)
            
        self._step(0)
        self._step(0)
        self._step(0)
        self._step(0)
        for _ in range(random.randint(0, RANDOM_START - 1)):
            self._step(0)

        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)
        self.n_steps += 1

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @ property
    def screen(self):
        return imresize(rgb2gray(self._screen)/255., self.dims)

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def act(self, action):
        start_lives = self.lives
        self.terminal = False
        self.action_vector[action] += 1.

        self._step(action)

        if start_lives > self.lives:
            self.terminal = True
        
        if not self.terminal:
            goal_reached = self.task.finished()
        else:
            goal_reached = False

        if goal_reached:
            self.reward = 1.0
            self.terminal = True
        else:
            self.reward = 0.0

        if self.args.lang_coeff > 0.0:
            lang_reward = self.args.lang_coeff * self.compute_language_reward()
            self.reward += lang_reward
        if self.n_steps > MAX_STEPS:
            self.terminal = True
        
        if self.terminal:
            self.reset()

        return self.state, goal_reached

    def setup_language_network(self):
        self.lang_net_graph = tf.Graph()
        with self.lang_net_graph.as_default():
            self.lang_network = LearnModel('predict', None, self.args.model_dir)
        sentence_id = (self.args.expt_id-1) * 3 + (self.args.descr_id-1)
        lang_data = pickle.load(open('./data/test_lang_data.pkl', 'rb'), encoding='bytes')
        self.lang = lang_data[sentence_id][self.args.lang_enc]

    def compute_language_reward(self):
        if self.n_steps < 2:
            logits = None
        else:
            with self.lang_net_graph.as_default():
                logits = self.lang_network.predict([self.action_vector], [self.lang])[0]

        if logits is None:
            self.potentials_list.append(0.)
        else:
            e_x = np.exp(logits - np.max(logits))
            self.potentials_list.append(e_x[1] - e_x[0] + self.args.noise * np.random.normal())

        self.action_vectors_list.append(list(self.action_vector[k] for k in spearman_corr_coeff_actions))
        self.rewards_list.append(self.potentials_list[-1])

        if len(self.potentials_list) > 1:
            lang_result = (self.gamma * self.potentials_list[-1] - self.potentials_list[-2])
            return lang_result
        else:
            return 0.

