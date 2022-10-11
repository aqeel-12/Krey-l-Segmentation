import nltk
from architecture.utils import *
import numpy as np


class UnSupervised:
    def __init__(self, tmat, emission, initial_distrib):

        self.tmat = tmat
        self.emission = emission
        self.initial_prob = initial_distrib

        self.num_states = len(self.tmat)
        self.num_letters = len(self.emission[0])

    def maximize(self):  # viterbi
        return

    def decode(self):  # tagging the input
        return

    def estimate(self, o_seq):  # forward backward algorithm
        backward_table = np.zeros((self.num_states, len(o_seq)))  # vit table
        for i in range(len(o_seq)):  # Filling in the table
            prev_prob = tf.reshape(self.forward[i, :], [1, -1])

    def calculate_alpha_backward(self, o_seq, end_prob=[0.5, 0.5]):
        backward_table = np.zeros((self.num_states, len(o_seq)))  # vit table

        backward_table[:, -1] = end_prob
        #print(backward_table)

        for i in range(len(o_seq)-2, -1, -1):  # start filling in the table
            beta_s_t = 0
            for tag_cell in range(self.num_states):
                for tag_prev in range(self.num_states):
                    beta_s_t += backward_table[tag_prev][i+1] * \
                        self.tmat[tag_prev][tag_cell] * \
                        self.emission[tag_cell][o_seq[i]]
                backward_table[tag_cell][i] = beta_s_t
        return backward_table

    def calculate_beta_forward(self, o_seq):

        forward_table = np.zeros((self.num_states, len(o_seq)))  # vit table
        emission_t0 = [self.emission[i][o_seq[0]]
                       for i in range(self.num_states)]
        forward_table[:, 0] = np.multiply(emission_t0, self.initial_prob)

        for i in range(1, len(o_seq)):  # start filling in the table
            alpha_s_t = 0
            for tag_cell in range(self.num_states):
                for tag_prev in range(self.num_states):
                    alpha_s_t += forward_table[tag_prev][i-1] * \
                        self.tmat[tag_prev][tag_cell] * \
                        self.emission[tag_cell][o_seq[i]]
                forward_table[tag_cell][i] = alpha_s_t
        return forward_table
