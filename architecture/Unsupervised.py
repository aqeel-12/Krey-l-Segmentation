import nltk
from architecture.utils import *
import numpy as np


from collections import defaultdict
import copy


class UnSupervised:
    def __init__(self, tmat, emission, initial_distrib):

        self.tmat = tmat
        self.emission = emission
        self.initial_prob = initial_distrib

        self.num_states = len(self.tmat)
        self.num_letters = len(self.emission[0])

    def calculate_beta_backward(self, o_seq, tmat, emission, end_prob=[0.5, 0.5]):
        backward_table = np.zeros((self.num_states, len(o_seq)))  # vit table

        backward_table[:, -1] = end_prob
        #print(backward_table)

        for i in range(len(o_seq)-2, -1, -1):  # start filling in the table
            beta_s_t = 0
            for tag_cell in range(self.num_states):
                for tag_prev in range(self.num_states):
                    beta_s_t += backward_table[tag_prev][i+1] * \
                        tmat[tag_cell][tag_prev] * \
                        emission[tag_prev][o_seq[i+1]]
                backward_table[tag_cell][i] = beta_s_t
                beta_s_t = 0
        return backward_table

    def calculate_alpha_forward(self, o_seq, tmat, emission, initial_prob):

        forward_table = np.zeros((self.num_states, len(o_seq)))  # vit table
        emission_t0 = [emission[i][o_seq[0]]
                       for i in range(self.num_states)]
        forward_table[:, 0] = np.multiply(emission_t0, initial_prob)

        for i in range(1, len(o_seq)):  # start filling in the table
            alpha_s_t = 0
            for tag_cell in range(self.num_states):
                for tag_prev in range(self.num_states):
                    alpha_s_t += forward_table[tag_prev][i-1] * \
                        tmat[tag_prev][tag_cell] * \
                        emission[tag_cell][o_seq[i]]
                forward_table[tag_cell][i] = alpha_s_t
                alpha_s_t = 0
        return forward_table

    def baum_welch(self, o_seq, n_iter):
        tmat = copy.deepcopy(self.tmat)
        emission = copy.deepcopy(self.emission)
        initial = self.initial_prob
        end = [0.5, 0.5]
        M = len(tmat[0])
        T = len(o_seq)

        for _ in range(n_iter):
            alpha = self.calculate_alpha_forward(
                o_seq, tmat, emission, initial)  # matrix

            beta = self.calculate_beta_backward(o_seq, tmat, emission, end)

            prod_alpha_beta = alpha*beta
            prod_alpha_beta_normed = prod_alpha_beta / \
                sum(prod_alpha_beta)  # normalized alpha dot beta

            #emission aux is well emission auxillary Sum(P(state_i |observed_t) of all time t )
            #-----seq-------
            #|
            #state
            #|
            emission_aux = defaultdict(lambda: defaultdict(float))
            #transition aux.
            # LESSON LEARNED!! KNOW MATRIX ALGEBRA LIKE A PRO to avoid calculating entry by entry like this!!
            #P(state_i|state_j)
            transition_aux = defaultdict(lambda: defaultdict(float))

            for i, obs in enumerate(o_seq):
                for state in range(self.num_states):
                    emission_aux[state][obs] += prod_alpha_beta_normed[state][i]
                    if i == 0:
                        continue
                    else:
                        for state2 in range(self.num_states):
                            prev_step = alpha[state2][i-1]*tmat[state2][state]
                            beta_now = beta[state][i]*emission[state][obs]
                            #print(alpha[state2][i-1])
                            #print(obs, state, state2,
                            #      alpha[state][i-1], tmat[state][state2],
                            #      beta[state][i],emission[state][obs])
                            #print(obs, state, state2, prev_step*beta_now/(sum(prod_alpha_beta)[i]))
                            transition_aux[state2][state] += prev_step * \
                                beta_now/(sum(prod_alpha_beta)[i])
            #print(tmat[state2][state])
            new_tmat = np.zeros((self.num_states, self.num_states))
            for si in range(self.num_states):
                norm_factor = sum(prod_alpha_beta_normed[si])
                for sj in range(self.num_states):
                    new_tmat[sj][si] = transition_aux[si][sj]/norm_factor
                for o in set(o_seq):
                    #print(si, o, emission_aux[si][o]/norm_factor)
                    if np.isnan(emission_aux[si][o]/norm_factor):
                        emission[si][o] = 0
                    else:
                        emission[si][o] = emission_aux[si][o]/norm_factor
            if not np.isnan(sum(sum(new_tmat))): #dont change the tmat if it is not 0
                tmat = new_tmat
            initial = [i[0] for i in prod_alpha_beta_normed]
            end = [term[-1]/sum(prod_alpha_beta_normed[si])
                   for si, term in enumerate(prod_alpha_beta_normed)]
            #print("this is tmat", self.tmat)
        return tmat, emission, initial, end
        #return tmat, emit, start, stop

    #' Authors: the following 3 mtehods are borrowed from OmarAlAkkad and a8nguyen from challenge 2!


    def _argmax(self, V, tag_list, t, i, transition_mat):
        ans = -1
        best = None
        for s in tag_list:
            temp = V[i-1][s] * transition_mat[t][s]
            if temp > ans:
                ans = temp
                best = s
        return (best, ans)


    def _get_best_tag(self, sent, V, B, tags):
        best_ending = None
        best_max = -1

        for tag in tags:
            if V[len(sent) - 1][tag] > best_max:
                best_max = V[len(sent) - 1][tag]
                best_ending = tag
        seq = [best_ending]
        for i in reversed(range(1, len(sent))):
            prev = B[i][seq[-1]]
            #print( seq[-1])
            seq.append(prev)
        #print(len(seq), len(sent))
        return seq[::-1]


    def viterbi(self, words, emission_vectors, transition_mat, sentence_dict):
        V = defaultdict(lambda: defaultdict(float))
        B = defaultdict(lambda: defaultdict(str))
        tag_list = sentence_dict.keys()

        for t in tag_list:
            V[0][t] = sentence_dict[t] * emission_vectors[t][words[0]]

        for i in range(1, len(words)):
            for t in tag_list:
                pair = self._argmax(V, tag_list, t, i, transition_mat)
                B[i][t] = pair[0]
                #print(words[i])
                V[i][t] = pair[1]*emission_vectors[t][words[i]]

        final_labels = self._get_best_tag(words, V, B, tag_list)
        return final_labels
