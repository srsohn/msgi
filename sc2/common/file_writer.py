"""Class implementation for plotting the results of the experiments."""
import os
import copy
import numpy as np

class SC2FileWriter(object):
    """Handles all the SC2 experiment data.
    """

    def __init__(self, args, dirname=''):
        self._save_mean_dir = os.path.join(dirname, 'mean')
        self._save_data_dir = os.path.join(dirname, 'data')
        self._save_ep_len_dir = os.path.join(dirname, 'ep_len')

        # buffers
        self._mean_scores = list()
        self._scores = list()
        self._ep_lens = list()

    def reset(self):
        '''Empty the buffer.'''
        self._mean_scores = list()
        self._scores = list()
        self._ep_lens = list()

    def store(self, ep: int, mean=None, data=None, ep_len=None):
        '''Store the episode return to the buffer.'''
        if mean is not None:
            self._mean_scores.append([ep, mean, 0., 0.])
        if data is not None:
            self._scores.append(np.concatenate(([ep], data)))
        if ep_len is not None:
            self._ep_lens.append([ep, ep_len, 0., 0.])

    def save(self, mean=None, data=None, ep_len=None):
        '''Dump the buffer in a json file format.'''
        mean_scores = copy.deepcopy(self._mean_scores)
        scores = copy.deepcopy(self._scores)
        ep_lens = copy.deepcopy(self._ep_lens)

        if mean is not None:
            mean_scores = mean
        if data is not None:
            scores = data
        if ep_len is not None:
            ep_lens = ep_len

        # convert to numpy array
        if not isinstance(mean_scores, np.ndarray):
            mean_scores = np.asarray(mean_scores)
        if not isinstance(scores, np.ndarray):
            scores = np.asarray(scores)
        if not isinstance(ep_lens, np.ndarray):
            ep_lens = np.asarray(ep_lens)

        # save mean scores
        mean_dir = self._save_mean_dir + '.txt'
        print("Saving the mean scores @ {}".format(mean_dir))
        np.savetxt(fname=mean_dir, X=mean_scores)

        # save all scores
        if len(scores) > 0:
            score_dir = self._save_data_dir + '.txt'
            print("Saving all scores @ {}".format(score_dir))
            np.savetxt(fname=score_dir, X=scores)

        # save episode lengths
        ep_len_dir = self._save_ep_len_dir + '.txt'
        print("Saving the episode lengths @ {}".format(ep_len_dir))
        np.savetxt(fname=ep_len_dir, X=ep_lens)

