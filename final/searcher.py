import copy
import numpy as np
from openbox import Optimizer,sp
from thpo.abstract_searcher import AbstractSearcher
class Searcher(AbstractSearcher):

    def __init__(self, parameters_config, n_iter, n_suggestion):
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)

    def parse_suggestions_history(self, suggestions_history):
        suggestions_history_use = copy.deepcopy(suggestions_history)
        x_datas = [[item[1] for item in sorted(suggestion[0].items(), key=lambda x: x[0])]
                   for suggestion in suggestions_history_use]
        y_datas = [-suggestion[1] for suggestion in suggestions_history_use]
        x_datas = np.array(x_datas)
        y_datas = np.array(y_datas)
        return x_datas, y_datas

    def get_bounds(self):
        """ Get sorted parameter space

        Return:
            _bounds: The sorted parameter space
        """
        def _get_param_value(param):
            value = [param['double_min_value'], param['double_max_value']]
            return value

        _bounds = np.array(
            [_get_param_value(item[1]) for item in sorted(self.parameters_config.items(), key=lambda x: x[0])],
            dtype=np.float
        )
        return _bounds
    def suggest_old(self, suggestions_history, n_suggestions=1):
        p_names = [p_name for p_name, p_conf in
                   sorted(self.parameters_config.items(), key=lambda x: x[0])]  # 所有的参数名称
        next_suggestions=[]
        _bounds = self.get_bounds()
        space = sp.Space()
        t=[]
        for i in range(len(p_names)):
            x1=sp.Real(p_names[i], _bounds[i][0], _bounds[i][1])
            t.append(x1)
        space.add_variables(t)
        if (suggestions_history is None) or (len(suggestions_history) <= 0):
            opt = Optimizer(space, max_runs=50, task_id='quick_start',surrogate_type='gp',
        acq_optimizer_type='batchmc')
            history = opt.run_init()
            print(history)
            for i in range(len(history)):
                next_suggest = {}
                for j in range(len(p_names)):
                    next_suggest[p_names[j]] = float(history[i][p_names[j]])
                next_suggestions.append(next_suggest)
            print(next_suggestions)
        else:
            x_datas, y_datas = self.parse_suggestions_history(suggestions_history)
            print(suggestions_history)
            print(y_datas)
            #batchmc
            opt1 = Optimizer(space, max_runs=50, task_id='quick_start',surrogate_type='gp',
        acq_optimizer_type='batchmc')
            opt1.add_history(x_datas, y_datas, p_names)
            history = opt1.run(5)
            for i in range(len(history)):
                next_suggest = {}
                for j in range(len(p_names)):
                    next_suggest[p_names[j]] = float(history[i][p_names[j]])
                next_suggestions.append(next_suggest)
        return next_suggestions

    def get_my_score(self, reward):
        """ Get the most trusted reward of all iterations.

        Returns:
            most_trusted_reward: float
        """
        return reward[-1]['value']
    def suggest(self, iteration_number, running_suggestions, suggestion_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters. new implementation of final competition

        Args:
            iteration_number: int ,the iteration number of experiment, range in [1, 140]

            running_suggestions: a list of historical suggestion parameters and rewards, in the form of
                    [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
                Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                    {'p1': 0, 'p2': 0, 'p3': 0}
                Reward: a list of dict, each dict of the list corresponds to an iteration,
                    the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound} 
                    Reward example:
                        [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
                         {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
                        ]

            suggestion_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions

            n_suggestion: int, number of suggestions to return

        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        MIN_TRUSTED_ITERATION = 10
        new_suggestions_history = []
        print(len(suggestion_history))
        print(MIN_TRUSTED_ITERATION)
        for suggestion in suggestion_history:
            iterations_of_suggestion = len(suggestion['reward'])
            if iterations_of_suggestion >= MIN_TRUSTED_ITERATION:
                cur_score = self.get_my_score(suggestion['reward'])
                new_suggestions_history.append([suggestion["parameter"], cur_score])
        return self.suggest_old(new_suggestions_history, n_suggestions)

    def is_early_stop(self, iteration_number, running_suggestions, suggestion_history):
        """ Decide whether to stop the running suggested parameter experiment.

        Args:
            iteration_number: int, the iteration number of experiment, range in [1, 140]

            running_suggestions: a list of historical suggestion parameters and rewards, in the form of
                    [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
                Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                    {'p1': 0, 'p2': 0, 'p3': 0}
                Reward: a list of dict, each dict of the list corresponds to an iteration,
                    the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound} 
                    Reward example:
                        [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
                         {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
                        ]

            suggestions_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions

        Returns:
            stop_list: list of bool, indicate whether to stop the running suggestions.
                    len(stop_list) must be the same as len(running_suggestions), for example:
                        len(running_suggestions) = 3, stop_list could be : 
                            [True, True, True] , which means to stop all the three running suggestions
        """

        # Early Stop algorithm demo 2:
        #
        #   If there are 3 or more suggestions which had more than 7 iterations, 
        #   the worst running suggestions will be stopped
        #
        MIN_ITERS_TO_STOP = 10
        MIN_SUGGUEST_COUNT_TO_STOP = 3
        MAX_ITERS_OF_DATASET = self.n_iteration
        ITERS_TO_GET_STABLE_RESULT = 14
        INITIAL_INDEX = -1

        res = [False] * len(running_suggestions)
        # if iteration_number + ITERS_TO_GET_STABLE_RESULT <= MAX_ITERS_OF_DATASET:
        #         score_min_idx = INITIAL_INDEX
        #         score_min = float("inf")
        #         count = 0
        #         # Get the worst suggestion of current running suggestions
        #         for idx, suggestion in enumerate(running_suggestions):
        #             if len(suggestion['reward']) >= MIN_ITERS_TO_STOP:
        #                 count = count + 1
        #                 cur_low = self.get_my_low(suggestion['reward'])
        #                 if score_min_idx == INITIAL_INDEX or cur_low < score_min:
        #                     score_min_idx = idx
        #                     score_min = cur_low
        #         # Stop the worst suggestion
        #         if count >= MIN_SUGGUEST_COUNT_TO_STOP and score_min_idx != INITIAL_INDEX:
        #             res[score_min_idx] = True
        return res
