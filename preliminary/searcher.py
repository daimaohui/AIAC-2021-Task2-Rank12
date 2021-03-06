# coding=utf-8
import random
import copy

import numpy as np
# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher
from sklearn import tree  # 0.4
from skopt import gbrt_minimize
def func_tree(x_x,tree):
   return tree.predict(x_x)
class Searcher(AbstractSearcher):
    searcher_name = "RandomSearcher"
    def __init__(self, parameters_config, n_iter, n_suggestion):
        """ Init searcher

        Args:
            parameters_config: parameters configuration, consistent with the definition of parameters_config of EvaluateFunction. dict type:
                    dict key: parameters name, string type
                    dict value: parameters configuration, dict type:
                        "parameter_name": parameter name
                        "parameter_type": parameter type, 1 for double type, and only double type is valid
                        "double_max_value": max value of this parameter
                        "double_min_value": min value of this parameter
                        "double_step": step size
                        "coords": list type, all valid values of this parameter.
                            If the parameter value is not in coords,
                            the closest valid value will be used by the judge program.

                    parameter configuration example, eg:
                    {
                        "p1": {
                            "parameter_name": "p1",
                            "parameter_type": 1
                            "double_max_value": 2.5,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0, 2.5]
                        },
                        "p2": {
                            "parameter_name": "p2",
                            "parameter_type": 1,
                            "double_max_value": 2.0,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0]
                        }
                    }
                    In this example, "2.5" is the upper bound of parameter "p1", and it's also a valid value.

        n_iteration: number of iterations
        n_suggestion: number of suggestions to return
        """
        self.tree = tree.DecisionTreeRegressor(max_depth=10, random_state=53)
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)
    def parse_suggestions_history(self, suggestions_history):
        """ Parse historical suggestions to the form of (x_datas, y_datas), to obtain GP training data

        Args:
            suggestions_history: suggestions history

        Return:
            x_datas: Parameters
            y_datas: Reward of Parameters
        """
        suggestions_history_use = copy.deepcopy(suggestions_history)
        x_datas = [[item[1] for item in sorted(suggestion[0].items(), key=lambda x: x[0])]
                   for suggestion in suggestions_history_use]
        y_datas = [suggestion[1] for suggestion in suggestions_history_use]
        x_datas = np.array(x_datas)
        y_datas = np.array(y_datas)
        return x_datas, y_datas

    def random_sample(self):
        """ Generate a random sample in the form of [value_0, value_1,... ]

        Return:
            sample: a random sample in the form of [value_0, value_1,... ]
        """
        sample = [p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)] for p_name, p_conf
              in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        return sample
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
    def suggest(self, suggestion_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters.

        Args:
            suggestions_history: a list of historical suggestion parameters and rewards, in the form of
                    [[Parameter, Reward], [Parameter, Reward] ... ]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                        Reward: a float type value

                    The parameters and rewards of each iteration are placed in suggestions_history in the order of iteration.
                        len(suggestions_history) = n_suggestion * iteration(current number of iteration)

                    For example:
                        when iteration = 2, n_suggestion = 2, then
                        [[{'p1': 0, 'p2': 0, 'p3': 0}, -222.90621774147272],
                         [{'p1': 0, 'p2': 1, 'p3': 3}, -65.26678723205647],
                         [{'p1': 2, 'p2': 2, 'p3': 2}, 0.0],
                         [{'p1': 0, 'p2': 0, 'p3': 4}, -105.8151893979122]]

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
        next_suggestions = []
        if len(suggestion_history)>n_suggestions*10:#????????????
            x_datas, y_datas = self.parse_suggestions_history(suggestion_history)  # ????????????
            p_names = [p_name for p_name, p_conf in
                       sorted(self.parameters_config.items(), key=lambda x: x[0])]  # ?????????????????????
            self.tree.fit(x_datas,y_datas) #????????????
            _bounds=self.get_bounds()
            for index in range(n_suggestions):
                next_suggest={}
                x_tries = np.array([self.random_sample() for _ in range(int(1000))])
                ys = func_tree(x_x=x_tries, tree=self.tree)
                x_max = x_tries[ys.argmax()]
                max_acq = ys.max()
                for i in range(100):
                    x0=self.random_sample()
                    tt = gbrt_minimize(
                        func=lambda x: -func_tree(x_x=[x], tree=self.tree)[0],
                        dimensions=_bounds, n_calls=10, x0=x0,
                        y0=-func_tree(x_x=[x0], tree=self.tree)[0])
                    if -tt.fun>max_acq:
                        x_max=tt.x
                        max_acq=-tt.fun
                for j in range(len(p_names)):
                    next_suggest[p_names[j]] =float(x_max[j])
                next_suggestions.append(next_suggest)
            return next_suggestions
        else: #???????????????
            for i in range(n_suggestions):
                next_suggest = {
                    p_name: p_conf["coords"][random.randint(int(len(p_conf["coords"])/4*0),int(len(p_conf["coords"])-1))]
                    for p_name, p_conf in self.parameters_config.items()
                }
                next_suggestions.append(next_suggest)
        return next_suggestions
