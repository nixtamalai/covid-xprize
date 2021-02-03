import pandas as pd
import os
from glob import glob
from pandas.core.frame import DataFrame
from tqdm import tqdm
import numpy as np
from numpy.random import randint
from typing import Union, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import RegressorMixin
from collections import OrderedDict
from copy import deepcopy
from covid_xprize.nixtamalai.helpers import add_geo_id
from covid_xprize.nixtamalai.analyze_predictor import IP_COLS
from microtc.utils import load_model
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask = True):
    """
    Taken from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

    
def prescription_cases(output: Union[str, None] = "presc-cases.csv") -> pd.DataFrame:
    FILES = glob(os.path.join(ROOT_DIR, "..", "..", "prescriptions/*2021-01-28.csv"))
    FILES.sort()
    prescriptions = {os.path.basename(fname).split("-")[0]:
                     pd.read_csv(fname, parse_dates=["Date"], index_col=["Date"])
                     for fname in tqdm(FILES)}
    presc_norm = {k: v - prescriptions['332423242324'] for k, v in prescriptions.items()}
    presc_norm = {k: v.sum() for k, v in presc_norm.items()}
    presc_norm_df = pd.DataFrame(presc_norm).T
    presc_norm_df.replace(0, 1, inplace=True)
    presc_norm_df = np.log(presc_norm_df)
    presc_norm_df.fillna(0, inplace=True)
    if output:
        presc_norm_df.to_csv(output)
    return presc_norm_df


def training_set(df: pd.DataFrame, region: str) -> Tuple[np.ndarray, np.ndarray]:
    data = df.loc[:, region]
    y = data.values
    X = np.array([[int(i) for i in x] for x in data.index])
    return X, y


class SHC(object):
    def __init__(self, regressor: RegressorMixin) -> None:
        self._regressor = regressor
        self._max_value = [int(x) for x in "332423242324"]
        self._scnd_best = None

    @property
    def max_value(self):
        return self._max_value

    @property
    def regressor(self):
        return self._regressor

    @property
    def visited_points(self):
        return self._visited

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SHC":
        self.regressor.fit(X, y)
        hy = self.regressor.predict(X)
        _ = {self.id(x): v for x, v in zip(X, hy)}
        self._visited = _
        # Process
        ele, fit = self.random()
        for _ in tqdm(range(len(self.visited_points), 32768)):
            next = self.next(ele, fit)
            if next is None:
                if self._scnd_best is not None:
                    next = self._scnd_best
                    self._scnd_best = None
                else:
                    next = self.random()
            ele, fit = next
            self.visited_points[self.id(ele)] = fit
        return self

    def next(self, ele: np.ndarray, fit: float) -> Union[None, Tuple[np.ndarray, float]]:
        elements = self.neighbors(ele)
        fits = self.fitness(elements)
        index = np.where(fits < fit)[0]
        if len(index) == 0:
            return None
        np.random.shuffle(index)
        if len(index) >= 2:
            self._scnd_best = elements[index[1]], fits[index[1]]
        return elements[index[0]], fits[index[0]]

    @staticmethod
    def id(element: np.ndarray) -> str:
        return "".join(map(str, element))

    def fitness(self, element: np.ndarray) -> float:
        return self.regressor.predict(np.atleast_2d(element))

    def random(self):
        for _ in range(100):
            ind = [randint(x + 1) for x in self.max_value]
            key = self.id(ind)
            if key not in self.visited_points:
                fit = self.fitness(ind)[0]
                self.visited_points[key] = fit
                return ind, fit
        raise Exception("Could not find any more points")

    def neighbors(self, element: np.ndarray) -> np.ndarray:
        n = len(element)
        res = []
        visited = set(self.id(element))
        for k in range(n):
            new = deepcopy(element)
            lst = list(range(self.max_value[k] + 1))
            np.random.shuffle(lst)
            value = lst[0] if lst[0] != element[k] else lst[1]
            new[k] = value
            key = self.id(new)
            if key in visited or key in self.visited_points:
                continue
            res.append(new)
            visited.add(key)
        return res


class MSHC(SHC):
    def __init__(self, weights: np.ndarray,
                 npis_pf: list,
                 hist: set,
                 **kwargs) -> None:
        super(MSHC, self).__init__(**kwargs)
        self._weights = weights
        self._npis_pf = npis_pf
        self._visited = hist

    @property
    def weights(self):
        return self._weights

    def fitness(self, element: np.ndarray) -> np.ndarray:
        _ = np.atleast_2d(element)
        cases = self.regressor.predict(_)
        cost = (self.weights * _).sum(axis=1)
        return np.vstack([cases, cost]).T

    def fit(self, X, y):
        self.regressor.fit(X, y)
        points = list(self._npis_pf.keys())
        for point in points:
            if point not in self._npis_pf:
                continue
            self.iter(point)
        return self

    def iter(self, point):
        fit = self._npis_pf[point]
        for _ in range(100):
            point = list(map(int, point))
            neighbors = self.neighbors(np.array(point))
            if len(neighbors) == 0:
                return
            fits = self.fitness(neighbors)
            index = is_pareto_efficient(np.vstack([fit, fits]),
                                        return_mask=False)
            if index.shape[0] == 1 and index[0] == 0:
                return
            elif index.shape[0] > 1:
                index = index[1:]
                np.random.shuffle(index)
            for i in index:
                key = "".join(map(str, neighbors[i-1]))
                self._npis_pf[key] = fits[i-1].tolist()
            _ = is_pareto_efficient(np.array(list(self._npis_pf.values())))
            keys = list(self._npis_pf.keys())
            for k, flag in zip(keys, _):
                if not flag:
                    del self._npis_pf[k]
            point = neighbors[index[0] - 1]
            point = "".join(map(str, point))
            fit = fits[index[0] - 1]


def run(index):
    from microtc.utils import save_model
    df = prescription_cases()
    cols = list(df.columns)
    cols.sort()
    country = cols[index]
    X, y = training_set(df, country)
    shc = SHC(RandomForestRegressor())
    try:
        shc.fit(X, y)
    except ValueError:
        print(country, "*******")
        return
    save_model(shc.visited_points,
               os.path.join(ROOT_DIR, "prescriptions", str(index) + ".pickle.gz"))


def _policy(args):
    weights, country, country_id, X, y = args
    w = weights.loc[weights.GeoID == country, IP_COLS].values
    # GeoID = weights.GeoID.values.copy()
    # GeoID.sort()
    # regions_id = {v: k for k, v in enumerate(GeoID)}
    prescriptions_path = os.path.join(ROOT_DIR,
                                      "2021-01-28-prescriptions/%s.pickle.gz" % country_id)
    presc = load_model(prescriptions_path)
    cost = {k: [v, (np.array([int(i) for i in k]) * w).sum()] for k, v in presc.items()}
    npis = list(cost.keys())
    npis.sort(key=lambda x: cost[x][0])
    _ = np.array([cost[k] for k in npis])
    index = is_pareto_efficient(_, return_mask=False)
    _ = OrderedDict()
    for x in index:
        key = npis[x]
        _[key] = cost[key]
    mshc = MSHC(w, _, set(npis),
            regressor=RandomForestRegressor()).fit(X, y)
    ss = list(mshc._npis_pf.items())
    ss.sort(key=lambda x: x[1][0])
    if len(ss) > 10:
        ind2 = np.linspace(1, len(ss) -2, 10).round().astype(np.int)
        npis = [ss[i][0] for i in ind2]
    else:
        npis = [x[0] for x in ss]
    return [country, npis]


def policy(weights):
    from multiprocessing import Pool, cpu_count
    add_geo_id(weights)
    w_regions = set(weights.GeoID)
    df = prescription_cases()
    regions = df.columns.sort_values()
    args = [training_set(df, region) for region in regions]
    args = [(weights, country, country_id, X, y)
            for country, (X, y), country_id in zip(regions, args,
                                                   range(len(regions))) if country in w_regions]    
    res = dict()
    with Pool(cpu_count()) as pool:
        for k, v in tqdm(pool.imap_unordered(_policy, args), total=len(args)):
            res[k] = v
    return res


if __name__ == "__main__" and False:
    from multiprocessing import Pool, cpu_count
    df = prescription_cases()
    cols = list(df.columns)
    cols.sort()
    with Pool(processes=cpu_count()) as pool:
        [_ for _ in tqdm(pool.imap_unordered(run, range(len(cols))),
                         total=len(cols))]
