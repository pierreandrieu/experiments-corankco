from corankco.dataset import DatasetSelector
from corankco.partitioning.parconsPartition import ParConsPartition
from corankco.algorithms.median_ranking import MedianRanking
from corankco.scoringscheme import ScoringScheme
from experiments.experiment import ExperimentFromDataset
from typing import List, Tuple, Dict
import numpy as np


class BenchTime(ExperimentFromDataset):
    """

    Class for experiment 1 of IJAR paper.
    Objective: compare the time execution of different versions of the exact algorithm.

    """

    def __init__(self,
                 dataset_folder: str,
                 algs: List[MedianRanking],
                 scoring_scheme: ScoringScheme,
                 dataset_selector_exp: DatasetSelector = None,
                 steps: int = 5,
                 max_time: float = float('inf'),
                 repeat_time_computation_until: float = 1.):
        """

        :param dataset_folder: the folder of the BiologicalDataset
        :param algs: the list of algorithms to compare
        :param scoring_scheme: the scoring scheme to use for computation
        :param dataset_selector_exp: a DatasetSelector object to filter datasets
        :param steps: range of size of datasets for the output. For instance, if 10, the mean time computation is
        computed for dataset elements whose size is 30-39, 40-49, and so on
        :param max_time: max time allowed before the algorithm is killed and will not be re-used
        :param repeat_time_computation_until: for small datasets, the computation of the consensus is done as many times
        as needed to reach a time computation of this parameter. Then, the average time is used
        """

        super().__init__(dataset_folder, dataset_selector_exp)

        # the algorithms to compare
        self.__algs: List[MedianRanking] = algs
        # the scoring scheme to use
        self.__scoring_scheme = scoring_scheme
        # the steps (see docstring if __init__)
        self.__steps = steps
        # the max time allowed for an algorithm
        self.__max_time = max_time
        # see docstring
        self.__repeat_time_computation_until = repeat_time_computation_until

    def _run_raw_data(self, path_to_store_results: str = None) -> str:
        """

        :param path_to_store_results: the file to write each line of the raw result of the experiment
        a raw result is a result directly obtained by computation, not yet parsed to get a figure with x = f(y)
        :return: the string containing the raw result to be given to a parsing function. The string returned
        is in CSV format:
        For this experiment: dataset, nb_elements, time of alg_0, time of alg_1, ..., time of ... alg_n

        """
        res = ""
        # first line
        ligne: str = "dataset;nb_elements"
        for alg in self.__algs:
            ligne += ";" + alg.get_full_name()
        ligne += "\n"
        if path_to_store_results is not None:
            self._write_line_in_file(path_to_store_results, ligne)
        res += ligne

        # if an algorithm has exceeded the max time allowed, it will not run anymore
        must_run = [True] * len(self.__algs)

        # for each dataset
        for dataset in sorted(self._datasets, key=lambda dataset_obj: dataset_obj.nb_elements):
            ligne: str = ""
            ligne += dataset.name + ";" + str(dataset.nb_elements)
            id_alg = 0
            # write the time computation of each algorithm to compare
            # if an algorithm has been killed, it returns float("inf")
            for alg in self.__algs:
                if must_run[id_alg]:
                    time_computation = alg.bench_time_consensus(dataset,
                                                                self.__scoring_scheme,
                                                                True,
                                                                self.__repeat_time_computation_until)
                    ligne += ";" + str(time_computation)
                    if time_computation > self.__max_time:
                        must_run[id_alg] = False
                    id_alg += 1
                else:
                    ligne += ";" + str(float("inf"))
            ligne += "\n"
            if path_to_store_results is not None:
                self._write_line_in_file(path_to_store_results, ligne)
            res += ligne
        return res

    def _run_final_data(self, raw_data: str) -> str:
        """

        :param raw_data: the initial data, not yet parsed to be in a figure
        :return: a string in csv format as in the figures of the article:
        range_size_datasets;mean_time_alg_1;mean_time_alg_2, ..., mean_time_alg_n
        """
        res = "size_datasets"
        for alg in self.__algs:
            res += ";"+alg.get_full_name() + "_mean_time"
        res += "\n"

        # dict containing all the important piece of information
        # each unique id of range of elements to consider, associated with a dict where keys are the algorithm
        # and the values the time computation of the latter algorithm for each dataset whose nb of elements is
        # in the considered range
        h_res: Dict[int, Dict[MedianRanking, List[float]]] = {}

        # given an int as key that is a nb of elements for a dataset, value = the id of the tuple of nb of elements
        # associated. For instance, mapping[42] = 1 if ranges = [(30, 39), (40, 49), ]
        mapping_nb_elements_group: Dict[int, int] = {}
        cpt: int = 0

        # unique id of ranges of elements to consider
        key_mapping: int = 0
        # the dict algorithm -> list of result time computation for the datasets whose nb of elements
        # are in the considered range
        h_res[0]: Dict[MedianRanking, List[float]] = {}

        for alg in self.__algs:
            h_res[0][alg]: List[float] = []

        # the tuples of nb of elements to consider. For instance: (30, 39), (40, 49), ...
        tuples_groups: List[Tuple[int, int]] = []


        for i in range(self._dataset_selector.nb_elem_min, self._dataset_selector.nb_elem_max+1):
            cpt += 1
            mapping_nb_elements_group[i] = key_mapping

            # initialize each list of float with an empty list
            if cpt == self.__steps:
                key_mapping += 1
                cpt = 0
                h_res[key_mapping]: Dict[MedianRanking, List[float]] = {}
                for alg in self.__algs:
                    h_res[key_mapping][alg]: List[float] = []

        # create the tuples of range of nb of elements to consider
        for i in range(self._dataset_selector.nb_elem_min, self._dataset_selector.nb_elem_max+1, self.__steps):
            tuples_groups.append((i, i+self.__steps-1))

        # now, parse the raw data to fill the main dict variable

        # for each nonempty line without the first one
        for line in raw_data.split("\n")[1:]:
            if len(line) > 1:
                # split the line into list of cols
                cols = line.split(";")
                nb_elements = int(cols[1])
                # the col of the first result of an algorithm
                col_first_alg = len(cols) - len(self.__algs)
                # for each result of algorithm
                for j in range(col_first_alg, len(cols)):
                    # h_res[mapping_nb_elements_group] returns the unique int id of the group-range
                    # for instance: h_res[31] = 0 if (30, 39) is the first group
                    h_res[mapping_nb_elements_group[nb_elements]][self.__algs[j-col_first_alg]].append(float(cols[j]))

        # now, use the dict to generate the final parsed csv data:
        # tuple of range of elements;mean_res_alg_1, ..., mean_res_alg_n
        for i in range(len(tuples_groups)):
            tuple_i = tuples_groups[i]
            res += str(tuple_i)
            for alg in self.__algs:
                res += ";" + str(np.mean(np.asarray(h_res[i][alg])))
            res += "\n"
        return res

#######################################################################################################################


class BenchScalabilityScoringScheme(ExperimentFromDataset):
    """

    Class for the experiment 2 of the IJAR article.
    Objective: test the scalability of our most optimized exact algorithm using the BiologicalDataset

    """
    def __init__(self,
                 dataset_folder: str,
                 alg: MedianRanking,
                 scoring_schemes: List[ScoringScheme],
                 dataset_selector_exp: DatasetSelector = None,
                 steps: int = 5,
                 max_time: float = float('inf'),
                 repeat_time_computation_until: float = 1.):
        """

        :param dataset_folder: the path of the BiologicalDataset
        :param alg: the algorithm to evaluate
        :param scoring_schemes: the scoring schemes of the experiment
        :param dataset_selector_exp: to filter the datasets
        :param steps: range of size of datasets for the output. For instance, if 10, the mean time computation is
        computed for dataset elements whose size is 30-39, 40-49, and so on
        :param max_time: max time allowed before the algorithm is killed and will not be re-used
        :param repeat_time_computation_until: for small datasets, the computation of the consensus is done as many times
        as needed to reach a time computation of this parameter. Then, the average time is used
        """

        super().__init__(dataset_folder, dataset_selector_exp)
        # attributes of the class
        self.__alg = alg
        self.__scoring_schemes = scoring_schemes
        self.__steps = steps
        self.__max_time = max_time
        self.__repeat_time_computation_until = repeat_time_computation_until

    def _run_raw_data(self, path_to_store_results: str = None) -> str:
        """

        :param path_to_store_results: the file to write each line of the raw result of the experiment
        a raw result is a result directly obtained by computation, not yet parsed to get a figure with x = f(y)
        :return: the string containing the raw result to be given to a parsing function. The string returned
        is in CSV format:
        For this experiment: dataset;nb_elements;time computation(TC) with scoring_scheme_1;...,;TC with SC_n

        """
        # first line
        ligne: str = "dataset;nb_elements"
        res: str = ""
        for scoring_scheme in self.__scoring_schemes:
            ligne += ";" + scoring_scheme.get_nickname()
        ligne += "\n"
        res += ligne

        if path_to_store_results is not None:
            self._write_line_in_file(path_to_store_results, ligne)

        # the algorithm is killed towards a scoring scheme if for a dataset, the time execution exceeded the associated
        # attribute
        flag = [True] * len(self.__scoring_schemes)
        # foe each considered dataset in increasing order of nb of elements
        for dataset in sorted(self._datasets, key=lambda dataset_obj: dataset_obj.nb_elements):
            ligne: str = dataset.name + ";" + str(dataset.nb_elements)
            id_scoring_scheme = 0
            # for each scoring scheme
            for scoring_scheme in self.__scoring_schemes:
                if flag[id_scoring_scheme]:
                    # get the time computation
                    time_computation = self.__alg.bench_time_consensus(dataset, scoring_scheme, True)
                    if time_computation > self.__max_time:
                        flag[id_scoring_scheme] = False
                else:
                    time_computation = float('inf')
                ligne += ";" + str(time_computation)
                id_scoring_scheme += 1
            ligne += "\n"
            res += ligne
            if path_to_store_results:
                self._write_line_in_file(path_to_store_results, ligne)
        return res

    def _run_final_data(self, raw_data: str) -> str:
        """

        :param raw_data: the initial data, not yet parsed to be in a figure
        :return: a string in csv format as in the figures of the article:
        range_size_datasets;mean_time_alg_sc1;mean_time_alg_sc2, ..., mean_time_alg_scn

        """
        # first line
        res = "size_datasets"
        for sc in self.__scoring_schemes:
            res += ";"+sc.get_nickname() + "_mean_time"
        res += "\n"

        # key: unique int id of the range of elements to consider.
        # value: a dict where key = scoring scheme, value = all the time computation of the datasets whose nb of
        # elements is in the considered range
        h_res: Dict[int, Dict[ScoringScheme, List[float]]] = {}

        mapping_nb_elements_group: Dict[int, int] = {}
        # as above, see comments
        cpt: int = 0
        key_mapping: int = 0
        h_res[0]: Dict[ScoringScheme, List[float]] = {}

        # initialize each list of float with an empty list
        for sc in self.__scoring_schemes:
            h_res[0][sc]: List[float] = []

        # the ranges of nb of elements to consider
        tuples_groups: List[Tuple[int, int]] = []
        # stores for each int the id of the associated range of elements
        for i in range(self._dataset_selector.nb_elem_min, self._dataset_selector.nb_elem_max+1):
            cpt += 1
            mapping_nb_elements_group[i] = key_mapping

            if cpt == self.__steps:
                key_mapping += 1
                cpt: int = 0
                h_res[key_mapping]: Dict[ScoringScheme, List[float]] = {}
                for sc in self.__scoring_schemes:
                    h_res[key_mapping][sc]: List[float] = []
        # fill the tuples
        for i in range(self._dataset_selector.nb_elem_min, self._dataset_selector.nb_elem_max+1, self.__steps):
            tuples_groups.append((i, i+self.__steps-1))
        for line in raw_data.split("\n")[1:]:
            if len(line) > 1:
                cols: List[str] = line.split(";")
                nb_elements: int = int(cols[1])
                col_first_sc: int = len(cols) - len(self.__scoring_schemes)
                for j in range(col_first_sc, len(cols)):
                    h_res[mapping_nb_elements_group[nb_elements]][self.__scoring_schemes[j-col_first_sc]].append(float(cols[j]))

        # for each tuple to consider
        for i in range(len(tuples_groups)):
            tuple_i: Tuple[int, int] = tuples_groups[i]
            res += str(tuple_i)
            # compute the mean time for the algorithm, with the associated scoring scheme
            for sc in self.__scoring_schemes:
                res += ";" + str(np.mean(np.asarray(h_res[i][sc])))
            res += "\n"
        return res
#######################################################################################################################


class BenchPartitioningScoringScheme(ExperimentFromDataset):
    """

    Class for experiments 3 and 4 of IJAR paper. The objective is to evaluate the efficiency of the ParCons partitioning
    according to different values of scoring schemes

    """
    def __init__(self,
                 dataset_folder: str,
                 scoring_schemes_exp: List[ScoringScheme],
                 changing_coeff: Tuple[int, int],
                 intervals: List[Tuple[int, int]] = None,
                 dataset_selector_exp: DatasetSelector = None,
                 ):
        """

        :param dataset_folder: the path of the BiologicalDataset
        :param scoring_schemes_exp: the scoring schemes to compare
        :param changing_coeff: (num of row, num of col) of the penalty to be changed in the experiment
        for instance, if b3 is changed, then num = 0 and col = 2 as scoringscheme[0][2] = b3
        :param intervals: the intervals to consider
        :param dataset_selector_exp: to filter the datasets
        """
        super().__init__(dataset_folder, dataset_selector_exp)
        self.__scoring_schemes = scoring_schemes_exp
        self.__changing_coeff = changing_coeff

        # get the ranges of nb of elements to consider
        if intervals is not None:
            self.__intervals = intervals
        else:
            sorted_datasets = sorted(self._datasets, key=lambda dataset_obj: dataset_obj.nb_elements)
            self.__intervals = [(sorted_datasets[0].nb_elements, sorted_datasets[-1].nb_elements)]

    def _run_raw_data(self, path_to_store_results: str = None) -> str:
        res = ""
        ligne = "dataset;nb_elements;"
        for scoring_scheme in self.__scoring_schemes:
            ligne += str(scoring_scheme.penalty_vectors) + ";"
        ligne += "\n"
        res += ligne
        if path_to_store_results is not None:
            self._write_line_in_file(path_to_store_results, ligne)
        # for dataset in sorted(self.datasets):
        for dataset in sorted(self._datasets, key=lambda dataset_obj: dataset_obj.nb_elements):
            ligne: str = dataset.name + ";" + str(dataset.nb_elements) + ";"
            for scoring_scheme in self.__scoring_schemes:
                # computes the size of the biggest sub-problem obtained
                ligne += str(ParConsPartition.size_of_biggest_subproblem(dataset, scoring_scheme)) + ";"
            ligne += "\n"
            res += ligne
            if path_to_store_results is not None:
                self._write_line_in_file(path_to_store_results, ligne)
        return res

    def _run_final_data(self, raw_data: str) -> str:
        """

         :param raw_data: the initial data, not yet parsed to be in a figure
         :return: a string in csv format as in the figures of the article:
         in this experiment: range_size_datasets;mean_time_alg_sc1;mean_time_alg_sc2, ..., mean_time_alg_scn

         """
        mapping_int_interval: Dict[int, Tuple[int, int]] = {}
        for interval in self.__intervals:
            for i in range(interval[0], interval[1]+1):
                mapping_int_interval[i] = interval
        res = ""
        for scoring_scheme in self.__scoring_schemes:
            value_coeff = scoring_scheme.penalty_vectors[self.__changing_coeff[0]][self.__changing_coeff[1]]
            res += ";" + str(value_coeff)
        res += "\n"
        nb_scoring_schemes = len(self.__scoring_schemes)
        h_res = {}
        for interval in self.__intervals:
            h_res[interval] = {}
            for scoring_scheme in self.__scoring_schemes:
                h_res[interval][scoring_scheme] = []

        for line in raw_data.split("\n")[1:]:
            if len(line) > 1:
                cols = line.split(";")
                id_scoring_scheme = 0
                nb_elem = int(cols[1])
        for interval in self.__intervals:
            for i in range(interval[0], interval[1]+1):
                mapping_int_interval[i] = interval
        res = "_"
        for scoring_scheme in self.__scoring_schemes:
            value_coeff = scoring_scheme.penalty_vectors[self.__changing_coeff[0]][self.__changing_coeff[1]]
            res += ";" + str(value_coeff)
        res += "\n"
        nb_scoring_schemes = len(self.__scoring_schemes)
        h_res = {}
        for interval in self.__intervals:
            h_res[interval] = {}
            for scoring_scheme in self.__scoring_schemes:
                h_res[interval][scoring_scheme] = []

        for line in raw_data.split("\n")[1:]:
            if len(line) > 1:
                cols = line.split(";")
                id_scoring_scheme = 0
                nb_elem = int(cols[1])
                for i in range(len(cols)-nb_scoring_schemes-1, len(cols)-1):
                    target_value = float(cols[i])
                    h_res[mapping_int_interval[nb_elem]][self.__scoring_schemes[id_scoring_scheme]].append(target_value)
                    id_scoring_scheme += 1

        for interval in self.__intervals:
            res += str(interval)
            for scoring_scheme in self.__scoring_schemes:
                res += ";" + str(round(float(np.mean(np.asarray(h_res[interval][scoring_scheme]))), 2))
            res += "\n"
        return res
