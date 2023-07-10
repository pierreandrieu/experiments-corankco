from experiments.experiment import Experiment
from corankco.dataset import Dataset
from corankco.algorithms.algorithmChoice import get_algorithm, Algorithm
from corankco.scoringscheme import ScoringScheme
from corankco.algorithms.median_ranking import MedianRanking
from corankco.ranking import Ranking
from corankco.element import Element
import random
from itertools import groupby
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Dict


class SchoolYear:
    """

    Class to represent a school year for the Student Experiment of IJAR paper

    """
    def __init__(self,
                 nb_students_track1: int,
                 nb_students_track2: int,
                 nb_classes_total: int,
                 nb_classes_track1: int,
                 nb_classes_track2: int,
                 mean_track1: float,
                 variance_track1: float,
                 mean_track2: float,
                 variance_track2: float,
                 topk: int):
        """

        class related to "Student experiment" of IJAR paper. This class represent a "one year school"
        :param nb_students_track1: the number of students in track 1
        :param nb_students_track2: the number of students in track 2
        :param nb_classes_total: the number of classes
        :param nb_classes_track1: the number of classes to be involved in for students of track 1
        :param nb_classes_track2: the number of classes to be involved in for students of track 2
        :param mean_track1: the mean the normal law for the generation of marks for the students of track 1
        :param variance_track1: the variance of the normal law for the generation of marks for the students of track 1
        :param mean_track2: the mean of the normal law for the generation of marks for the students of track 2
        :param variance_track2: the variance of the normal law for the generation of marks for the students of track 2
        :param topk: the top-k to compare
        """

        # initialize the marks of students as a matrix ndarray, -1 initially everywhere
        # marks[i][j] = mark of student i at the exam of class j, -1 if not involved in the class
        self._marks: np.ndarray = np.zeros(
            (nb_students_track1 + nb_students_track2, nb_classes_total), dtype=float) - 1
        # total number of students
        nb_total_students: int = nb_students_track1 + nb_students_track2
        # the classes as a list of ints
        list_classes: List[int] = list(range(nb_classes_total))
        # a list of (average mark for student i, i)
        self._students_total_average: List[Tuple[float, int]] = []

        # the first i rows are the marks of the students of track 1
        for i in range(nb_students_track1):
            student: np.ndarray = self._marks[i]
            # get uniformly the classes in which student i is involved
            classes_chosen: List[int] = random.sample(list_classes, nb_classes_track1)
            # generate the marks according to the normal law
            marks = np.round(np.random.randn(nb_classes_track1) * variance_track1 + mean_track1, 1)
            # french system: notes are between 0 and 20
            marks[marks > 20.] = 20
            marks[marks < 0] = 0
            # store the marks of student i in the matrix of marks. The marks remain -1 for the classes such that
            # student i is not involved
            student[classes_chosen] = marks
            # append the mean mark of student i as a tuple with i (i = unique int ID of student)
            self._students_total_average.append((np.round(np.mean(marks), 1), i))

        # same procedure for students of track 2, whose index start at nb_students_track1
        for i in range(nb_students_track1, nb_total_students):
            student: np.ndarray = self._marks[i]
            classes_chosen: List[int] = random.sample(list_classes, nb_classes_track2)
            marks: np.ndarray = np.round(np.random.randn(nb_classes_track2) * variance_track2 + mean_track2, 1)
            marks[marks > 20.] = 20
            marks[marks < 0] = 0
            student[classes_chosen] = marks
            self._students_total_average.append((np.round(np.mean(marks), 1), i))

        # sort students by average mark
        self._students_total_average = sorted(self._students_total_average, reverse=True)

        # stores the topk students according to their average mark
        self._goldstandard: Set[int] = set()
        for i in range(topk):
            self._goldstandard.add(self._students_total_average[i][1])

    @property
    def marks(self) -> np.ndarray:
        """

        :return: the 2D matrix of marks of the Student Experiment. marks[i][j] = mark of student i for class j, -1 if
        student i is not involved in class j

        """
        return self._marks

    @property
    def goldstandard(self) -> Set[int]:
        """

        :return: a Set of the int ID of the top-k students. Value of k was defined in __init__

        """
        return self._goldstandard

    @staticmethod
    def school_year_to_dataset(school_year: 'SchoolYear') -> Dataset:
        """

        :param school_year: the SchoolYear object to consuder
        :return: a Dataset containing one ranking by class, ranking i is the ranking of the students involved in the
        i-th class, students that are not involved in i-th class are non-ranked in ranking i


        """
        rankings: List[Ranking] = []
        marks: np.ndarray = school_year.marks
        nb_students, nb_classes = marks.shape
        # for each class, create the rankings of the involved students according to their mark
        for i in range(nb_classes):
            marks_class: np.ndarray = marks[:, i]
            # students not involved (mark == -1) are not stored, they will be non-ranked
            marks_students: List[Tuple[float, int]] = \
                [(marks_class[j], j) for j in range(nb_students) if marks_class[j] >= 0]
            # sort the tuples by decreasing order or marks
            marks_students: List[Tuple[float, int]] = sorted(marks_students, reverse=True)

            # create the rankings of the students according to their mark. If same mark, same bucket
            ranking_students: List[Set[Element]] = [set(Element(x[1]) for x in group) for _, group in
                                                    groupby(marks_students, key=itemgetter(0))]
            rankings.append(Ranking(ranking_students))
        return Dataset(rankings)


class MarksExperiment(Experiment):
    """

    class to execute the student experiment on IJAR paper. This class heritates from Experiment to get standard useful
    methods in the context of experiments.

    """

    def __init__(self,
                 nb_years: int,
                 nb_students_track1: int,
                 nb_students_track2: int,
                 nb_classes_total: int,
                 nb_classes_track1: int,
                 nb_classes_track2: int,
                 mean_track1: float,
                 variance_track1: float,
                 mean_track2: float,
                 variance_track2: float,
                 topk: int,
                 scoring_schemes: List[ScoringScheme],
                 algo: MedianRanking = get_algorithm(Algorithm.ParCons, parameters={
                     "bound_for_exact": 150, "auxiliary_algorithm": get_algorithm(alg=Algorithm.BioConsert)}),
                 ):
        """

        :param nb_years: the number of years to consider in the experiment.
        :param nb_students_track1: the number of students in track 1 (for each year)
        :param nb_students_track2: the number of students in track 2 (for each year)
        :param nb_classes_total: the number of classes (for each year)
        :param nb_classes_track1: the number of classes to be involved in for students of track 1 (for each year)
        :param nb_classes_track2: the number of classes to be involved in for students of track 2 (for each year)
        :param mean_track1: the mean the normal law for the generation of marks for the students of track 1 (for each
        year)
        :param variance_track1: the variance of the normal law for the generation of marks for the students of track 1
        (for each year)
        :param mean_track2: the mean of the normal law for the generation of marks for the students of track 2 (for each
        year)
        :param variance_track2: the variance of the normal law for the generation of marks for the students of track 2
        (for each year)
        :param topk: the top-k to compare (for each year)
        :param scoring_schemes: the list scoring_schemes to consider (to compare the obtained consensus) in the
        experiment
        :param algo: the algorithm to compute consensus rankings, one by year
        """
        super().__init__()
        # the attributes of the experiment object
        self.__alg: MedianRanking = algo
        self.__scoring_schemes: List[ScoringScheme] = scoring_schemes
        self.__nb_years: int = nb_years
        self.__nb_students_track_1: int = nb_students_track1
        self.__nb_students_track_2: int = nb_students_track2
        self.__nb_classes_total: int = nb_classes_total
        self.__nb_classes_track_1: int = nb_classes_track1
        self.__nb_classes_track_2: int = nb_classes_track2
        self.__mean_track1: float = mean_track1
        self.__variance_track1: float = variance_track1
        self.__mean_track2: float = mean_track2
        self.__variance_track2: float = variance_track2
        self.__topk: int = topk

    def _run_raw_data(self, path_to_store_results: str = None) -> str:
        """

        :param path_to_store_results: the file to write each line of the raw result of the experiment
        a raw result is a result directly obtained by computation, not yet parsed to get a figure with a x = f(y)
        :return: the string containing the raw result to be given to a parsing function

        """
        # first line
        line: str = "year;b5-b4;nb_students_both_topkconsensus_topkgoldstandard\n"
        if path_to_store_results is not None:
            self._write_line_in_file(path_to_store_results, line)
        # res stores the full raw data
        res: str = line
        # create one SchoolYear by year
        for i in range(self.__nb_years):
            school_year = SchoolYear(self.__nb_students_track_1,
                                     self.__nb_students_track_2,
                                     self.__nb_classes_total,
                                     self.__nb_classes_track_1,
                                     self.__nb_classes_track_2,
                                     self.__mean_track1,
                                     self.__variance_track1,
                                     self.__mean_track2,
                                     self.__variance_track2,
                                     self.__topk)
            # create the Dataset for the year that is the ranking of the students for each class
            dataset_year: Dataset = SchoolYear.school_year_to_dataset(school_year)

            # for each scoring scheme
            for scoring_scheme in self.__scoring_schemes:
                # compute the consensus
                consensus = self.__alg.compute_consensus_rankings(dataset_year, scoring_scheme, True)
                # get the number of students both in the consensus and top-k according to the average marks
                both_gs_topk: int = \
                    consensus.evaluate_topk_ranking(goldstandard=school_year.goldstandard, top_k=self.__topk)
                # get useful information
                line: str = str(i) + ";" + str(scoring_scheme.b5) + ";" + str(both_gs_topk)+"\n"
                if path_to_store_results is not None:

                    self._write_line_in_file(path_to_store_results, line)
                res += line
        return res

    def _run_final_data(self, raw_data: str) -> str:
        """

        :param raw_data: the raw data of the experiment
        :return: the final parsed data that corresponds to the figures in IJAR paper

        """
        # stores for each value of b5 the list of the number of students both in consensus and goldstandard
        h_res: Dict[float, List[float]] = {}
        for scoring_scheme in self.__scoring_schemes:
            h_res[scoring_scheme.b5]: List[float] = []
        # for each line of the raw data,
        for line in raw_data.split("\n")[1:]:
            if len(line) > 1:
                # append to the list associated with b5 the nb of students both in consensus and goldstandard
                _, b5, target = line.split(";")
                h_res[float(b5)].append(float(target))
        # prepare the final data : associate for each b5-b4 value the average nb of students both in consensus and
        # goldstandard
        res = "b5-b4;common_goldstandard_topkconsensus\n"
        for scoring_scheme in self.__scoring_schemes:
            res += str(scoring_scheme.b5) + ";" + str(np.round(np.mean(np.asarray(h_res[scoring_scheme.b5])), 2))+"\n"
        return res

    def _display(self, final_data: str):
        x_axis = []
        y_axis = []
        data_split = final_data.split("\n")
        for line in data_split[1:]:
            if len(line) > 1:
                cols = line.split(";")
                x_axis.append(float(cols[0]))
                y_axis.append(float(cols[1]))
        plt.xlabel("B5-B4")
        plt.ylim(0, 20)
        plt.ylabel("average (  | top-20 consensus ∩ top-20 goldstandard | )")
        plt.scatter(x_axis, y_axis)
        plt.savefig("/home/pierre/Bureau/fig_marks.png")
        plt.show()
