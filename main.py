from experiments.exp_1_2_3_4_bench import BenchPartitioningScoringScheme, BenchTime, BenchScalabilityScoringScheme
from experiments.exp_5_orphanet import ExperimentOrphanet
from experiments.exp_6_students import MarksExperiment
import corankco as crc
from corankco.algorithms.exact.exactalgorithmcplexforpaperoptim1 import ExactAlgorithmCplexForPaperOptim1
from typing import Set, List, Dict, Callable
import random
import numpy as np
import sys
import argparse
from pathlib import Path


def run_bench_time_alg_exacts_ijar(path_dataset: str, raw_data=False, figures=False):
    # get the scoring schemes (the KCFs)
    print("Run bench time computation of EA, EA-optim1, EA-optim1-optim2")
    print("Estimated time: 36h on Intel Core i7-7820HQ CPU 2.9 GHz * 8")
    # kcf1 = crc.ScoringScheme.get_unifying_scoring_scheme_p(1.)
    # kcf2 = crc.ScoringScheme.get_extended_measure_scoring_scheme()
    kcf3 = crc.ScoringScheme.get_induced_measure_scoring_scheme_p(1.)
    kcf4 = crc.ScoringScheme.get_pseudodistance_scoring_scheme_p(1.)
    kcfs: List[crc.ScoringScheme] = [kcf3, kcf4]
    # not optimized exact algorithm
    ea: crc.RankAggAlgorithm = crc.ExactAlgorithm(optimize=False)
    # exact algorithm with optim1
    ea_optim1: crc.RankAggAlgorithm = ExactAlgorithmCplexForPaperOptim1()

    # exact algorithm with optim 1 and 2
    ea_optim1_optim2: crc.RankAggAlgorithm = crc.ExactAlgorithm(optimize=True)

    algorithms_for_bench: List[crc.RankAggAlgorithm] = [
        ea, ea_optim1, ea_optim1_optim2
    ]
    # run experiment for each scoring scheme (KCF)
    for kcf in kcfs:
        # file_output: str = join(folder_output, "exp1_" + kcf.get_nickname() + "_" + datetime.now().strftime(
        #    "time=%m-%d-%Y_%H-%M-%S") + ".csv")
        bench = BenchTime(
            dataset_folder=path_dataset,
            # algorithms for the bench time
            algs=algorithms_for_bench,
            # the scoring scheme that is the kcf to consider
            scoring_scheme=kcf,
            # to select tuples of rankings with number of elements between 30 and 119 and at least 3 rankings
            dataset_selector_exp=crc.DatasetSelector(nb_elem_min=30, nb_elem_max=49, nb_rankings_min=3),
            # range of size of datasets for the output
            steps=10,
            # re-compute the consensus until final time computation > 1 sec.
            # the average time computation is then returned
            repeat_time_computation_until=0.)

        # run experiment and print results. If parameter is true: also print all parameters of experiment (readme)
        # and the raw data that was used to compute the final data. If parameter is false, only final data is displayed
        bench.run(raw_data, figures=figures)


# runs experiment 2 in research paper
def run_bench_exact_optimized_scoring_scheme_ijar(path_dataset: str, raw_data=False, figures=False):
    print("Run experiment scalability of exact algorithm.")
    print("Estimated time: 24h on Intel Core i7-7820HQ CPU 2.9 GHz * 8")
    # file_output: str = join(folder_output, "exp2_" + datetime.now().strftime("time=%m-%d-%Y_%H-%M-%S") + ".csv")
    # get the scoring schemes to consider in the experiment (the KCFs)
    # kcf1: crc.ScoringScheme = crc.ScoringScheme.get_unifying_scoring_scheme_p(1.)
    # kcf2: crc.ScoringScheme = crc.ScoringScheme.get_extended_measure_scoring_scheme()
    kcf3: crc.ScoringScheme = crc.ScoringScheme.get_induced_measure_scoring_scheme_p(1.)
    kcf4: crc.ScoringScheme = crc.ScoringScheme.get_pseudodistance_scoring_scheme_p(1.)
    kcfs: List[crc.ScoringScheme] = [kcf3, kcf4]

    # use the fastest exact algorithm to test the scalability
    ea_optim1_optim2: crc.RankAggAlgorithm = crc.ExactAlgorithm(optimize=True)
    # run experiment for each scoring scheme (KCF)
    bench: BenchScalabilityScoringScheme = BenchScalabilityScoringScheme(
        dataset_folder=path_dataset,
        # the algorithm to consider
        alg=ea_optim1_optim2,
        # the kcfs to consider
        scoring_schemes=kcfs,
        # the dataset selector for selection according to the size
        dataset_selector_exp=crc.DatasetSelector(nb_elem_min=30, nb_elem_max=59, nb_rankings_min=3),
        # range of size of datasets for the output
        steps=10,
        # max time computation allowed. for each kcf, the computation stops
        # when for a tuple of rankings the time computation is higher
        max_time=600,
        # re-compute the consensus until final time computation > 1 sec. The average time computation is then returned
        repeat_time_computation_until=0.)

    # run experiment and print results. If parameter is true: also print all parameters of experiment (readme)
    # and the raw data that was used to compute the final data. If parameter is false, only final data is displayed
    bench.run(raw_data, figures)


# runs experiment 3 in research paper
def run_count_subproblems_t_ijar(path_dataset: str, raw_data=False):
    print("Run experiment subproblems t.")
    print("Estimated time: ~ 90 min on Intel Core i7-7820HQ CPU 2.9 GHz * 8")
    # file_output: str = join(folder_output, "exp3_" + datetime.now().strftime("time=%m-%d-%Y_%H-%M-%S") + ".csv")

    # the list of KCFs to consider in the experiment
    kcfs: List[crc.ScoringScheme] = []
    # vector of different values to test in the experiment for t1, t2, t4, t5
    penalties_t: List[float] = [0.0, 0.25, 0.5, 0.75, 1.]

    for penalty in penalties_t:
        # vector b is constant, t varies
        scoringscheme: crc.ScoringScheme = crc.ScoringScheme(
            [[0., 1., 1., 0., 1., 0], [penalty, penalty, 0., penalty, penalty, penalty]])
        kcfs.append(scoringscheme)

    bench: BenchPartitioningScoringScheme = BenchPartitioningScoringScheme(
        dataset_folder=path_dataset,
        # the kcfs to consider
        scoring_schemes_exp=kcfs,
        # all the files (tuples of rankings) are considered
        dataset_selector_exp=crc.DatasetSelector(),
        # = T[1], for printing changing value of T
        # (num of row, num of col) of the penalty to be changed in the experiment
        # for instance, if b3 is changed, then num = 0 and col = 2 as scoringscheme[0][2] = b3
        changing_coeff=(1, 0),
        # range of number of elements to consider for the output
        intervals=[(30, 59), (60, 99), (100, 299), (300, 1121)])

    # run experiment and print results. If parameter is true: also print all parameters of experiment (readme)
    # and the raw data that was used to compute the final data. If parameter is false, only final data is displayed
    bench.run(raw_data)


# runs experiment 4 in research paper
def run_count_subproblems_b6_ijar(path_dataset: str, raw_data=False):
    print("Run experiment subproblems b5.")
    print("Estimated time: ~ 90 min on Intel Core i7-7820HQ CPU 2.9 GHz * 8")
    # file_output: str = join(folder_output, "exp4_" + datetime.now().strftime("time=%m-%d-%Y_%H-%M-%S") + ".csv")

    kcfs: List[crc.ScoringScheme] = []
    # sets the values of b6 to consider (note that b4 is set to 0)
    penalties_6: List[float] = [0.0, 0.25, 0.5, 0.75, 1.]
    # creation of scoring schemes ( KCFs )
    for penalty_6 in penalties_6:
        kcfs.append(crc.ScoringScheme([[0., 1., 1., 0., 1., penalty_6], [1., 1., 0., 1., 1., 0.]]))

    bench: BenchPartitioningScoringScheme = BenchPartitioningScoringScheme(
        dataset_folder=path_dataset,
        # the kcfs to consider
        scoring_schemes_exp=kcfs,
        # all the files (tuples of rankings) are considered
        dataset_selector_exp=crc.DatasetSelector(),
        # = B[6], for printing changing value of B
        changing_coeff=(0, 5),
        # range of number of elements to consider for the output
        intervals=[(30, 59), (60, 99), (100, 299), (300, 1121)])

    # run experiment and print results. If raw_data is true: also print all parameters of experiment (readme)
    # and the raw data that was used to compute the final data. If parameter is false, only final data is displayed
    bench.run(raw_data)


# runs experiment 5 in research paper
def run_experiment_bio_orphanet(dataset_path: str, raw_data=False, figures=False):
    print("Run experiment goldstandard bio.")
    print("Estimated time: ~ 120 min on Intel Core i7-7820HQ CPU 2.9 GHz * 8")
    # file_output: str = join(folder_output, "exp5_" + datetime.now().strftime("time=%m-%d-%Y_%H-%M-%S") + ".csv")

    # sets the values of b5-b4 to consider (note that b4 is set to 0)
    values_b5: List[float] = [0.0, 0.25, 0.5, 0.75, 1, 2]
    kcfs: List[crc.ScoringScheme] = []
    # creation of the scoring schemes (the KCFs)
    for value_b5 in values_b5:
        kcfs.append(crc.ScoringScheme([[0., 1., 1., 0., value_b5, 0.], [1., 1., 0., value_b5, value_b5, 0]]))

    exp1: ExperimentOrphanet = ExperimentOrphanet(
        dataset_folder=dataset_path,
        # the kcfs to consider
        scoring_schemes=kcfs,
        # the top-k to consider
        top_k_to_test=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        # algorithm to compute the consensus
        algo=crc.ParCons(bound_for_exact=150, auxiliary_algorithm=crc.BioConsert()),
        # selects all the tuples of rankings with at least 100 elements and 3 rankings
        # dataset_selector=DatasetSelector(nb_elem_min=100, nb_rankings_min=3)
        dataset_selector=crc.DatasetSelector(nb_elem_min=100, nb_rankings_min=3),

    )

    # run experiment and print results. If raw_data is true: also print all parameters of experiment (readme)
    # and the raw data that was used to compute the final data. If parameter is false, only final data is displayed
    exp1.run(raw_data, figures=figures)


def run_experiment_students_ijar(raw_data=False, figures=False):
    print("Run experiment students.")
    print("Estimated time: ~ 30 min on Intel Core i7-7820HQ CPU 2.9 GHz * 8")
    # file_output: str = join(folder_output, "exp6_" + datetime.now().strftime("time=%m-%d-%Y_%H-%M-%S") + ".csv")

    # seed 1 is set for python and numpy
    random.seed(1)
    np.random.seed(1)
    # sets the values of b5-b4 to consider (note that b4 is set to 0)
    values_b5: List[float] = [0., 0.25, 0.5, 0.75, 1., 2]
    kcfs: List[crc.ScoringScheme] = []
    # creation of the scoring schemes (the KCFs)
    for value_b5 in values_b5:
        kcfs.append(crc.ScoringScheme([[0., 1., 1., 0., value_b5, 0.], [1., 1., 0., value_b5, value_b5, 0]]))
    """"
    the parameters are all the ones detailled in the research paper. 100 student classes, each student class
    has 280 students from track 1 and 20 from track 2. In tract 1: choose uniformly 14 classes over 17 and in track
    2: choose uniformly 9 classes over the same 17. The marks obtained by students of track 1: N(10, 5*5) and by 
    students of track 2 : N(16, 4*4). Evaluation is made using top-20 of the consensuses
    """
    expe: MarksExperiment = MarksExperiment(
        # number of tuples of rankings to create
        nb_years=100,
        # number of students in track1
        nb_students_track1=280,
        # number of students in track2
        nb_students_track2=20,
        # number of classes the students can choose
        nb_classes_total=17,
        # number of classes the students of track1 choose (uniformly)
        nb_classes_track1=14,
        # number of classes the students of track2 choose (uniformly)
        nb_classes_track2=9,
        # mean marks for students in track1 for each class (normal distribution)
        mean_track1=10,
        # square of standard deviation of students marks in track1 for each class
        variance_track1=5,
        # mean marks for students in track2 for each class (normal distribution)
        mean_track2=16,
        # square of standard deviation of students marks in track2 for each class
        variance_track2=4,
        # top-k to consider for the experiment (comparison consensus and overall average)
        topk=20,
        # kcfs to consider
        scoring_schemes=kcfs,
        # algorithm to compute consensus
        algo=crc.CopelandMethod())

    # run experiment and print results. If raw_data is true: also print all parameters of experiment (readme)
    # and the raw data that was used to compute the final data. If parameter is false, only final data is displayed
    expe.run(raw_data, figures)


def validate_experiments(value: str) -> Set[int]:
    values: List[str] = value.split(',')
    experiments_to_run: Set[int] = set()
    for val in values:
        try:
            val_int = int(val)
            if 1 <= val_int <= 6:
                experiments_to_run.add(val_int)
            else:
                raise argparse.ArgumentTypeError(
                    f"Invalid value for exp: {val}. Valid options are {list(range(1, 7))}.")
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value for exp: {val}. Exp should be a comma separated list of numbers.")
    return experiments_to_run


def display_manual():
    print("Program to reproduce the 6 experiments of ijar2022 rank aggregation paper. One argument at least is needed.")
    print("There are 6 experiments: exp=1,2,3,4,5,6")
    print("Exp 1 and 2 form the part1 (bench time computation).")
    print("Exp 3 and 4 form the part2 (evaluation of partitioning).")
    print("Exp 5 and 6 form the part3 (evaluation of model with goldstandards).")
    print("If you want to reproduce the two experiments of part 1: then the argument is \"part=1\"")
    print("If you want to reproduce the two experiments of part 1 and 2: then the argument is \"part=1,2\"")
    print("If you want to reproduce experiments 1, 4, 6: then the argument is \"exp=1,4,6\"")
    print("You can combine arguments: exp=1,3 part=3 runs experiments 1, 3, 5 and 6.")
    print("If you want to reproduce all the experiments, then the argument is \"all\"")


if len(sys.argv) == 1:
    display_manual()

else:
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', type=validate_experiments, default=set(),
                        help='List of experiments to run, e.g., exp=1,2,3. Valid options are {1, ..., 6}.')
    parser.add_argument('--all', action='store_true',
                        help='If set, run all experiments (overrides -exp).')
    parser.add_argument('--raw_data', action='store_true',
                        help='If set, display raw data.')
    parser.add_argument('--figures', action='store_true',
                        help='If set, display figures.')
    args = parser.parse_args()

    if args.all:
        args.exp = set(range(1, 7))

    exp_to_run, raw_data_display, figures_display = args.exp, args.raw_data, args.figures

    print(f"Experiments to run, by order of needed computation time: {exp_to_run}")
    print(f"Display all raw data: {args.raw_data}")
    print(f"Display figures: {args.figures}")
    path_biological_dataset = str(Path(".") / "ijar_data" / "datasets" / "biological_dataset")

    # associate the int number of experiment with the function to execute
    experiments: Dict[int, Callable] = {
        1: run_bench_time_alg_exacts_ijar,
        2: run_bench_exact_optimized_scoring_scheme_ijar,
        3: run_count_subproblems_t_ijar,
        4: run_count_subproblems_b6_ijar,
        5: run_experiment_bio_orphanet,
        6: run_experiment_students_ijar
    }
    # for each experiment to run, by increasing order of needed computation time
    for exp in [exp_order for exp_order in [6, 3, 4, 5, 2, 1] if exp_order in exp_to_run]:
        # only 2 arguments for 6th experiment which does not need the BiologicalDataset
        if exp == 6:
            experiments[exp](raw_data_display, figures_display)
        elif exp in [3, 4]:
            experiments[exp](path_biological_dataset, raw_data_display)
        # 3 arguments for the other ones
        else:
            experiments[exp](path_biological_dataset, raw_data_display, figures_display)
