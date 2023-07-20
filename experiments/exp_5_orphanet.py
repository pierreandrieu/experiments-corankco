from experiments.experiment import ExperimentFromDataset
from experiments.orphanet_parser import OrphanetParser
from experiments.biological_objects import Disease
from typing import List, Set, Dict, Tuple
from experiments.utils import join_paths, get_parent_path
import numpy as np
import matplotlib.pyplot as plt
import corankco as crc


class ExperimentOrphanet(ExperimentFromDataset):
    """

    Class to run the experiments 4 and 5 of IJAR paper
    Heritage: ExperimentFromDataset which provides useful methods to handle datasets within an experiment

    """
    def __init__(self,
                 dataset_folder: str,
                 scoring_schemes: List[crc.ScoringScheme],
                 top_k_to_test: List[int],
                 algo: crc.RankAggAlgorithm = crc.ParCons(bound_for_exact=150),
                 dataset_selector: crc.DatasetSelector = None,
                 ):
        """

        :param dataset_folder: the folder of the biological dataset
        :param scoring_schemes: the scoring schemes to compare
        :param top_k_to_test: the top-k considered for the experiment
        :param algo: the algo to compute consensus rankings
        :param dataset_selector: the DatasetSelector object to filter the datasets

        """
        super().__init__(dataset_folder=dataset_folder, dataset_selector=dataset_selector)
        # load the xml file of Orphanet Database
        self.__orphanetParser: OrphanetParser = OrphanetParser.get_orpha_base_for_ijar(
            join_paths(get_parent_path(get_parent_path(dataset_folder)), "supplementary_data"))
        self.__algo: crc.RankAggAlgorithm = algo
        # keep only the datasets whose disease is in orphanet with at least one common gene between database NCBI Gene
        # and Orphanet
        self.__remove_useless_datasets()
        self.__scoring_schemes: List[crc.ScoringScheme] = []
        self.__consensus: Dict[crc.ScoringScheme, List[Tuple[crc.Dataset, crc.Consensus]]] = {}
        self.__scoring_schemes: List[crc.ScoringScheme] = scoring_schemes
        self.__top_k_to_test: List[int] = top_k_to_test

    def __is_mesh_term_in_orphanet(self, mesh_term: str) -> bool:
        """
        :param mesh_term: the mesh term of the disease to check if it is in Orphanet
        :return: True iif Orphanet has an entry for this disease

        """
        return self.__orphanetParser.contains_mesh(mesh_term)

    def get_disease_from_mesh(self, mesh_term: str) -> Disease:
        """

        :param mesh_term: the mesh term of the target disease
        :return: the associated Disease object
        """
        return self.__orphanetParser.get_disease_from_mesh(mesh_term)

    def __remove_useless_datasets(self):
        """
        Remove from the attribut list of dataset the datasets whose target disease is not considered by Orphanet
        (Orphanet only consider rare diseases) or such that there is no known gene associated to the disease in Orphanet
        database, or no common gene between NCBI gene (Database used to generate the BiologicalDataset) and Orphanet
        :return: None

        """
        res: List[crc.Dataset] = []
        # for each dataset of the BiologicalDataset
        for dataset in self._datasets:
            # get the name of the associated disease
            mesh_name_disease: str = dataset.name.split("_")[-1]
            # if the disease is in orphanet
            if self.__is_mesh_term_in_orphanet(mesh_name_disease):
                # get the set of the associated genes
                goldstandard: Set[int] = self.get_disease_from_mesh(
                    mesh_name_disease).get_assessed_associated_genes_with_ncbi_gene_id()
                real_gs: Set[int] = {gene_goldstandard for gene_goldstandard in goldstandard
                                        if dataset.contains_element(gene_goldstandard)}
                if len(real_gs) >= 1:
                    res.append(dataset)
        self._datasets = res

    def _run_raw_data(self, path_to_store_results: str = None) -> str:
        """

        :param path_to_store_results: the file to write each line of the raw result of the experiment
        a raw result is a result directly obtained by computation, not yet parsed to get a figure with x = f(y)
        :return: the string containing the raw result to be given to a parsing function. The string returned
        is in CSV format:
        b5_value;disease_name;nb_elements;goldstandard;size_goldstandard;consensus

        """
        # dict to associate to each disease represented by the string mesh term, its goldstandard i.e. genes associated
        # according to Orphanet
        h_disease_gs: Dict[str, Set[int]] = {}
        for dataset in self._datasets:
            mesh: str = dataset.name.split("_")[-1]
            initial_gs: Set[int] = self.get_disease_from_mesh(mesh).get_assessed_associated_genes_with_ncbi_gene_id()
            h_disease_gs[dataset.name] = {gene for gene in initial_gs if dataset.contains_element(gene)}
        line: str = "b5-b4;dataset;nb_elements;goldstandard;size_goldstandard;consensus\n"
        res: str = line
        if path_to_store_results is not None:
            self._write_line_in_file(path_to_store_results, line)
        # consensus computation (consensus is stored as an attribute of the class to be re-used later)
        self.__compute_consensus()
        # for each scoring_scheme
        for sc in self.__scoring_schemes:
            # for each Tuple[disease, consensus] associated to the scoring scheme
            for dataset, consensus in self.__consensus[sc]:
                # get the goldstandard and print the information
                gs = h_disease_gs[dataset.name]
                # store the useful information
                line: str = str(sc.b_vector[4]) + ";" + dataset.name + ";" + str(dataset.nb_elements) + ";" + str(gs) + ";" \
                       + str(len(gs)) + ";" + str(consensus[0]) + "\n"
                res += line
                if path_to_store_results is not None:
                    self._write_line_in_file(path_to_store_results, line)

        return res

    def _run_final_data(self, raw_data: str) -> str:
        """

        :param raw_data: the initial data, not yet parsed to be in a figure
        :return: a string in csv format, k;b5-b4;
        """
        # list of all the top-k to test / compare
        top_k_all: List[int] = self.__top_k_to_test

        # first line
        res: str = "k"
        for scoring_scheme in self.__scoring_schemes:
            res += ";b5-b4=" + str(scoring_scheme.b_vector[4]-scoring_scheme.b_vector[3])
        res += "\n"

        # for each k of target top k, associate a dict where keys are the values of b5 parameter and values are
        # the number of common elements between the associated consensus and the goldstandard
        h_res: Dict[int, Dict[float, List[int]]] = {}

        # for each top-k
        for top_k in top_k_all:
            # associate the top-k with an association b5 -> list of int where the ints are the number of common elements
            # between the consensus obtained and the goldstandard for each dataset
            # the length of the list is the number of datasets
            h_res[top_k]: Dict[float, List[int]] = {}
            # for each value of b5, initialize the associated list, empty for now
            for sc in self.__scoring_schemes:
                h_res[top_k][sc.b_vector[4]]: List[int] = []

        # now, filling the lists in the res Dict
        for top_k in top_k_all:
            # memoization
            h_res_topk: Dict[float, List[int]] = h_res[top_k]
            # for each line of the raw data b5_value;disease_name;nb_elements;goldstandard;size_goldstandard;consensus
            for line in raw_data.split("\n")[1:]:
                if len(line) > 1:
                    # get each piece of information of the line
                    b5_str, _, _, gs_str, _, consensus_str = line.split(";")
                    # memoization
                    h_res_topk_sc = h_res_topk[float(b5_str)]
                    # get the gold standard as a set of ints
                    # the [1:-1] is to remove { and }
                    gs: Set[int] = {int(elem) for elem in gs_str[1:-1].split(", ")}
                    h_res_topk_sc.append(crc.Consensus(
                        [crc.Ranking.from_string(consensus_str)]).evaluate_topk_ranking(gs, top_k=top_k))

        # now, getting the final csv file. columns = top_k, res_by_sc1, ..., res_by_sc_n
        # here, the res of a kcf, given a top k, is the average number of common elements between the top-k consensus
        # rankings and the goldstandard
        for top_k in top_k_all:
            res += str(top_k)
            h_topk = h_res[top_k]
            for sc in self.__scoring_schemes:
                res += ";" + str(np.sum(np.asarray(h_topk[sc.b_vector[4]])))
            res += "\n"
        return res

    def __compute_consensus(self):
        """
        Computes the consensus rankings for each scoring scheme of the experiment and each retained dataset
        :return: None
        """
        # Dict whoch associates for each scoring_scheme a list of couple (dataset, consensus)
        for sc in self.__scoring_schemes:
            self.__consensus[sc]: List[crc.Dataset, crc.Consensus] = []
            for dataset in self._datasets:
                consensus = self.__algo.compute_consensus_rankings(dataset, sc, True)
                self.__consensus[sc].append((dataset, consensus))

    def _display(self, final_data: str) -> None:
        """

        :param final_data: the parsed data to display in a figure
        :return: None
        """
        # the x points
        x_axis = []
        # the list of the y_i points, y_i is associated with i-th scoring_scheme
        y_axis: List[List[float]] = []
        for i in range(len(self.__scoring_schemes)):
            y_axis.append([])
        # use the final parsed data
        data_split = final_data.split("\n")
        # number of columns of the parsed data
        nb_columns = len(data_split[0].split(";"))
        # id of column where the result of the first scoring scheme to consider is stored
        first_col_result: int = nb_columns-len(self.__scoring_schemes)
        # for each line, except the first one and the empty ones at the end
        for line in data_split[1:]:
            if len(line) > 1:
                cols: List[str] = line.split(";")
                # get the x point
                x_axis.append(float(cols[0]))
                # get the y_i associated to each scoring_scheme
                for i in range(first_col_result, nb_columns):
                    y_axis[i-first_col_result].append(float(cols[i]))
        plt.xlabel("B5-B4")
        plt.ylabel("Sum of number of genes of the GS in top-k consensus")
        # list of colors to display
        colors: List[str] = ["b", "g", "r", "m", "y", "k"]
        id_col: int = 0
        for y in y_axis:
            plt.scatter(x_axis, y, edgecolors=colors[id_col])
            id_col += 1
        plt.show()
