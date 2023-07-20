"""
Module to manage genes, diseases, biological databases Orphanet and NCBI Gene
"""
from typing import Dict, Tuple, Set, List
from enum import Enum, unique


@unique
class Database(Enum):
    """
    Enumeration of biological databases, for crossref.
    """
    Orphanet = 0
    GeneNCBI = 1
    Ensembl = 2
    OMIM = 3
    Reactome = 4
    Genatlas = 5
    HGNC = 6
    SwissProt = 7
    Iuphar = 8
    IMGT_Gene_db = 9
    miRBase = 10


class Disease:
    """
    Class to represent a Disease.
    """

    def __init__(self, id_d: int or str, id_orpha: int or str, name: str):
        """

        :param id_d: The unique int id of the disease.
        :param id_orpha: The Orphanet id of the disease.
        :param name: The name of the disease.
        """
        self.__id_d: str = str(id_d)
        self.__name: str = name
        self.__id_orpha: str = str(id_orpha)
        # id of MeSH database
        self.__id_MeSH: str = ""
        # associated genes, with nature and status as a tuple of str
        self.__genes: Dict[Gene, Tuple[str, str]] = {}
        # set of assessed genes
        self.__assessed_genes: Set[Gene] = set()

    def add_associated_gene(self, gene: 'Gene', nature: str, status: str) -> None:
        """

        :param gene: The Gene object to associate with the Disease
        :param nature: Orpahnet information, string
        :param status: Orphanet information, string
        :return: None
        """
        self.__genes[gene] = nature, status
        if status.upper() == "ASSESSED":
            self.__assessed_genes.add(gene)

    @property
    def associated_genes(self) -> Dict['Gene', Tuple[str, str]]:
        """

        :return: The genes associated with the Disease as keys, two orphanet information (str) as values
        """
        return self.__genes

    @property
    def name(self) -> str:
        """

        :return: The name of the disease.
        """
        return self.__name

    @property
    def id_mesh(self) -> str:
        """

        :return: The MeSH id of the Disease
        """
        return self.__id_MeSH

    @id_mesh.setter
    def id_mesh(self, id_mesh: str) -> None:
        """
        Setter for id_mesh

        :param id_mesh: the updated mesh id
        :return: None
        """
        self.__id_MeSH = id_mesh

    @property
    def id_orpha(self) -> str:
        """

        :return: The orphanet id of the disease.
        """
        return self.__id_orpha

    @property
    def assessed_genes(self) -> Set['Gene']:
        """

        :return: The Genes related to the disease, with status "assessed" by Orphanet
        """
        return self.__assessed_genes

    def __str__(self) -> str:
        """
        :return: A string representation of the disease.
        """

        base_str: str = f"id:{self.__id_d};id_orpha:{self.__id_orpha};name:{self.__name}"
        return f"{base_str};{self.__id_MeSH}" if self.__id_MeSH != "-" else base_str

    def str_associated_genes(self) -> str:
        """
        :return: A string representation of the genes associated to the disease.
        """
        return "\n".join(f"\t{gene} : {remarks[0]}, {remarks[1]}" for gene, remarks in self.__genes.items()) + "\n"

    def get_associated_genes_with_ncbi_gene_id(self) -> Set[int]:
        """

        :return: The Set of int IDs from NCBI Gene databse of the genes associated to the Disease.
        """
        return {int(gene.crossrefs[Database.GeneNCBI.name]) for gene in self.__genes if
                Database.GeneNCBI.name in gene.crossrefs}

    def get_assessed_associated_genes_with_ncbi_gene_id(self) -> Set[int]:
        """

        :return: The Set of int IDs from NCBI Gene databse of the genes associated to the Disease and tagged assessed by
                 Orphanet.
        """
        return {int(gene.crossrefs[Database.GeneNCBI.name]) for gene in self.__assessed_genes if
                Database.GeneNCBI.name in gene.crossrefs}


class Gene:
    """
    Class to represent Genes. A Gene object may come from different biological databases.
    """
    def __init__(self, id_gene: int or str, name: str, symbol: str, database_from: str):
        """
        To create a Gene instance.

        :param id_gene: The ID of the gene. According to the database, the id may be int or str.
        :param name: The name of the gene.
        :param symbol: The symbol of the gene.
        :param database_from: The database where the information about the gene come from.
        """
        # store gene id as str, for types consistance.
        self.__id_gene = str(id_gene)
        # crossrefs of the gene as a dict where a key is a database name and value is the gene id as str
        # for the database.
        self.__crossref: Dict[str, str] = {}
        # name of the gene
        self.__name: str = name
        # symbol of the gene
        self.__symbol: str = symbol
        # database name where the information come from
        self.__database_from: str = database_from

    def add_crossref(self, db: Database, id_gene_db_crossref: str) -> None:
        """

        :param db: The Database with the cross reference.
        :param id_gene_db_crossref: The gene id for the associated database.
        :return: None
        """
        self.__crossref[db.name] = id_gene_db_crossref

    @property
    def crossrefs(self) -> Dict[str, str]:
        """

        :return: The cross-references of the Gene as a Dict where key = database name and value =
                 gene id as str for the database.
        """
        return self.__crossref

    @property
    def id_gene(self) -> str:
        """

        :return: The id of the gene, stored as str.
        """
        return self.__id_gene

    @property
    def name(self) -> str:
        """

        :return: The name of the gene.
        """
        return self.__name

    @property
    def symbol(self) -> str:
        """

        :return: The symbol of the gene.
        """
        return self.__symbol

    def __str__(self) -> str:
        """
        :return: A formatted string with the gene's ID, the database it originates from, its name, symbol,
                 and cross-references.
        """
        return f"{self.__id_gene} ({self.__database_from}): {self.__name} ({self.__symbol}) " \
               f"- crossref: {str(self.__crossref)}"

    def __repr__(self) -> str:
        """

        :return: A String representation of the Gene.
        """
        return str(self.__id_gene)


class BiologicalDatabase:

    def __init__(self):
        self.__fill_dbref_name()
        self._list_genes: List[Gene] = []

    def __fill_dbref_name(self):
        self._mapping_db: Dict[str, Database] = {}

    def get_db_id(self, db_name: str) -> Database:
        return self._mapping_db[db_name]

    def get_genes(self) -> List[Gene]:
        return self._list_genes


class GeneNcbiParser(BiologicalDatabase):
    def __init__(self, path_data: str):
        super().__init__()
        self._fill_dbref_name()
        self.__gene_from_id = {}
        with open(path_data, encoding="ISO-8859-1") as fileOrphanet:
            lines = fileOrphanet.read().split("\n")
            for line in lines[1:-1]:
                line_splitted = line.split("\t")
                crossref = line_splitted[5]
                gene_id = line_splitted[1]
                gene_name = line_splitted[2]
                gene = Gene(gene_id, line_splitted[8], gene_name, "GeneNCBI")
                self._list_genes.append(gene)
                self.__gene_from_id[gene_id] = gene
                for db in crossref.split("|"):
                    dbcrossref = db.split(":")[0]
                    if dbcrossref != "-":
                        gene.add_crossref(self._mapping_db[dbcrossref], db.split(":")[-1])

    def _fill_dbref_name(self) -> None:
        self._mapping_db["MIM"] = Database.OMIM
        self._mapping_db["Ensembl"] = Database.Ensembl
        self._mapping_db["HGNC"] = Database.HGNC
        self._mapping_db["IMGT/GENE-DB"] = Database.IMGT_Gene_db
        self._mapping_db["miRBase"] = Database.miRBase

    def get_gene(self, ncbi_id: str or int) -> Gene:
        return self.__gene_from_id[str(ncbi_id)]
