from experiments.database_enum import Database
from typing import Tuple, Dict


class Gene:

    def __init__(self, id_gene: int or str, name: str, symbol: str, database_from: str):
        self.__id_gene = str(id_gene)
        self.__crossref = {}
        self.__name = name
        self.__symbol = symbol
        self.__database_from = database_from

    def add_crossref(self, db: Database, id_gene_db_crossref: str):
        self.__crossref[db.name] = id_gene_db_crossref

    def __get_crossref(self) -> Dict[Database, str]:
        return self.__crossref

    def __get_id(self) -> str:
        return self.__id_gene

    def __get_name(self) -> str:
        return self.__name

    def __get_symbol(self) -> str:
        return self.__symbol

    crossref = property(__get_crossref)
    id_gene = property(__get_id)
    name = property(__get_name)
    symbol = property(__get_symbol)

    def similarity(self, gene) -> Tuple[float, int]:
        crossref_in_both_db = 0
        same_val = 0
        gene_crossref = gene.get_crossref()
        for db_self in self.__crossref:
            if db_self in gene_crossref:
                crossref_in_both_db += 1
                if self.__crossref[db_self] == gene_crossref[db_self]:
                    same_val += 1
        if crossref_in_both_db == 0:
            return -1., 0
        return same_val / crossref_in_both_db, crossref_in_both_db

    def has_same_name(self, gene) -> bool:
        return self.__name == gene._name and self.__symbol == gene.__symbol

    def __str__(self) -> str:
        return self.__id_gene + " (" + self.__database_from + "):  " + self.__name + " (" + self.__symbol + ")" \
               + " - crossref: " + str(self.__crossref)

    def __repr__(self) -> str:
        return str(self.__id_gene)
