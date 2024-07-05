from abc import ABC, abstractmethod
from src.search_result import SearchResult
from typing import Any, List, Iterable

class VectorDB(ABC):

    @abstractmethod
    def insert_vector(entities: Iterable[Any]) -> None:
        """ 
            Добавляет `entities` в базу.
            `entities` должен содержать эмбеддинги, может содержать метаданные
        """
        raise NotImplementedError
    
    @abstractmethod
    def search_vector(embedding: Iterable[Any], topk: int) -> List[SearchResult]:
        """ Выполняет поиск по вектору в базе """
        raise NotImplementedError
    
    @abstractmethod
    def create_index():
        """ Обучает индекс векторной БД """
        raise NotImplementedError