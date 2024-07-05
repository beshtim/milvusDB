from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np

from src.vector_db import VectorDB
from src.search_result import SearchResult

from typing import Any, Dict, Iterable, List, Optional

class MilvusDB(VectorDB):
    availible_collections: Dict[str, Dict[str, Any]] = dict()
    
    def __init__(
        self, 
        ip: str, 
        port: str,
        connection_alias: str = 'default',
    ) -> None:
        """
        Initializes the object with the given IP address, port, and connection alias.
        
        Parameters:
            ip (str): The IP address.
            port (str): The port number.
            connection_alias (str, optional): The alias for the connection. Defaults to 'default'.
        """
        super().__init__()

        self.docker_ip = ip
        self.docker_port = port
        self.connection_alias = connection_alias

        connections.connect(self.connection_alias, host=ip, port=port)
        self._check_existing_collections()

    def _check_existing_collections(self):
        """ 
        Подгрузка существующих коллекций после перезапуска 
        Строгой необходимости в этом нет - можно вручную переподключиться к коллекциям 
            если чуть-чуть видоизменить create_collection и скрестить с тем, что тут
        """
        for collection_name in utility.list_collections():
            _collection = Collection(collection_name)
            _schema = _collection.schema
            _description = _collection.description
            
            self.availible_collections[collection_name] = {
                'name': collection_name,
                'milvus': Collection(collection_name),

                'description': _description,
                # Рассчет на то, что поле для "поиска" первое в списке
                'search_field_name': _schema.fields[0].name,
                'embedding_dim': _schema.fields[0].params['dim'],
                'schema': _schema
            }

    def create_collection(
        self, 
        collection_name: str, 
        collection_description: str,
        collection_search_field_name: str, 
        embedding_dim: int,
        fields: Optional[List[FieldSchema]] = None,
        overwrite: bool = False
    ):
        """
        Create a collection in the database with the given name, description, search field name, and fields.

        Parameters:
            collection_name (str): The name of the collection to create.
            collection_description (str): The description of the collection.
            collection_search_field_name (str): The name of the search field in the collection.
            fields (Optional[List[FieldSchema]]): List of field schemas for the collection. Defaults to None.
            overwrite (bool): Flag to indicate if the collection should be overwritten if it already exists. Defaults to False.
        """
        if utility.has_collection(collection_name) and overwrite:
            self.drop_collection()

        if fields is None:
            fields = MilvusDB.get_default_fields(embedding_dim, collection_search_field_name)
        
        schema = CollectionSchema(fields, collection_description)
        milvus = Collection(collection_name, schema)

        self.availible_collections[collection_name] = {
            'name': collection_name,
            'milvus': milvus,

            'embedding_dim': embedding_dim,
            'description': collection_description,
            'search_field_name': collection_search_field_name,
            'schema': schema,
        }

    def _assert_object_has_collection(self, collection_name: str):
        """
        Assert that an object has a specific collection.

        Parameters:
            collection_name (str): The name of the collection to check.

        Raises:
            ValueError: If the specified collection is not found.
        """
        if collection_name not in list(self.availible_collections.keys()):
            raise ValueError(f'Collection `{collection_name}` not found')
    
    def flush(self, collection_name: str):
        self._assert_object_has_collection(collection_name)
        collection_obj = self.availible_collections[collection_name]

        collection_obj['milvus'].flush()
        
    def insert_vector(
        self, 
        collection_name: str,
        entities: Iterable[Iterable[Any]], 
        flush: bool = False
    ):
        """
        Insert a vector into the specified collection.

        Parameters:
            collection_name (str): The name of the collection to insert the vector into.
            entities (Iterable[Iterable[Any]]): The vector or vectors to insert.
            flush (bool, optional): Whether to flush the collection after insertion. Defaults to False.

        Returns:
            The result of the insertion operation.
        """
        self._assert_object_has_collection(collection_name)
        collection_obj = self.availible_collections[collection_name]

        insert_result = collection_obj['milvus'].insert(entities)

        if flush:
            collection_obj['milvus'].flush()

        return insert_result 

    def load_collection_to_memory(self, collection_name: str):
        """
        Load a collection to memory.
        
        Parameters:
            collection_name (str): The name of the collection to load
        """
        # Перед поиском по коллекции надо загрузить ее в память
        self._assert_object_has_collection(collection_name)
        collection_obj = self.availible_collections[collection_name]

        collection_obj['milvus'].load()
    
    def create_index(
        self, 
        collection_name: str,
        index_parameters: Optional[Dict[str, Any]] = None
    ):
        """
        A method to create an index for a given collection.

        Parameters:
            collection_name (str): The name of the collection to create an index for.
            index_parameters (Optional[Dict[str, Any]]): Optional index parameters. If not provided, default index parameters will be used.
        """
        # https://milvus.io/docs/build_index.md
        # https://milvus.io/docs/index.md

        self._assert_object_has_collection(collection_name)
        collection_obj = self.availible_collections[collection_name]

        if index_parameters is None:
            index_parameters = MilvusDB.get_default_index_params()

        collection_obj['milvus'].create_index(
            collection_obj['search_field_name'], 
            index_parameters
        )
        
    def search_vector(
        self, 
        collection_name: str,
        batch_of_vectors_to_search: Iterable[Iterable[float]], 
        search_parameters: Optional[Dict[str, Any]] = None,
        output_fields: List[str] = ["example_name", 'pk'],
        overwrite_search_field: Optional[str] = None,
        topk: int = 1
    ) -> List[List[SearchResult]]:
        """
        Persforms search for vectors in a specified collection with specified parameters. 

        Parameters:
            collection_name (str): The name of the collection to search in.
            batch_of_vectors_to_search (Iterable[Iterable[float]]): A batch of vectors to search for.
            search_parameters (Optional[Dict[str, Any]]): Parameters for the search operation. Defaults to None.
            output_fields (List[str]): Fields to include in the output. Defaults to ["example_name", 'pk'].
            topk (int): The number of top results to return. Defaults to 1.

        Returns:
            List[List[SearchResult]]: A list of search results for the input vectors.
        """
        # https://milvus.io/docs/search.md
        # https://milvus.io/docs/v2.3.x/query.md

        self._assert_object_has_collection(collection_name)
        collection_obj = self.availible_collections[collection_name]

        if search_parameters is None:
            search_parameters = MilvusDB.get_default_search_params()
        
        #   Потому что нужен батч векторов и бывает забываешь делать [вектор]
        batch_of_vectors_to_search = np.array(batch_of_vectors_to_search)
        if len(batch_of_vectors_to_search.shape) == 1:
            batch_of_vectors_to_search = np.expand_dims(batch_of_vectors_to_search, axis=0)

        search_field_name = collection_obj['search_field_name']
        if overwrite_search_field is not None:
            search_field_name = overwrite_search_field

        batch_of_milvus_results = collection_obj['milvus'].search(
            batch_of_vectors_to_search, 
            search_field_name, 
            search_parameters, 
            limit=topk, 
            output_fields = output_fields
        )

        #   Кстати поиск еще вот так можно делать
        # result = hello_milvus.query(expr="random > -14", output_fields=["random", "embeddings"])
        # result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > -12", output_fields=["random"])

        batch_of_results = []

        for single_milvus_result in batch_of_milvus_results:
            results = []
            for milvus_hit in single_milvus_result:
                result = SearchResult(
                    # Эти два параметра есть всегда при любом поиске
                    milvus_hit.id, 
                    milvus_hit.distance, 

                    # Остальные же складываются milvus_hit.entity.<НАЗВАНИЕ>, вроде можно и milvus_hit.entity[<НАЗВАНИЕ>] 
                    kwarg_dict = {output_field_name : milvus_hit.get(output_field_name) for output_field_name in output_fields}
                )
                results.append(result)
            batch_of_results.append(results)

        return batch_of_results
    
    def drop_collection(self, collection_name: str) -> None:
        """
        Drops a collection if it exists and removes it from the available collections.
        
        Parameters:
            collection_name (str): The name of the collection to drop.
        
        Returns:
            None
        """
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        if self.availible_collections.get(collection_name, None) is not None:
            del self.availible_collections[collection_name]

    def disconnect(self,) -> None:
        connections.disconnect(self.connection_alias)

    @classmethod
    def get_default_fields(cls, embedding_dim: int = 512, collection_search_field_name: str = 'embedding') -> List[FieldSchema]:
        """ Пример дефолтных полей """
        return [
            FieldSchema(name=collection_search_field_name, dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="example_name", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        ]
    
    @classmethod
    def get_default_index_params(cls) -> Dict[str, Any]:
        """ Пример дефолтных параметров создания индекса """
        # https://milvus.io/docs/v2.3.x/build_index.md
        return {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
    
    @classmethod
    def get_default_search_params(cls) -> Dict[str, Any]:
        """ Пример дефолтных параметров для поиска """

        # https://milvus.io/docs/v2.3.x/search.md
        return {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
