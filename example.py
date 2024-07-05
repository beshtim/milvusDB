from attr import fields
from src import MilvusDB, VectorDBConfig
import numpy as np
from tqdm import tqdm

def main():
    
    # Делаем пару переменных для удобства
    dataset_size: int = 40_000
    collection_name: str = 'test'
    search_field: str = 'embedding'

    # Инициализируем MilvusDB (подключаемся к, например, контейнеру)
    cfg = VectorDBConfig()
    milvus = MilvusDB(cfg.embedding_dim, cfg.db_ip, cfg.db_port)

    # Удаляем старую коллекцию, если надо
    milvus.drop_collection(collection_name)
    # И создаем новую
    milvus.create_collection(
        collection_name = collection_name, 
        collection_description = 'test_description', 
        collection_search_field_name = search_field,
        embedding_dim = cfg.embedding_dim,

        # Можно не прописывать - дефолтные возьмутся из этой же функции
        fields = MilvusDB.get_default_fields(cfg.embedding_dim, search_field),
        overwrite = True
    )

    # Создаем данные для заполнения коллекции
    class_labels = np.arange(dataset_size, dtype=np.int32)
    averaged_embeddings = np.random.rand(dataset_size, cfg.embedding_dim)

    #   Важно! Порядок должен быть такой же, как у fields выше
    #       т.е. если оно идет "эмбеддинги, имена", то и инсерт в таком же порядке
    entities = [
        averaged_embeddings,
        [f'random_person_{i}' for i in (class_labels)],
    ]

    # И заполняем
    #   Почему-то у меня не давало разом закинуть все данные
    #   поэтому чуть-чуть костылями, но идейно 
    #   ничего не мешает сделать milvus.insert_vector(collection_name, entities, True)
    #   без итерирования

    for idx in tqdm(range(0, 30_000, 10_000)):
        entities = [
            averaged_embeddings[idx:idx+10_000],
            [f'random_person_{i}' for i in (class_labels)][idx:idx+10_000],
        ]

        # Rule of thumb такой, что лучше лишний раз не "коммитить" изменения
        #   и продолжать их накапливать
        milvus.insert_vector(collection_name, entities, flush = False)
    
    # Делая flush (коммит) в конце, после добавления всех векторов
    milvus.flush(collection_name)

    # Создаем (обучаем) индекс и подгружаем его в память
    milvus.create_index(
        collection_name,

        # Можно не прописывать - дефолтные возьмутся из этой же функции
        index_parameters = MilvusDB.get_default_index_params()
    )
    milvus.load_collection_to_memory(collection_name)

    # Делаем поиск
    #   Функция ожидает именно батч векторов, желательно сразу в np.array
    #   Но нчие страшного если оно List - конвертнется внутри
    #   про батч можно тоже не волноваться - защита от такого внутри имеется
    result = milvus.search_vector(
        collection_name, 
        [averaged_embeddings[0]],

        # Можно не прописывать - дефолтные возьмутся из этой же функции
        search_parameters = MilvusDB.get_default_search_params()
    )

    # Выводим результат!
    print(result)

    # Ну и под конец работы дропает коллекцию и отключаемся от БД
    milvus.drop_collection(collection_name)
    milvus.disconnect()

if __name__ == '__main__':
    main()