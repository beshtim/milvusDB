# MilvusBD

Векторная БД Milvus

## Что это и зачем

Пример того, как просто использовать SOTA self-hosted векторную БД

Проектов, где может быть необходимо делать поиск по эмбеддингам много, поэтому чтобы меньше тратить времени на вопросы "какую же БД взять" и "как ей пользоваться" есть этот репозиторий

Если все, что вам надо - хранить эмбеддинги и "названия", то даже ничего менять не надо *(кроме параметров поиска и индекса - они совсем не оптимальные)*

А если все же что-то чуть интереснее - то там ничего сложного

## Как это установить и запустить

Официальные [гитхаб](https://github.com/milvus-io/milvus) и [документация](https://milvus.io/docs/overview.md), где можно найти вообще все и по любому вопросу

Минимальный рабочий вариант: CPU, все держится на одном баш скрипте, в котором кстати и конфигурация располагается

```bash
standalone_embed.sh start # скачивание и запуск контейнера
standalone_embed.sh stop # останавливает контейнер
standalone_embed.sh delete # подчищает остаточные данные
```

Версия контейнера там же - `milvusdb/milvus:v2.3.9`

А еще есть:  

- [докер композ](https://milvus.io/docs/v2.3.x/install_standalone-docker-compose.md)  
- [k8s](https://milvus.io/docs/v2.3.x/install_cluster-milvusoperator.md)  
- [GPU](https://milvus.io/docs/v2.3.x/install_standalone-helm-gpu.md) - кластер, но можно через `minikube`

Ну и само API для питона - `pymilvus` *(+numpy и tqdm для удобства)*  
*(кстати есть SDK не только для питона - смотрите в гитхабе)*  

```bash
pip install -r requirements.txt
```

А еще у них версии контейнеров связаны с версиями библ, поэтому при апгрейде надо обновляться - табличка есть в [pypi](https://pypi.org/project/pymilvus/)

*(раскидывать отдельные части проекта по разным модулям / контейнерам это очень хорошо с точки зрения сис диса)*

## Пример и чуть-чуть по коду

Минимальный пример можно посмотреть в `./example.py`. 

*Пример* чуть-чуть оверкил? Да, но зато *почти* Plug&Play

Наследование от `VectorDB` исключительно чтобы минимизировать боль миграции и/или добавления других ВБД. `SearchResult` аналогично для унификации интерфейса взаимодействия.

Сам `MilvusDB` может держать в себе несколько коллекций *(можно даже обычные БД в него же затолкать, милвус и такое умеет)* и автоматически подтягивать информацию о них при инициализации *(но я не прям чтоб проверял)*

## Сломался standalone_embed.sh

Бывает

```bash
./src/vector_db/milvus/standalone_embed.sh start
sed $'s/\r$//' ./src/milvus/standalone_embed.sh > ./src/milvus/standalone_embed.sh
```
