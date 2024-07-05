from dataclasses import dataclass

@dataclass
class VectorDBConfig:
    embedding_dim: int = 512

    db_ip: str = 'localhost'
    db_port: str = '19530'