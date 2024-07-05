from typing import Tuple
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

def average_embeddings(embeddings_file: str, labels_file: str) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """ 
    Усреднение эмбеддингов с одинаковым лейблом 
    Исследования подтвердили, что это лучший вариант
        (по медианному, какому либо перцентилю или как либо иначе не лучше)
    """
    
    embeddings = np.load(embeddings_file)
    labels = np.load(labels_file)

    unique_labels = np.unique(labels)

    averaged_embeddings = []
    class_labels = []

    for label in tqdm(unique_labels):
        indices = np.where(labels == label)[0]

        avg_embedding = np.mean(embeddings[indices], axis=0)

        averaged_embeddings.append(avg_embedding)
        class_labels.append(label)

    return np.array(averaged_embeddings), np.array(class_labels)