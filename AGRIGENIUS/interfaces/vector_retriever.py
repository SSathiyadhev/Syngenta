# interfaces/vector_retriever.py
from abc import ABC, abstractmethod
from models import CropType, PestType

class BaseVectorRetriever(ABC):
    """
    Abstract Base Class enforcing strict structural interfaces
    for semantic retrieval-augmented generation (RAG) knowledge databases.
    """
    
    @abstractmethod
    def similarity_search(self, crop: CropType, pest: PestType) -> str:
        """
        Queries a vector index using embedding similarity calculations.
        Must return verified corporate agronomic manual instructions as a string.
        """
        pass