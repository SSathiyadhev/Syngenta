# infrastructure/local_faiss.py
from interfaces.vector_retriever import BaseVectorRetriever
from models import CropType, PestType
from core.logger import setup_production_logger

logger = setup_production_logger("LocalFaissRetriever")

class LocalFaissRetriever(BaseVectorRetriever):
    """Implements an index lookup framework modeling an embedding similarity store (e.g., FAISS)."""
    def __init__(self):
        # Maps keys completely to lowercase tokens based on actual crop_pest tags
        self._knowledge_index = {
            "wheat_thrips": "CRITICAL MANDATE: Apply lambda-cyhalothrin at early vegetative checks. Avoid excess nitrogen lines.",
            "wheat_stem_borer": "CRITICAL MANDATE: Apply chlorantraniliprole 18.5% SC configurations. Monitor tillering timelines closely.",
            "wheat_cutworm": "CRITICAL MANDATE: Apply broad-spectrum Syngenta soil fortifiers during twilight operational windows.",
            "wheat_ear_cockle": "CRITICAL MANDATE: Enforce strict clean seed-stock sortings. Prevent soil dampness spread lanes.",
            "potato_leaf_folder": "CRITICAL MANDATE: Integrate target emamectin benzoate arrays. Keep field borders clear of vectors.",
            "mustard_aphids": "CRITICAL MANDATE: Implement structural thiamethoxam treatments immediately if populations cross boundaries."
        }

    def similarity_search(self, crop: CropType, pest: PestType) -> str:
        """Simulates retriever.similarity_search() token matches using dataset enums."""
        query_token = f"{crop.value.lower()}_{pest.value.lower().replace(' ', '_')}"
        logger.info(f"Querying semantic embeddings index repository for vector key: {query_token}")
        return self._knowledge_index.get(query_token, "Apply pre-approved standard broad-spectrum protective Syngenta formulas safely.")