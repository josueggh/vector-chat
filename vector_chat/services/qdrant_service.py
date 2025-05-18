"""
QdrantService for vector storage and retrieval.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models

from vector_chat.config import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Service for interacting with Qdrant vector database.
    """

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        vector_size: Optional[int] = None,
        url: str = QDRANT_URL,
        api_key: Optional[str] = QDRANT_API_KEY,
        distance: models.Distance = models.Distance.COSINE,
    ):
        """
        Initialize Qdrant client and ensure collection exists.

        Args:
            collection_name: Name of the collection to use
            vector_size: Size of vectors to store (required for new collections)
            url: URL of the Qdrant server
            api_key: API key for Qdrant server
            distance: Distance metric to use

        Raises:
            ValueError: If collection doesn't exist and vector_size is not provided
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

        # Check if collection exists, create if needed
        if not self.client.collection_exists(self.collection_name):
            if vector_size is None:
                raise ValueError(
                    f"Collection '{collection_name}' does not exist. "
                    "Provide vector_size to create it."
                )
            logger.info(
                f"Creating collection '{collection_name}' with vector size {vector_size}"
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance),
            )
        else:
            logger.info(f"Using existing collection: {collection_name}")

    def upsert(
        self,
        ids: List[Union[str, int]],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Insert or update vectors in the collection.

        Args:
            ids: List of unique IDs for the vectors
            vectors: List of vector embeddings
            payloads: Optional list of payload dictionaries

        Raises:
            Exception: If there's an error upserting vectors
        """
        try:
            points = []
            for i, vec in enumerate(vectors):
                payload = payloads[i] if payloads else {}
                points.append(
                    models.PointStruct(id=ids[i], vector=vec, payload=payload)
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(
                f"Upserted {len(points)} vectors into collection '{self.collection_name}'"
            )
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise

    def search(
        self, vector: List[float], top_k: int = 5, score_threshold: float = 0.3
    ) -> List[Tuple[Union[str, int], float, Dict[str, Any]]]:
        """
        Search for similar vectors in the collection.

        Args:
            vector: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of tuples (id, score, payload)

        Raises:
            Exception: If there's an error searching
        """
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=top_k,
                    with_payload=True,
                    score_threshold=score_threshold,
                )
            results = [(hit.id, hit.score, hit.payload) for hit in hits]
            logger.debug(f"Found {len(results)} results for search query")
            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise

    def check_collection_exists(self) -> bool:
        """
        Check if the collection exists.

        Returns:
            True if collection exists, False otherwise
        """
        try:
            return self.client.collection_exists(self.collection_name)
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False
