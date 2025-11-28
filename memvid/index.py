"""
Index management for embeddings, BM25, and hybrid search

Enhanced with BM25 support and Reciprocal Rank Fusion (RRF) for hybrid retrieval.
Based on 2025 research showing hybrid search outperforms single-method approaches.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional, Literal
import logging
from pathlib import Path
import pickle
from tqdm import tqdm

from .config import get_default_config

logger = logging.getLogger(__name__)

# Optional BM25 support
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.info("rank_bm25 not installed. BM25 search disabled. Install with: pip install rank-bm25")


class IndexManager:
    """
    Manages embeddings, FAISS index, BM25 index, and metadata for fast retrieval.
    
    Supports three search modes:
    - 'vector': Semantic search using FAISS (default, original behavior)
    - 'bm25': Keyword-based search using BM25 (exact matching)
    - 'hybrid': Combines vector and BM25 using Reciprocal Rank Fusion (RRF)
    
    Research shows hybrid search typically outperforms single-method approaches,
    especially for technical documentation and precise terminology matching.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IndexManager
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_default_config()
        self.embedding_model = SentenceTransformer(self.config["embedding"]["model"])
        self.dimension = self.config["embedding"]["dimension"]
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Initialize BM25 index (if available)
        self.bm25_index = None
        self.tokenized_corpus = []
        self.bm25_enabled = BM25_AVAILABLE
        
        # Metadata storage
        self.metadata = []
        self.chunk_to_frame = {}  # Maps chunk ID to frame number
        self.frame_to_chunks = {}  # Maps frame number to chunk IDs
        
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        index_type = self.config["index"]["type"]
        
        if index_type == "Flat":
            # Exact search - best quality, slower for large datasets
            index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVF":
            # Inverted file index - faster for large datasets
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.config["index"]["nlist"])
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
        # Add ID mapping for retrieval
        index = faiss.IndexIDMap(index)
        return index
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        Simple whitespace tokenization with lowercasing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization - can be enhanced with nltk or spacy
        return text.lower().split()
    
    def _build_bm25_index(self, chunks: List[str]):
        """
        Build BM25 index from chunks
        
        Args:
            chunks: List of text chunks
        """
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available. Install with: pip install rank-bm25")
            return
        
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        self.tokenized_corpus = [self._tokenize(chunk) for chunk in chunks]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built successfully")

    def add_chunks(self, chunks: List[str], frame_numbers: List[int],
                   show_progress: bool = True) -> List[int]:
        """
        Add chunks to index with robust error handling and validation

        Args:
            chunks: List of text chunks
            frame_numbers: Corresponding frame numbers for each chunk
            show_progress: Show progress bar

        Returns:
            List of successfully added chunk IDs
        """
        if len(chunks) != len(frame_numbers):
            raise ValueError("Number of chunks must match number of frame numbers")

        logger.info(f"Processing {len(chunks)} chunks for indexing...")

        # Phase 1: Validate and filter chunks
        valid_chunks = []
        valid_frames = []
        skipped_count = 0

        for chunk, frame_num in zip(chunks, frame_numbers):
            if self._is_valid_chunk(chunk):
                valid_chunks.append(chunk)
                valid_frames.append(frame_num)
            else:
                skipped_count += 1
                logger.warning(f"Skipping invalid chunk at frame {frame_num}: length={len(chunk) if chunk else 0}")

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid chunks out of {len(chunks)} total")

        if not valid_chunks:
            logger.error("No valid chunks to process")
            return []

        logger.info(f"Processing {len(valid_chunks)} valid chunks")

        # Phase 2: Generate embeddings with batch processing and error recovery
        try:
            embeddings = self._generate_embeddings(valid_chunks, show_progress)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

        if embeddings is None or len(embeddings) == 0:
            logger.error("No embeddings generated")
            return []

        # Phase 3: Add to FAISS index
        try:
            chunk_ids = self._add_to_index(embeddings, valid_chunks, valid_frames)
            logger.info(f"Successfully added {len(chunk_ids)} chunks to index")
        except Exception as e:
            logger.error(f"Failed to add chunks to index: {e}")
            return []
        
        # Phase 4: Build BM25 index (new)
        if self.bm25_enabled:
            try:
                # Rebuild BM25 index with all chunks (including newly added)
                all_texts = [m["text"] for m in self.metadata]
                self._build_bm25_index(all_texts)
            except Exception as e:
                logger.warning(f"Failed to build BM25 index: {e}. BM25 search will be unavailable.")
                self.bm25_index = None
        
        return chunk_ids

    def _is_valid_chunk(self, chunk: str) -> bool:
        """Validate chunk for SentenceTransformer processing - SIMPLIFIED"""
        if not isinstance(chunk, str):
            return False

        chunk = chunk.strip()

        # Basic checks only
        if len(chunk) == 0:
            return False

        if len(chunk) > 8192:  # SentenceTransformer limit
            return False

        # Remove the harsh alphanumeric requirement - academic text has lots of punctuation!
        # Just ensure it's not binary data
        try:
            chunk.encode('utf-8')  # Can be encoded as UTF-8
            return True
        except UnicodeEncodeError:
            return False

    def _generate_embeddings(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings with error handling and batch processing"""

        # Try full batch first
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks (full batch)")
            embeddings = self.embedding_model.encode(
                chunks,
                show_progress_bar=show_progress,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True  # Helps with numerical stability
            )
            return np.array(embeddings).astype('float32')

        except Exception as e:
            logger.warning(f"Full batch embedding failed: {e}. Trying batch processing...")

            # Fall back to smaller batches
            return self._generate_embeddings_batched(chunks, show_progress)

    def _generate_embeddings_batched(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings in smaller batches with individual error handling"""

        all_embeddings = []
        valid_chunks = []
        batch_size = 100  # Smaller batches

        total_batches = (len(chunks) + batch_size - 1) // batch_size

        if show_progress:
            from tqdm import tqdm
            batch_iter = tqdm(range(0, len(chunks), batch_size),
                              desc="Processing chunks in batches",
                              total=total_batches)
        else:
            batch_iter = range(0, len(chunks), batch_size)

        for i in batch_iter:
            batch_chunks = chunks[i:i + batch_size]

            try:
                # Try batch
                batch_embeddings = self.embedding_model.encode(
                    batch_chunks,
                    show_progress_bar=False,
                    batch_size=16,  # Even smaller internal batch
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                all_embeddings.extend(batch_embeddings)
                valid_chunks.extend(batch_chunks)

            except Exception as e:
                logger.warning(f"Batch {i//batch_size} failed: {e}. Processing individually...")

                # Process individually
                for chunk in batch_chunks:
                    try:
                        embedding = self.embedding_model.encode(
                            [chunk],
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        all_embeddings.extend(embedding)
                        valid_chunks.append(chunk)

                    except Exception as chunk_error:
                        logger.error(f"Failed to embed individual chunk (length={len(chunk)}): {chunk_error}")
                        # Skip this chunk entirely
                        continue

        if not all_embeddings:
            raise RuntimeError("No embeddings could be generated")

        logger.info(f"Generated embeddings for {len(valid_chunks)} out of {len(chunks)} chunks")
        return np.array(all_embeddings).astype('float32')

    def _add_to_index(self, embeddings: np.ndarray, chunks: List[str], frame_numbers: List[int]) -> List[int]:
        """Add embeddings to FAISS index with error handling"""

        if len(embeddings) != len(chunks) or len(embeddings) != len(frame_numbers):
            # This can happen if some chunks were skipped during embedding
            min_len = min(len(embeddings), len(chunks), len(frame_numbers))
            embeddings = embeddings[:min_len]
            chunks = chunks[:min_len]
            frame_numbers = frame_numbers[:min_len]
            logger.warning(f"Trimmed to {min_len} items due to length mismatch")

        # Assign IDs
        start_id = len(self.metadata)
        chunk_ids = list(range(start_id, start_id + len(chunks)))

        # Train index if needed (for IVF)
        try:
            underlying_index = self.index.index  # Get the actual index from IndexIDMap wrapper

            if isinstance(underlying_index, faiss.IndexIVFFlat):
                nlist = underlying_index.nlist

                if not underlying_index.is_trained:
                    logger.info(f"🧠 FAISS IVF index requires training (nlist={nlist})")
                    logger.info(f"📊 Available embeddings: {len(embeddings)}")

                    # Check if we have enough data for training
                    if len(embeddings) < nlist:
                        logger.warning(f"❌ Insufficient training data: need at least {nlist} embeddings, got {len(embeddings)}")
                        logger.warning(f"💡 IVF indexes require more data. For single documents, consider using 'Flat' index type in config.")
                        logger.info(f"🔄 Auto-switching to IndexFlatL2 for reliable operation")
                        # Replace with flat index
                        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                        logger.info(f"✅ Switched to Flat index (exact search, slower but works with any dataset size)")
                    else:
                        recommended_min = nlist * 10  # IVF works better with 10x+ the nlist size
                        if len(embeddings) < recommended_min:
                            logger.warning(f"⚠️ Suboptimal training data: {len(embeddings)} embeddings (recommended: {recommended_min}+)")
                            logger.warning(f"💡 Consider using larger dataset or 'Flat' index for better results")

                        logger.info(f"🏋️ Training FAISS IVF index...")
                        logger.info(f"   - Training vectors: {len(embeddings)}")
                        logger.info(f"   - Clusters (nlist): {nlist}")
                        logger.info(f"   - Expected memory: ~{(len(embeddings) * self.dimension * 4) / 1024 / 1024:.1f} MB")

                        # Use sufficient training data
                        training_data = embeddings[:min(50000, len(embeddings))]
                        underlying_index.train(training_data)
                        logger.info("✅ FAISS IVF training completed successfully")
                else:
                    logger.info(f"✅ FAISS IVF index already trained (nlist={nlist})")
            else:
                logger.info(f"ℹ️ Using {type(underlying_index).__name__} (no training required)")

        except Exception as e:
            logger.error(f"❌ Index training failed with error: {e}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            logger.info(f"🔄 Falling back to IndexFlatL2 for reliability")
            logger.info(f"💡 To avoid this fallback, use 'Flat' index type in config for small datasets")
            # Fallback to simple flat index
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            logger.info(f"✅ Fallback complete - using exact search")

        # Add to index
        try:
            self.index.add_with_ids(embeddings, np.array(chunk_ids, dtype=np.int64))
        except Exception as e:
            logger.error(f"Failed to add embeddings to FAISS index: {e}")
            raise

        # Store metadata
        for i, (chunk, frame_num, chunk_id) in enumerate(zip(chunks, frame_numbers, chunk_ids)):
            try:
                metadata = {
                    "id": chunk_id,
                    "text": chunk,
                    "frame": frame_num,
                    "length": len(chunk)
                }
                self.metadata.append(metadata)

                # Update mappings
                self.chunk_to_frame[chunk_id] = frame_num
                if frame_num not in self.frame_to_chunks:
                    self.frame_to_chunks[frame_num] = []
                self.frame_to_chunks[frame_num].append(chunk_id)

            except Exception as e:
                logger.error(f"Failed to store metadata for chunk {chunk_id}: {e}")
                # Continue with other chunks
                continue

        return chunk_ids
    
    def _search_vector(self, query: str, top_k: int) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Semantic search using FAISS vector index
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk_id, score, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Gather results (convert distance to similarity score)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # Valid result
                # Convert L2 distance to similarity score (higher is better)
                score = 1.0 / (1.0 + float(dist))
                metadata = self.metadata[idx]
                results.append((int(idx), score, metadata))
        
        return results
    
    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Keyword search using BM25
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk_id, score, metadata) tuples
        """
        if not self.bm25_index:
            logger.warning("BM25 index not available. Falling back to vector search.")
            return self._search_vector(query, top_k)
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Gather results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                metadata = self.metadata[idx]
                results.append((int(idx), float(scores[idx]), metadata))
        
        return results
    
    def _rrf_fusion(
        self, 
        vector_results: List[Tuple[int, float, Dict[str, Any]]], 
        bm25_results: List[Tuple[int, float, Dict[str, Any]]], 
        top_k: int,
        k: int = 60
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Reciprocal Rank Fusion (RRF) to combine vector and BM25 results.
        
        RRF is a simple but effective fusion method that's robust to score differences
        between retrieval methods. Formula: RRF(d) = Σ 1/(k + rank(d))
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of final results to return
            k: RRF constant (default 60, as per original paper)
            
        Returns:
            Fused results sorted by RRF score
        """
        rrf_scores = {}
        chunk_metadata = {}
        
        # Process vector results
        for rank, (chunk_id, score, metadata) in enumerate(vector_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            chunk_metadata[chunk_id] = metadata
        
        # Process BM25 results
        for rank, (chunk_id, score, metadata) in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            chunk_metadata[chunk_id] = metadata
        
        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for chunk_id, rrf_score in sorted_chunks[:top_k]:
            results.append((chunk_id, rrf_score, chunk_metadata[chunk_id]))
        
        return results
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        mode: Literal["vector", "bm25", "hybrid"] = "vector"
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for similar chunks using specified mode.
        
        Args:
            query: Search query
            top_k: Number of results to return
            mode: Search mode
                - 'vector': Semantic search using embeddings (default, original behavior)
                - 'bm25': Keyword-based search using BM25
                - 'hybrid': Combines vector and BM25 using RRF fusion
            
        Returns:
            List of (chunk_id, score, metadata) tuples
            
        Example:
            # Original behavior (backward compatible)
            results = index_manager.search("machine learning", top_k=5)
            
            # Exact keyword matching
            results = index_manager.search("ISO 27001", top_k=5, mode="bm25")
            
            # Best of both worlds
            results = index_manager.search("security compliance", top_k=5, mode="hybrid")
        """
        if mode == "vector":
            return self._search_vector(query, top_k)
        
        elif mode == "bm25":
            if not self.bm25_index:
                logger.warning("BM25 not available. Install rank-bm25 or falling back to vector search.")
                return self._search_vector(query, top_k)
            return self._search_bm25(query, top_k)
        
        elif mode == "hybrid":
            if not self.bm25_index:
                logger.warning("BM25 not available for hybrid search. Using vector search only.")
                return self._search_vector(query, top_k)
            
            # Get results from both methods (fetch more for better fusion)
            fetch_k = min(top_k * 3, len(self.metadata))
            vector_results = self._search_vector(query, fetch_k)
            bm25_results = self._search_bm25(query, fetch_k)
            
            # Fuse results
            return self._rrf_fusion(vector_results, bm25_results, top_k)
        
        else:
            raise ValueError(f"Unknown search mode: {mode}. Use 'vector', 'bm25', or 'hybrid'.")
    
    def get_chunks_by_frame(self, frame_number: int) -> List[Dict[str, Any]]:
        """Get all chunks associated with a frame"""
        chunk_ids = self.frame_to_chunks.get(frame_number, [])
        return [self.metadata[chunk_id] for chunk_id in chunk_ids]
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by ID"""
        if 0 <= chunk_id < len(self.metadata):
            return self.metadata[chunk_id]
        return None
    
    def save(self, path: str):
        """
        Save index to disk
        
        Args:
            path: Path to save index (without extension)
        """
        path = Path(path)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix('.faiss')))
        
        # Save metadata and mappings
        data = {
            "metadata": self.metadata,
            "chunk_to_frame": self.chunk_to_frame,
            "frame_to_chunks": self.frame_to_chunks,
            "config": self.config,
            "bm25_enabled": self.bm25_enabled and self.bm25_index is not None
        }
        
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save BM25 data separately (tokenized corpus for rebuilding)
        if self.bm25_index and self.tokenized_corpus:
            bm25_path = path.with_suffix('.bm25.pkl')
            with open(bm25_path, 'wb') as f:
                pickle.dump(self.tokenized_corpus, f)
            logger.info(f"Saved BM25 data to {bm25_path}")
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """
        Load index from disk
        
        Args:
            path: Path to load index from (without extension)
        """
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix('.faiss')))
        
        # Load metadata and mappings
        with open(path.with_suffix('.json'), 'r') as f:
            data = json.load(f)
        
        self.metadata = data["metadata"]
        self.chunk_to_frame = {int(k): v for k, v in data["chunk_to_frame"].items()}
        self.frame_to_chunks = {int(k): v for k, v in data["frame_to_chunks"].items()}
        
        # Update config if available
        if "config" in data:
            self.config.update(data["config"])
        
        # Load BM25 data if available
        bm25_path = path.with_suffix('.bm25.pkl')
        if bm25_path.exists() and BM25_AVAILABLE:
            try:
                with open(bm25_path, 'rb') as f:
                    self.tokenized_corpus = pickle.load(f)
                self.bm25_index = BM25Okapi(self.tokenized_corpus)
                logger.info(f"Loaded BM25 index from {bm25_path}")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
                self.bm25_index = None
        elif data.get("bm25_enabled") and self.metadata:
            # Rebuild BM25 from metadata if pkl not found
            logger.info("Rebuilding BM25 index from metadata...")
            all_texts = [m["text"] for m in self.metadata]
            self._build_bm25_index(all_texts)
        
        logger.info(f"Loaded index from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_chunks": len(self.metadata),
            "total_frames": len(self.frame_to_chunks),
            "index_type": self.config["index"]["type"],
            "embedding_model": self.config["embedding"]["model"],
            "dimension": self.dimension,
            "avg_chunks_per_frame": np.mean([len(chunks) for chunks in self.frame_to_chunks.values()]) if self.frame_to_chunks else 0,
            "bm25_enabled": self.bm25_index is not None,
            "search_modes_available": self._get_available_search_modes()
        }
    
    def _get_available_search_modes(self) -> List[str]:
        """Get list of available search modes"""
        modes = ["vector"]  # Always available
        if self.bm25_index:
            modes.extend(["bm25", "hybrid"])
        return modes
