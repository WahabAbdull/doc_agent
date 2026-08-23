import re
from typing import List, Dict, Any, Tuple
from document_loader import LoadedDocument

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    np = None


class DocumentChunk:
    """Represents a discrete searchable chunk of text with source metadata."""
    def __init__(self, chunk_id: str, doc_name: str, section_label: str, text: str, page_number: int = None):
        self.chunk_id = chunk_id
        self.doc_name = doc_name
        self.section_label = section_label
        self.text = text.strip()
        self.page_number = page_number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "section_label": self.section_label,
            "text": self.text,
            "page_number": self.page_number
        }


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks respecting sentence / paragraph breaks."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    
    # Split by double newlines (paragraphs) first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If single paragraph exceeds chunk size, split by sentences or punctuation
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.?!])\s+', para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if current_length + len(sentence) > chunk_size and current_chunk:
                    chunk_str = " ".join(current_chunk)
                    chunks.append(chunk_str)
                    # Overlap: keep last sentence
                    current_chunk = [current_chunk[-1]] if len(current_chunk) > 1 else []
                    current_length = sum(len(s) for s in current_chunk)
                current_chunk.append(sentence)
                current_length += len(sentence)
        else:
            if current_length + len(para) > chunk_size and current_chunk:
                chunk_str = "\n\n".join(current_chunk)
                chunks.append(chunk_str)
                # Overlap: keep last paragraph if short
                if len(current_chunk[-1]) < overlap:
                    current_chunk = [current_chunk[-1]]
                else:
                    current_chunk = []
                current_length = sum(len(p) for p in current_chunk)
                
            current_chunk.append(para)
            current_length += len(para)
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return [c.strip() for c in chunks if c.strip()]


class RAGEngine:
    """Manages document chunks, indexing, and precision context retrieval."""
    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents: Dict[str, LoadedDocument] = {}
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def add_documents(self, loaded_docs: List[LoadedDocument]):
        """Add or update loaded documents and re-index chunks."""
        self.documents = {doc.filename: doc for doc in loaded_docs}
        self._build_index()

    def _build_index(self):
        """Chunk all documents and build TF-IDF search index."""
        self.chunks = []
        chunk_counter = 0

        for doc in self.documents.values():
            for sec in doc.sections:
                sec_text = sec.get("text", "").strip()
                sec_label = sec.get("label", "Section")
                page_num = sec.get("page_number", None)
                
                if not sec_text:
                    continue
                
                sub_chunks = chunk_text(sec_text, self.chunk_size, self.overlap)
                for sub_text in sub_chunks:
                    chunk_counter += 1
                    chunk_id = f"chunk_{chunk_counter}"
                    self.chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_name=doc.filename,
                            section_label=sec_label,
                            text=sub_text,
                            page_number=page_num
                        )
                    )

        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        # Build TF-IDF index for fast keyword & semantic retrieval
        if TfidfVectorizer is not None:
            corpus = [c.text for c in self.chunks]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                lowercase=True,
                max_features=10000
            )
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            except Exception:
                self.vectorizer = None
                self.tfidf_matrix = None

    def retrieve(self, query: str, top_k: int = 6) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve most relevant chunks for a user query."""
        if not self.chunks:
            return []

        # If total chunks is small (e.g. <= 6), return all
        if len(self.chunks) <= top_k:
            return [(chunk, 1.0) for chunk in self.chunks]

        if self.vectorizer is not None and self.tfidf_matrix is not None:
            try:
                query_vec = self.vectorizer.transform([query])
                scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                
                # Get top indices
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = []
                for idx in top_indices:
                    score = float(scores[idx])
                    # Include if score has non-zero relevance or top results
                    results.append((self.chunks[idx], score))
                return results
            except Exception:
                pass

        # Fallback simple keyword match scoring
        query_words = set(re.findall(r'\w+', query.lower()))
        scored_chunks = []
        for chunk in self.chunks:
            chunk_words = set(re.findall(r'\w+', chunk.text.lower()))
            overlap = len(query_words.intersection(chunk_words))
            score = overlap / max(len(query_words), 1)
            scored_chunks.append((chunk, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def format_context(self, query: str, top_k: int = 8) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Formats retrieved chunks into an isolated, strict context block for the LLM.
        Returns: (formatted_context_string, list_of_cited_sources)
        """
        # If total content is within reasonable token limits (e.g. < 40,000 chars),
        # we can provide the full documents organized by file.
        total_text_len = sum(len(d.full_text) for d in self.documents.values())
        
        sources = []
        context_parts = []
        
        if total_text_len <= 35000:
            # Full context mode
            for doc in self.documents.values():
                context_parts.append(
                    f"================================================\n"
                    f"DOCUMENT: {doc.filename}\n"
                    f"================================================\n"
                    f"{doc.full_text}\n"
                )
                sources.append({
                    "doc_name": doc.filename,
                    "section_label": "Complete Document",
                    "snippet": doc.full_text[:250] + "..." if len(doc.full_text) > 250 else doc.full_text
                })
        else:
            # Chunk retrieval mode
            retrieved = self.retrieve(query, top_k=top_k)
            for chunk, score in retrieved:
                context_parts.append(
                    f"--- Source: [{chunk.doc_name} | {chunk.section_label}] ---\n"
                    f"{chunk.text}\n"
                )
                sources.append({
                    "doc_name": chunk.doc_name,
                    "section_label": chunk.section_label,
                    "snippet": chunk.text[:250] + "..." if len(chunk.text) > 250 else chunk.text,
                    "score": round(score, 3)
                })

        formatted_context = "\n\n".join(context_parts)
        return formatted_context, sources

    def get_summary_stats(self) -> Dict[str, Any]:
        """Returns statistics of indexed documents."""
        total_docs = len(self.documents)
        total_chunks = len(self.chunks)
        total_chars = sum(len(d.full_text) for d in self.documents.values())
        doc_details = []
        for name, doc in self.documents.items():
            doc_details.append({
                "name": name,
                "type": doc.metadata.get("type", "Unknown"),
                "chars": len(doc.full_text),
                "sections": len(doc.sections)
            })
        return {
            "total_docs": total_docs,
            "total_chunks": total_chunks,
            "total_chars": total_chars,
            "documents": doc_details
        }
