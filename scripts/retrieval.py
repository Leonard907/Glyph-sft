from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
import re
from typing import List


def chunk_doc(document: str, tokenizer_name: str = "Qwen/Qwen3-VL-8B-Instruct", chunk_size: int = 1024) -> List[str]:
    """
    Chunk a document into fixed-size token chunks with no overlap.
    
    Args:
        document: Input document text to chunk
        tokenizer_name: HuggingFace tokenizer name (default: "bert-base-uncased")
        chunk_size: Maximum number of tokens per chunk (default: 1024)
    
    Returns:
        List of document chunks as strings
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize the entire document
    tokens = tokenizer.encode(document, add_special_tokens=False)
    
    chunks = []
    # Split tokens into chunks of size chunk_size with no overlap
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        # Decode the chunk back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    
    return chunks


def retrieve(query: str, documents: List[str], topk: int = 5, mode: str = "bm25") -> List[str]:
    """
    Retrieve top-k documents based on a query using BM25.
    
    Args:
        query: Input query string
        documents: List of documents to search through
        topk: Number of top documents to retrieve (default: 5)
        mode: Retrieval mode, currently only "bm25" is supported (default: "bm25")
    
    Returns:
        List of top-k documents in the original order they appear in the input
    """
    if mode != "bm25":
        raise ValueError(f"Mode '{mode}' not supported. Currently only 'bm25' is supported.")
    
    if not documents:
        return []
    
    # Tokenize documents and query for BM25
    # BM25 works with tokenized text, so we'll use simple word tokenization
    def tokenize(text: str) -> List[str]:
        # Simple tokenization: split on whitespace and punctuation
        # Convert to lowercase for case-insensitive matching
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    # Tokenize all documents
    tokenized_docs = [tokenize(doc) for doc in documents]
    
    # Initialize BM25
    bm25 = BM25Okapi(tokenized_docs)
    
    # Tokenize query
    tokenized_query = tokenize(query)
    
    # Get BM25 scores for all documents
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k indices (sorted by score in descending order)
    topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    
    # Sort indices to maintain original order
    topk_indices_sorted = sorted(topk_indices)
    
    # Return documents in original order
    retrieved_docs = [documents[i] for i in topk_indices_sorted]
    
    return retrieved_docs

