"""Collection of utilities for working with embeddings."""

from langchain_openai import OpenAIEmbeddings

from transformers import AutoTokenizer, AutoModel
import torch

from sentence_transformers import SentenceTransformer

def generate_embeddings(
        openai_api_key: str = None,
        chunks: list[str] = None,
        model: str = "text-embedding-3-small",
        model_name_or_path: str = None,
) -> list[list[float]]:
    """Generate embeddings for a list of chunks using either OpenAI or an open-source embedding model.

    Parameters
    ----------
    openai_api_key : str, optional
        OpenAI API key.

    chunks : list[str], optional
        List of chunks to generate embeddings for.

    model : str, optional
        OpenAI model to use for generating embeddings if `openai_api_key` is provided.

    model_name_or_path : str, optional
        Name or path of the pre-trained model to use if `openai_api_key` is not provided.

    Returns
    -------
    list[list[float]]
        List of embeddings for each chunk.

    """


    # Use open-source embedding model
    # Load pre-trained model and tokenizer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    return embeddings
