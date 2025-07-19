# Qwen3-Embedding-8B

A Replicate Cog implementation of the Qwen3-Embedding-8B model - a state-of-the-art multilingual text embedding model with 8 billion parameters.

## Model Overview

- **Model**: Qwen3-Embedding-8B
- **Parameters**: 8 billion
- **Context Length**: 32,768 tokens
- **Embedding Dimensions**: 32-4096 (configurable)
- **Languages**: 100+ languages supported
- **Performance**: #1 on MTEB multilingual leaderboard (70.58 score)

## Features

- **Multilingual Support**: Excellent performance across 100+ languages
- **Instruction-Aware**: Supports task-specific instructions for improved embeddings
- **Flexible Dimensions**: Output embeddings from 32 to 4096 dimensions
- **Batch Processing**: Efficient processing of multiple texts
- **GPU Optimized**: Runs on CUDA-enabled GPUs for fast inference

## API Reference

### Input Parameters

| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|-------|
| `text` | `str` or `List[str]` | Text(s) to embed | Required | - |
| `instruction` | `str` | Optional task-specific instruction | `""` | - |
| `embedding_dim` | `int` | Output embedding dimension | `1024` | 32-4096 |
| `normalize` | `bool` | Whether to normalize embeddings | `True` | - |
| `batch_size` | `int` | Batch size for processing | `8` | 1-32 |

### Output Format

```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "shape": [2, 1024],
  "model": "Qwen3-Embedding-8B",
  "embedding_dim": 1024,
  "normalized": true,
  "instruction": "Represent this document for retrieval",
  "num_texts": 2
}
```

## Usage Examples

### Basic Text Embedding

```python
import replicate

# Single text embedding
output = replicate.run(
    "your-username/qwen3-embedding-8b",
    input={
        "text": "Hello, world! This is a sample text for embedding."
    }
)

print(f"Embedding shape: {output['shape']}")
print(f"First few values: {output['embeddings'][0][:5]}")
```

### Multiple Texts (Batch Processing)

```python
import replicate

# Batch embedding
texts = [
    "This is the first document about machine learning.",
    "The second document discusses natural language processing.",
    "A third document about computer vision and AI."
]

output = replicate.run(
    "your-username/qwen3-embedding-8b",
    input={
        "text": texts,
        "batch_size": 2
    }
)

print(f"Number of embeddings: {output['num_texts']}")
print(f"Shape of each embedding: {output['shape']}")
```

### Task-Specific Instructions

```python
import replicate

# Document retrieval task
output = replicate.run(
    "your-username/qwen3-embedding-8b",
    input={
        "text": "Python is a high-level programming language.",
        "instruction": "Represent this document for retrieval",
        "embedding_dim": 512
    }
)

# Query embedding for the same task
query_output = replicate.run(
    "your-username/qwen3-embedding-8b",
    input={
        "text": "What is Python programming?",
        "instruction": "Represent this query for retrieving relevant documents",
        "embedding_dim": 512
    }
)
```

### Multilingual Embedding

```python
import replicate

# Multiple languages
multilingual_texts = [
    "Hello, how are you?",           # English
    "Bonjour, comment allez-vous?",  # French
    "Hola, ¿cómo estás?",           # Spanish
    "你好，你好吗？",                 # Chinese
    "こんにちは、元気ですか？"          # Japanese
]

output = replicate.run(
    "your-username/qwen3-embedding-8b",
    input={
        "text": multilingual_texts,
        "instruction": "Represent this text for cross-lingual retrieval"
    }
)
```

### Semantic Search Example

```python
import replicate
import numpy as np

# Documents to search
documents = [
    "The cat sat on the mat.",
    "Machine learning is a subset of artificial intelligence.",
    "Paris is the capital of France.",
    "Deep learning uses neural networks with multiple layers."
]

# Get document embeddings
doc_output = replicate.run(
    "your-username/qwen3-embedding-8b",
    input={
        "text": documents,
        "instruction": "Represent this document for retrieval"
    }
)

# Query
query = "What is AI and machine learning?"
query_output = replicate.run(
    "your-username/qwen3-embedding-8b",
    input={
        "text": query,
        "instruction": "Represent this query for retrieving relevant documents"
    }
)

# Calculate similarities (cosine similarity)
doc_embeddings = np.array(doc_output['embeddings'])
query_embedding = np.array(query_output['embeddings'][0])

similarities = np.dot(doc_embeddings, query_embedding)
most_similar_idx = np.argmax(similarities)

print(f"Most similar document: {documents[most_similar_idx]}")
print(f"Similarity score: {similarities[most_similar_idx]:.4f}")
```

### Custom Embedding Dimensions

```python
import replicate

# Different embedding dimensions for different use cases
dimensions = [128, 256, 512, 1024, 2048]

text = "This is a test document for embedding dimension comparison."

for dim in dimensions:
    output = replicate.run(
        "your-username/qwen3-embedding-8b",
        input={
            "text": text,
            "embedding_dim": dim
        }
    )
    print(f"Dimension {dim}: Shape {output['shape']}")
```

## Common Use Cases

### 1. Document Retrieval
```python
instruction = "Represent this document for retrieval"
```

### 2. Question Answering
```python
# For documents
instruction = "Represent this document for answering questions"

# For queries  
instruction = "Represent this query for retrieving relevant documents"
```

### 3. Text Classification
```python
instruction = "Represent this text for classification"
```

### 4. Semantic Similarity
```python
instruction = "Represent this text for semantic similarity"
```

## Performance Tips

1. **Batch Processing**: Use batch processing for multiple texts to improve efficiency
2. **Appropriate Dimensions**: Choose embedding dimensions based on your use case:
   - 128-256: Fast similarity search
   - 512-1024: Balanced performance
   - 2048-4096: Maximum semantic richness
3. **Instructions**: Use task-specific instructions for better performance
4. **Normalization**: Keep normalization enabled for cosine similarity calculations

## Limitations

- Maximum context length: 32,768 tokens
- GPU memory requirements: ~16GB for optimal performance
- Processing time scales with text length and batch size

## Model Card

For detailed information about the model's capabilities, training data, and evaluation results, see the [official model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

## Citation

```bibtex
@article{qwen3-embedding,
  title={Qwen3-Embedding: A Multilingual Text Embedding Model},
  author={Qwen Team},
  year={2024},
  url={https://huggingface.co/Qwen/Qwen3-Embedding-8B}
}
```