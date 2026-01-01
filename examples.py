#!/usr/bin/env python3
"""
Example Usage of Cross-Modal Semantic Search Engine

This script demonstrates how to use the search engine once the CLIP model is available.
The model requires network access to download on first use (~350MB).

For offline usage, ensure the model is pre-downloaded to ~/.cache/clip/
"""

import os
import sys

def example_basic_usage():
    """Example 1: Basic usage with local images"""
    
    print("="*70)
    print("EXAMPLE 1: Basic Usage with Local Images")
    print("="*70)
    
    code = '''
from cross_modal_search import CrossModalSearchEngine

# Initialize the search engine
engine = CrossModalSearchEngine(model_name="ViT-B/32")

# Index your local images
image_paths = [
    "/path/to/image1.jpg",
    "/path/to/image2.jpg",
    "/path/to/image3.jpg",
]
engine.build_index(image_paths)

# Search with a text query
results = engine.search("a cute dog playing", top_k=3)

# Display the top result
engine.display_results("a cute dog playing", results)
'''
    print(code)


def example_url_images():
    """Example 2: Usage with image URLs"""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Usage with Image URLs")
    print("="*70)
    
    code = '''
from cross_modal_search import CrossModalSearchEngine

# Initialize the search engine
engine = CrossModalSearchEngine()

# Index images from URLs
image_urls = [
    "https://example.com/dog.jpg",
    "https://example.com/cat.jpg",
    "https://example.com/beach.jpg",
]
engine.build_index(image_urls)

# Search
results = engine.search("ocean and waves", top_k=1)
print(f"Best match: {engine.image_paths[results[0][0]]}")
print(f"Similarity: {results[0][1]:.4f}")
'''
    print(code)


def example_custom_queries():
    """Example 3: Multiple queries on same index"""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple Queries on Same Index")
    print("="*70)
    
    code = '''
from cross_modal_search import CrossModalSearchEngine

engine = CrossModalSearchEngine()
engine.build_index(["img1.jpg", "img2.jpg", "img3.jpg"])

# Run multiple queries
queries = [
    "a dog playing in grass",
    "mountain landscape",
    "technology and computers",
    "food on a plate"
]

for query in queries:
    results = engine.search(query, top_k=1)
    best_match = engine.image_paths[results[0][0]]
    score = results[0][1]
    print(f"Query: {query}")
    print(f"  → {best_match} (score: {score:.4f})")
'''
    print(code)


def example_class_usage():
    """Example 4: Understanding the output"""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Understanding the Output")
    print("="*70)
    
    explanation = '''
When you call engine.search(), you get back a list of tuples:
  [(image_index, similarity_score), ...]

Where:
- image_index: Index in the original image list (0-based)
- similarity_score: Cosine similarity between text and image (0-1 range)
  * 1.0 = perfect match
  * 0.0 = no similarity

Example output:
  Results for "a cat sitting":
    1. Image: /path/to/cat.jpg
       Similarity: 0.8532
    2. Image: /path/to/kitten.jpg
       Similarity: 0.7891
    3. Image: /path/to/dog.jpg
       Similarity: 0.4521
'''
    print(explanation)


def example_advanced():
    """Example 5: Advanced usage"""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Advanced Usage - Custom Device")
    print("="*70)
    
    code = '''
from cross_modal_search import CrossModalSearchEngine
import torch

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
engine = CrossModalSearchEngine(device=device)

# Large batch of images
image_paths = [f"image_{i}.jpg" for i in range(1000)]
engine.build_index(image_paths)

# Fast searching
for query in ["nature", "technology", "food", "animals"]:
    results = engine.search(query, top_k=5)
    print(f"Top 5 results for '{query}':")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"  {rank}. {engine.image_paths[idx]} ({score:.4f})")
'''
    print(code)


def main():
    """Display all examples"""
    
    print("\n" + "="*70)
    print("CROSS-MODAL SEMANTIC SEARCH ENGINE")
    print("Usage Examples and Documentation")
    print("="*70)
    
    print("\n" + "⚠️  NOTE: These examples require the CLIP model to be downloaded.")
    print("On first run, the model (~350MB) will be downloaded to ~/.cache/clip/")
    print("Ensure you have internet connectivity for the initial setup.\n")
    
    example_basic_usage()
    example_url_images()
    example_custom_queries()
    example_class_usage()
    example_advanced()
    
    print("\n" + "="*70)
    print("HOW IT WORKS")
    print("="*70)
    
    explanation = '''
The Cross-Modal Semantic Search Engine works by:

1. **Embedding Generation**:
   - Images → CLIP Image Encoder → 512-dimensional vectors
   - Text → CLIP Text Encoder → 512-dimensional vectors
   - Both are mapped to the same semantic space

2. **Indexing**:
   - All image embeddings are stored in a NumPy array
   - This simulates a vector database (like Pinecone, Weaviate)

3. **Searching**:
   - Query text is encoded to a vector
   - Cosine similarity is computed with all image vectors
   - Results are ranked by similarity score

4. **Why it works**:
   - CLIP was trained on 400M image-text pairs
   - It learned to align visual and textual concepts
   - No metadata or tags needed - semantic understanding!
'''
    print(explanation)
    
    print("\n" + "="*70)
    print("SYSTEM REQUIREMENTS")
    print("="*70)
    
    requirements = '''
- Python 3.7+
- PyTorch 2.0+
- CLIP (OpenAI)
- Pillow, scikit-learn, matplotlib, numpy
- ~350MB disk space for CLIP model
- Internet connection (first run only)
- Optional: CUDA-compatible GPU for faster processing
'''
    print(requirements)
    
    print("\n" + "="*70)
    print("For more information, see README.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
