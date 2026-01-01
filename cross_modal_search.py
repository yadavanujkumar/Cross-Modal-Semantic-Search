#!/usr/bin/env python3
"""
Cross-Modal Semantic Search Engine using OpenAI's CLIP Model

This script demonstrates a complete pipeline for searching images using natural language queries.
It uses CLIP (Contrastive Language-Image Pre-training) to encode both images and text into
a shared embedding space, enabling semantic search without metadata or tags.
"""

import os
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for environments without display
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
import urllib.request
from io import BytesIO


class CrossModalSearchEngine:
    """
    A Cross-Modal Semantic Search Engine that enables text-to-image search
    using OpenAI's CLIP model and cosine similarity.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the search engine with CLIP model.
        
        Args:
            model_name: CLIP model architecture (default: ViT-B/32)
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model: {model_name} on device: {self.device}")
        
        # Load CLIP model and preprocessing
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Storage for embeddings and metadata
        self.image_embeddings = None
        self.image_paths = []
        self.images = []
        
        print(f"Model loaded successfully. Embedding dimension: 512")
    
    def load_image(self, image_source: str) -> Image.Image:
        """
        Load an image from URL or local path.
        
        Args:
            image_source: URL or local file path
            
        Returns:
            PIL Image object
        """
        try:
            if image_source.startswith(('http://', 'https://')):
                # Load from URL
                with urllib.request.urlopen(image_source) as response:
                    image_data = response.read()
                image = Image.open(BytesIO(image_data)).convert('RGB')
            else:
                # Load from local path
                image = Image.open(image_source).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image from {image_source}: {e}")
            return None
    
    def encode_images(self, image_sources: List[str]) -> Tuple[np.ndarray, List[Image.Image], List[str]]:
        """
        Encode a list of images into 512-dimensional embeddings.
        
        Args:
            image_sources: List of image URLs or local paths
            
        Returns:
            Tuple containing:
                - NumPy array of shape (n_images, 512) containing image embeddings
                - List of successfully loaded PIL Image objects
                - List of corresponding image paths/URLs
        """
        print(f"\nEncoding {len(image_sources)} images...")
        embeddings = []
        valid_images = []
        valid_paths = []
        
        for idx, image_source in enumerate(image_sources):
            print(f"Processing image {idx + 1}/{len(image_sources)}: {image_source}")
            
            # Load image
            image = self.load_image(image_source)
            if image is None:
                continue
            
            # Preprocess and encode
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(image_features.cpu().numpy())
            valid_images.append(image)
            valid_paths.append(image_source)
        
        if not embeddings:
            raise ValueError("No valid images were loaded")
        
        # Stack all embeddings into a single array
        embeddings_array = np.vstack(embeddings)
        print(f"Successfully encoded {len(embeddings)} images")
        print(f"Embedding shape: {embeddings_array.shape}")
        
        return embeddings_array, valid_images, valid_paths
    
    def build_index(self, image_sources: List[str]):
        """
        Build the image index by encoding all images and storing embeddings.
        
        Args:
            image_sources: List of image URLs or local paths to index
        """
        print("\n" + "="*60)
        print("BUILDING IMAGE INDEX")
        print("="*60)
        
        self.image_embeddings, self.images, self.image_paths = self.encode_images(image_sources)
        
        print(f"\nIndex built successfully!")
        print(f"Total images indexed: {len(self.image_paths)}")
        print(f"Embedding dimension: {self.image_embeddings.shape[1]}")
    
    def encode_text(self, text_query: str) -> np.ndarray:
        """
        Encode a text query into a 512-dimensional embedding.
        
        Args:
            text_query: Natural language search query
            
        Returns:
            NumPy array of shape (1, 512) containing text embedding
        """
        # Tokenize and encode text
        text_input = clip.tokenize([text_query]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def search(self, text_query: str, top_k: int = 1) -> List[Tuple[int, float]]:
        """
        Search for images matching the text query using cosine similarity.
        
        Args:
            text_query: Natural language search query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (image_index, similarity_score)
        """
        if self.image_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        print("\n" + "="*60)
        print("SEARCHING")
        print("="*60)
        print(f"Query: '{text_query}'")
        
        # Encode text query
        text_embedding = self.encode_text(text_query)
        
        # Compute cosine similarity between query and all images
        similarities = cosine_similarity(text_embedding, self.image_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices]
        
        print(f"\nTop {top_k} results:")
        for rank, (idx, score) in enumerate(results, 1):
            print(f"  {rank}. Image: {self.image_paths[idx]}")
            print(f"     Similarity: {score:.4f}")
        
        return results
    
    def display_results(self, text_query: str, results: List[Tuple[int, float]]):
        """
        Display search results with images and similarity scores.
        
        Args:
            text_query: The search query used
            results: List of tuples (image_index, similarity_score)
        """
        n_results = len(results)
        fig, axes = plt.subplots(1, n_results, figsize=(5 * n_results, 5))
        
        if n_results == 1:
            axes = [axes]
        
        fig.suptitle(f'Search Results for: "{text_query}"', fontsize=16, fontweight='bold')
        
        for ax, (idx, score) in zip(axes, results):
            ax.imshow(self.images[idx])
            ax.axis('off')
            ax.set_title(f'Similarity: {score:.4f}\n{os.path.basename(self.image_paths[idx])}', 
                        fontsize=12)
        
        plt.tight_layout()
        plt.savefig('search_results.png', dpi=150, bbox_inches='tight')
        print("\nResults saved to 'search_results.png'")
        plt.close()  # Close the figure to free memory


def main():
    """
    Main demonstration function showing the complete pipeline.
    """
    print("="*60)
    print("CROSS-MODAL SEMANTIC SEARCH ENGINE DEMO")
    print("Using OpenAI's CLIP Model (ViT-B/32)")
    print("="*60)
    
    # Initialize the search engine
    search_engine = CrossModalSearchEngine(model_name="ViT-B/32")
    
    # Sample image URLs for demonstration
    # These are diverse images from various categories
    sample_images = [
        "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=500",  # Dog playing
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=500",  # Cat
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500",  # Mountain landscape
        "https://images.unsplash.com/photo-1519681393784-d120267933ba?w=500",  # Mountain/nature
        "https://images.unsplash.com/photo-1511367461989-f85a21fda167?w=500",  # Beach/ocean
        "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500",  # Shoes/sneakers
        "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=500",  # Food
        "https://images.unsplash.com/photo-1525547719571-a2d4ac8945e2?w=500",  # Laptop/computer
    ]
    
    # Build the image index
    search_engine.build_index(sample_images)
    
    # Example search queries
    queries = [
        "A dog playing in the grass",
        "A cat sitting",
        "Mountain landscape",
        "Beach and ocean",
        "Technology and computers"
    ]
    
    print("\n" + "="*60)
    print("RUNNING EXAMPLE QUERIES")
    print("="*60)
    
    # Perform searches
    for query in queries:
        results = search_engine.search(query, top_k=3)
        print()  # Add spacing between queries
    
    # Display the best result for the first query
    print("\n" + "="*60)
    print("DISPLAYING TOP RESULT FOR FIRST QUERY")
    print("="*60)
    
    results = search_engine.search(queries[0], top_k=1)
    search_engine.display_results(queries[0], results)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nThe system can now:")
    print("  ✓ Encode images into 512-dimensional vectors")
    print("  ✓ Build a searchable index (simulating a Vector DB)")
    print("  ✓ Encode text queries into the same embedding space")
    print("  ✓ Perform semantic search using cosine similarity")
    print("  ✓ Return and display relevant results")


if __name__ == "__main__":
    main()
