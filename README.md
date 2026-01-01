# Cross-Modal Semantic Search Engine

A powerful Python-based semantic search engine that enables natural language text queries to search through image collections using OpenAI's CLIP (Contrastive Language-Image Pre-training) model. No metadata or tags required!

## 🌟 Features

- **Text-to-Image Search**: Search images using natural language queries
- **CLIP Model**: Leverages OpenAI's ViT-B/32 pre-trained model
- **512-Dimensional Embeddings**: Maps both images and text to the same semantic space
- **Cosine Similarity**: Efficient similarity computation for retrieval
- **Flexible Input**: Supports both URLs and local image paths
- **Visual Results**: Displays search results with similarity scores

## 🛠️ Technical Stack

- **PyTorch**: Deep learning framework
- **CLIP**: OpenAI's vision-language model
- **Pillow (PIL)**: Image processing
- **scikit-learn**: Cosine similarity computation
- **matplotlib**: Result visualization
- **NumPy**: Vector database simulation

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/Cross-Modal-Semantic-Search.git
cd Cross-Modal-Semantic-Search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The installation will download the CLIP model (~350MB) on first run.

## 🚀 Usage

### Basic Usage

Run the demonstration script:
```bash
python cross_modal_search.py
```

This will:
1. Load the CLIP ViT-B/32 model
2. Index sample images from URLs
3. Perform example searches
4. Display results

### Custom Usage

```python
from cross_modal_search import CrossModalSearchEngine

# Initialize the search engine
engine = CrossModalSearchEngine(model_name="ViT-B/32")

# Index your images (URLs or local paths)
image_sources = [
    "path/to/image1.jpg",
    "path/to/image2.jpg",
    "https://example.com/image3.jpg"
]
engine.build_index(image_sources)

# Search with text query
results = engine.search("a cute dog playing", top_k=3)

# Display results
engine.display_results("a cute dog playing", results)
```

## 🏗️ Architecture

### 1. Indexing Pipeline
- Load images from URLs or local paths
- Preprocess images using CLIP's preprocessing
- Encode images into 512-dimensional vectors
- Store embeddings in NumPy array (simulating Vector DB)

### 2. Query Pipeline
- Accept natural language text query
- Encode query into 512-dimensional vector
- Compute cosine similarity with all indexed images
- Return top-k most similar images

### 3. Retrieval Logic
- Uses cosine similarity for semantic matching
- Normalized embeddings ensure consistent similarity scores
- Results ranked by similarity score (0-1 range)

## 📊 How It Works

```
Text Query: "a dog playing in grass"
     |
     v
[CLIP Text Encoder] → 512-dim vector
     |
     v
[Cosine Similarity] ← 512-dim vectors ← [CLIP Image Encoder] ← Image Collection
     |
     v
Top-k Most Similar Images
```

## 🎯 Example Queries

- "A dog playing in the grass"
- "Mountain landscape with snow"
- "Beach with ocean waves"
- "Modern laptop computer"
- "Delicious food on a plate"

## 📝 Code Structure

```
cross_modal_search.py
├── CrossModalSearchEngine (Main class)
│   ├── __init__()           # Initialize CLIP model
│   ├── load_image()         # Load image from URL/path
│   ├── encode_images()      # Generate image embeddings
│   ├── build_index()        # Build searchable index
│   ├── encode_text()        # Generate text embeddings
│   ├── search()             # Perform similarity search
│   └── display_results()    # Visualize results
└── main()                   # Demo function
```

## 🔬 Technical Details

- **Model**: ViT-B/32 (Vision Transformer with 32x32 patch size)
- **Embedding Dimension**: 512
- **Similarity Metric**: Cosine Similarity
- **Normalization**: L2 normalization on all embeddings
- **Device Support**: Automatic GPU/CPU selection

## 🎓 Use Cases

- **E-commerce**: Search product catalogs with natural language
- **Digital Asset Management**: Find images in large collections
- **Content Moderation**: Identify specific content types
- **Research**: Analyze image datasets semantically
- **Creative Tools**: Find inspiration based on descriptions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- Unsplash for sample images in demo

## 📚 References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP Repository](https://github.com/openai/CLIP)