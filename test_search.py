#!/usr/bin/env python3
"""
Quick test script for Cross-Modal Semantic Search Engine
Creates simple synthetic test images instead of downloading from URLs
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_images():
    """Create simple test images with different colors/patterns"""
    
    # Create a test_images directory
    os.makedirs('test_images', exist_ok=True)
    
    images = []
    
    # Image 1: Red image (could represent a dog)
    img1 = Image.new('RGB', (300, 300), color='red')
    draw = ImageDraw.Draw(img1)
    draw.text((100, 140), "DOG", fill='white')
    img1_path = 'test_images/red_image.jpg'
    img1.save(img1_path)
    images.append(img1_path)
    print(f"Created {img1_path}")
    
    # Image 2: Blue image (could represent water/ocean)
    img2 = Image.new('RGB', (300, 300), color='blue')
    draw = ImageDraw.Draw(img2)
    draw.text((80, 140), "OCEAN", fill='white')
    img2_path = 'test_images/blue_image.jpg'
    img2.save(img2_path)
    images.append(img2_path)
    print(f"Created {img2_path}")
    
    # Image 3: Green image (could represent grass/nature)
    img3 = Image.new('RGB', (300, 300), color='green')
    draw = ImageDraw.Draw(img3)
    draw.text((80, 140), "GRASS", fill='white')
    img3_path = 'test_images/green_image.jpg'
    img3.save(img3_path)
    images.append(img3_path)
    print(f"Created {img3_path}")
    
    # Image 4: Yellow image (could represent sun/beach)
    img4 = Image.new('RGB', (300, 300), color='yellow')
    draw = ImageDraw.Draw(img4)
    draw.text((80, 140), "BEACH", fill='black')
    img4_path = 'test_images/yellow_image.jpg'
    img4.save(img4_path)
    images.append(img4_path)
    print(f"Created {img4_path}")
    
    # Image 5: Gray image (could represent technology)
    img5 = Image.new('RGB', (300, 300), color='gray')
    draw = ImageDraw.Draw(img5)
    draw.text((50, 140), "COMPUTER", fill='white')
    img5_path = 'test_images/gray_image.jpg'
    img5.save(img5_path)
    images.append(img5_path)
    print(f"Created {img5_path}")
    
    return images

def main():
    """Test the Cross-Modal Search Engine with synthetic images"""
    
    # Import after matplotlib backend is set
    from cross_modal_search import CrossModalSearchEngine
    
    print("="*60)
    print("CROSS-MODAL SEMANTIC SEARCH ENGINE - QUICK TEST")
    print("="*60)
    
    # Create test images
    print("\nCreating test images...")
    test_images = create_test_images()
    
    # Initialize search engine
    print("\nInitializing search engine...")
    engine = CrossModalSearchEngine(model_name="ViT-B/32")
    
    # Build index
    print("\nBuilding index...")
    engine.build_index(test_images)
    
    # Test queries
    queries = [
        "grass and nature",
        "water and ocean",
        "technology and computer",
        "red color",
        "beach and sand"
    ]
    
    print("\n" + "="*60)
    print("TESTING QUERIES")
    print("="*60)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = engine.search(query, top_k=2)
        print()
    
    # Display top result for first query
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    results = engine.search(queries[0], top_k=1)
    engine.display_results(queries[0], results)
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nTest images created in: {os.path.abspath('test_images')}")
    print(f"Results visualization: {os.path.abspath('search_results.png')}")

if __name__ == "__main__":
    main()
