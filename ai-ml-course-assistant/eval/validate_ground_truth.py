"""
Validate ground_truth.json against indexed documents and images.
Checks if all referenced doc_ids and image_ids actually exist in the system.
"""

import json
from pathlib import Path
from typing import List, Dict, Set

def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_ground_truth():
    """Validate ground truth annotations."""
    
    # Load files
    print("Loading files...")
    gt = load_json('eval/ground_truth.json')
    docs = load_json('data/processed_docs.json')
    images = load_json('data/processed/images_metadata.json')
    
    # Create sets for fast lookup
    doc_ids = set(docs.keys())
    image_ids = {img['image_id'] for img in images}
    
    print('=' * 80)
    print('GROUND TRUTH VALIDATION')
    print('=' * 80)
    print(f'\nTotal queries: {len(gt["queries"])}')
    print(f'Indexed documents: {len(doc_ids)}')
    print(f'Indexed images: {len(image_ids)}')
    print('\n' + '-' * 80)
    
    # Validate each query
    errors = []
    warnings = []
    
    for query in gt['queries']:
        print(f'\n[Query {query["id"]}] {query["query"]}')
        print(f'  Type: {query["query_type"]} | Category: {query["category"]}')
        
        # Check documents
        for doc_id in query['relevant_docs']:
            if doc_id in doc_ids:
                print(f'  ✅ Doc: {doc_id}')
            else:
                error = f'Doc NOT FOUND: {doc_id}'
                print(f'  ❌ {error}')
                errors.append(f'Query {query["id"]}: {error}')
        
        # Check images
        if query['relevant_images']:
            for img_id in query['relevant_images']:
                if img_id in image_ids:
                    print(f'  ✅ Image: {img_id}')
                else:
                    error = f'Image NOT FOUND: {img_id}'
                    print(f'  ❌ {error}')
                    errors.append(f'Query {query["id"]}: {error}')
        else:
            print(f'  ℹ️  No images expected')
    
    print('\n' + '=' * 80)
    print('VALIDATION SUMMARY')
    print('=' * 80)
    
    if errors:
        print(f'\n❌ ERRORS ({len(errors)}):')
        for err in errors:
            print(f'  - {err}')
    else:
        print('\n✅ All documents and images are valid!')
    
    if warnings:
        print(f'\n⚠️  WARNINGS ({len(warnings)}):')
        for warn in warnings:
            print(f'  - {warn}')
    
    # Stats
    unique_docs = set(doc for q in gt['queries'] for doc in q['relevant_docs'])
    unique_images = set(img for q in gt['queries'] for img in q['relevant_images'])
    queries_with_images = sum(1 for q in gt['queries'] if q['relevant_images'])
    text_only = sum(1 for q in gt['queries'] if not q['relevant_images'])
    
    print(f'\n📊 Stats:')
    print(f'  - Unique docs referenced: {len(unique_docs)}')
    print(f'  - Unique images referenced: {len(unique_images)}')
    print(f'  - Queries with images: {queries_with_images}')
    print(f'  - Text-only queries: {text_only}')
    
    # Query type distribution
    type_dist = {}
    for q in gt['queries']:
        qtype = q['query_type']
        type_dist[qtype] = type_dist.get(qtype, 0) + 1
    
    print(f'\n📈 Query Type Distribution:')
    for qtype, count in type_dist.items():
        print(f'  - {qtype}: {count}')
    
    print('\n' + '=' * 80)
    
    return len(errors) == 0

if __name__ == '__main__':
    success = validate_ground_truth()
    exit(0 if success else 1)
