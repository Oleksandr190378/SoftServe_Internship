"""
Validate ground_truth.json against indexed documents and images.
Checks if all referenced doc_ids and image_ids actually exist in the system.

Usage:
    python eval/validate_ground_truth.py
    python eval/validate_ground_truth.py --ground-truth eval/ground_truth.json
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Default paths
DEFAULT_GROUND_TRUTH = 'eval/ground_truth.json'
DEFAULT_DOCS = 'data/processed_docs.json'
DEFAULT_IMAGES = 'data/processed/images_metadata.json'


def load_json(filepath: str) -> dict:
    """Load JSON file with error handling."""
    try:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filepath}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error loading {filepath}: {e}")
        sys.exit(1)

def validate_query_structure(query: dict, query_idx: int) -> List[str]:
    """
    Validate query has all required fields and correct types.
    
    Args:
        query: Query dict to validate
        query_idx: Index for error reporting
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    required_fields = {
        'id': int,
        'query': str,
        'query_type': str,
        'category': str,
        'relevant_docs': list,
        'relevant_images': list
    }
    
    for field, expected_type in required_fields.items():
        if field not in query:
            errors.append(f"Query {query_idx}: Missing required field '{field}'")
        elif not isinstance(query[field], expected_type):
            actual_type = type(query[field]).__name__
            errors.append(f"Query {query_idx}: Field '{field}' should be {expected_type.__name__}, got {actual_type}")
    
    # Validate query_type values
    if 'query_type' in query:
        valid_types = ['text_focused', 'visual', 'hybrid']
        if query['query_type'] not in valid_types:
            errors.append(f"Query {query['id']}: Invalid query_type '{query['query_type']}'. Must be one of {valid_types}")
    
    # Validate relevant_docs is list of strings
    if 'relevant_docs' in query and isinstance(query['relevant_docs'], list):
        for i, doc in enumerate(query['relevant_docs']):
            if not isinstance(doc, str):
                errors.append(f"Query {query['id']}: relevant_docs[{i}] should be string, got {type(doc).__name__}")
    
    # Validate relevant_images is list of strings
    if 'relevant_images' in query and isinstance(query['relevant_images'], list):
        for i, img in enumerate(query['relevant_images']):
            if not isinstance(img, str):
                errors.append(f"Query {query['id']}: relevant_images[{i}] should be string, got {type(img).__name__}")
    
    return errors


def validate_ground_truth_structure(gt: dict) -> List[str]:
    """
    Validate ground truth file structure.
    
    Args:
        gt: Ground truth dict
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check top-level structure
    if 'queries' not in gt:
        errors.append("Ground truth missing 'queries' key")
        return errors  # Can't continue without queries
    
    if not isinstance(gt['queries'], list):
        errors.append(f"Ground truth 'queries' should be list, got {type(gt['queries']).__name__}")
        return errors
    
    if len(gt['queries']) == 0:
        errors.append("Ground truth 'queries' is empty")
        return errors
    
    # Validate each query
    unique_ids = set()
    for idx, query in enumerate(gt['queries']):
        struct_errors = validate_query_structure(query, idx)
        errors.extend(struct_errors)
        
        # Check for duplicate IDs
        if 'id' in query:
            if query['id'] in unique_ids:
                errors.append(f"Duplicate query ID: {query['id']}")
            unique_ids.add(query['id'])
    
    return errors


def validate_document_references(gt: dict, doc_ids: Set[str], image_ids: Set[str]) -> Tuple[List[str], Dict]:
    """
    Validate that all referenced documents and images exist.
    
    Args:
        gt: Ground truth dict
        doc_ids: Set of available document IDs
        image_ids: Set of available image IDs
    
    Returns:
        Tuple of (error_list, stats_dict)
    """
    errors = []
    stats = {
        'missing_docs': {},
        'missing_images': {},
        'queries_checked': 0,
        'queries_with_missing_refs': 0
    }
    
    for query in gt['queries']:
        query_id = query.get('id', 'unknown')
        query_has_errors = False
        
        stats['queries_checked'] += 1
        
        # Check documents
        for doc_id in query.get('relevant_docs', []):
            if doc_id not in doc_ids:
                error = f"Query {query_id}: Document not found: '{doc_id}'"
                errors.append(error)
                query_has_errors = True
                
                # Track missing docs
                if doc_id not in stats['missing_docs']:
                    stats['missing_docs'][doc_id] = []
                stats['missing_docs'][doc_id].append(query_id)
        
        # Check images
        for img_id in query.get('relevant_images', []):
            if img_id not in image_ids:
                error = f"Query {query_id}: Image not found: '{img_id}'"
                errors.append(error)
                query_has_errors = True
                
                # Track missing images
                if img_id not in stats['missing_images']:
                    stats['missing_images'][img_id] = []
                stats['missing_images'][img_id].append(query_id)
        
        if query_has_errors:
            stats['queries_with_missing_refs'] += 1
    
    return errors, stats


def validate_image_integrity(images: list, doc_ids: Set[str]) -> List[str]:
    """
    Validate that all images have correct document references.
    
    Args:
        images: List of image metadata dicts
        doc_ids: Set of available document IDs
    
    Returns:
        List of error messages
    """
    errors = []
    
    for idx, img in enumerate(images):
        # Check required image fields
        if 'image_id' not in img:
            errors.append(f"Image[{idx}]: Missing 'image_id'")
            continue
        
        if 'doc_id' not in img:
            errors.append(f"Image '{img.get('image_id')}': Missing 'doc_id'")
            continue
        
        # Check if doc exists
        doc_id = img['doc_id']
        if doc_id not in doc_ids:
            errors.append(f"Image '{img['image_id']}': Referenced doc_id '{doc_id}' does not exist")
    
    return errors


def validate_ground_truth(gt_path: str, docs_path: str, images_path: str) -> bool:
    """Validate ground truth annotations."""
    
    # Load files
    print("Loading files...")
    gt = load_json(gt_path)
    docs = load_json(docs_path)
    images = load_json(images_path)
    
    # Create sets for fast lookup
    doc_ids = set(docs.keys())
    image_ids = {img['image_id'] for img in images if 'image_id' in img}
    
    print('=' * 80)
    print('GROUND TRUTH VALIDATION')
    print('=' * 80)
    print(f'\nInput files:')
    print(f'  - Ground truth: {gt_path}')
    print(f'  - Documents: {docs_path}')
    print(f'  - Images: {images_path}')
    
    print(f'\nIndexed content:')
    print(f'  - Total documents: {len(doc_ids)}')
    print(f'  - Total images: {len(image_ids)}')
    print(f'  - Total queries: {len(gt.get("queries", []))}')
    print('\n' + '-' * 80)
    
    all_errors = []
    
    # 1. Validate ground truth structure
    print('\n🔍 Step 1: Validating ground truth structure...')
    struct_errors = validate_ground_truth_structure(gt)
    if struct_errors:
        print(f"❌ Structure validation FAILED ({len(struct_errors)} errors):")
        for err in struct_errors:
            print(f"  - {err}")
        all_errors.extend(struct_errors)
    else:
        print("✅ Ground truth structure is valid")
    
    # 2. Validate image integrity
    print('\n🔍 Step 2: Validating image metadata...')
    image_errors = validate_image_integrity(images, doc_ids)
    if image_errors:
        print(f"❌ Image validation FAILED ({len(image_errors)} errors):")
        for err in image_errors:
            print(f"  - {err}")
        all_errors.extend(image_errors)
    else:
        print("✅ All images have valid metadata")
    
    # 3. Validate document references
    print('\n🔍 Step 3: Validating document and image references...')
    ref_errors, stats = validate_document_references(gt, doc_ids, image_ids)
    if ref_errors:
        print(f"❌ Reference validation FAILED ({len(ref_errors)} errors):")
        for err in ref_errors[:10]:  # Show first 10
            print(f"  - {err}")
        if len(ref_errors) > 10:
            print(f"  ... and {len(ref_errors) - 10} more errors")
        all_errors.extend(ref_errors)
    else:
        print("✅ All document and image references are valid")
    
    print('\n' + '=' * 80)
    print('VALIDATION SUMMARY')
    print('=' * 80)
    
    # Collect statistics from queries
    if 'queries' in gt:
        unique_docs = set(doc for q in gt['queries'] for doc in q.get('relevant_docs', []))
        unique_images = set(img for q in gt['queries'] for img in q.get('relevant_images', []))
        queries_with_images = sum(1 for q in gt['queries'] if q.get('relevant_images'))
        text_only = sum(1 for q in gt['queries'] if not q.get('relevant_images'))
        
        print(f'\n📊 Ground Truth Statistics:')
        print(f'  - Total queries: {len(gt["queries"])}')
        print(f'  - Unique documents referenced: {len(unique_docs)}')
        print(f'  - Unique images referenced: {len(unique_images)}')
        print(f'  - Queries with images: {queries_with_images}')
        print(f'  - Text-only queries: {text_only}')
        
        # Query type distribution
        type_dist = {}
        for q in gt['queries']:
            qtype = q.get('query_type', 'unknown')
            type_dist[qtype] = type_dist.get(qtype, 0) + 1
        
        print(f'\n📈 Query Type Distribution:')
        for qtype in sorted(type_dist.keys()):
            count = type_dist[qtype]
            print(f'  - {qtype}: {count}')
    
    # Missing references stats
    if stats['missing_docs']:
        print(f'\n⚠️  Missing Documents: {len(stats["missing_docs"])}')
        for doc_id, query_ids in sorted(stats['missing_docs'].items())[:5]:
            print(f'  - {doc_id}: referenced by {len(query_ids)} queries')
    
    if stats['missing_images']:
        print(f'\n⚠️  Missing Images: {len(stats["missing_images"])}')
        for img_id, query_ids in sorted(stats['missing_images'].items())[:5]:
            print(f'  - {img_id}: referenced by {len(query_ids)} queries')
    
    # Final result
    print('\n' + '=' * 80)
    if all_errors:
        print(f'❌ VALIDATION FAILED: {len(all_errors)} error(s) found')
        print('=' * 80)
        return False
    else:
        print('✅ VALIDATION PASSED: All checks successful!')
        print('=' * 80)
        return True


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Validate ground_truth.json against indexed documents and images'
    )
    parser.add_argument(
        '--ground-truth',
        default=DEFAULT_GROUND_TRUTH,
        help=f'Path to ground_truth.json (default: {DEFAULT_GROUND_TRUTH})'
    )
    parser.add_argument(
        '--docs',
        default=DEFAULT_DOCS,
        help=f'Path to processed_docs.json (default: {DEFAULT_DOCS})'
    )
    parser.add_argument(
        '--images',
        default=DEFAULT_IMAGES,
        help=f'Path to images_metadata.json (default: {DEFAULT_IMAGES})'
    )
    
    args = parser.parse_args()
    
    success = validate_ground_truth(
        gt_path=args.ground_truth,
        docs_path=args.docs,
        images_path=args.images
    )
    
    sys.exit(0 if success else 1)



if __name__ == '__main__':
    main()
