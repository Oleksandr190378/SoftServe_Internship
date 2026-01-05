"""Remove chunks arrays from processed_docs.json to reduce file size"""
import json
from pathlib import Path

registry_path = Path("data/processed_docs.json")

# Load registry
print("Loading registry...")
with open(registry_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Original file size: {registry_path.stat().st_size:,} bytes")
print(f"Documents: {len(data)}")

# Remove chunks arrays
chunks_removed = 0
for doc_id, doc_data in data.items():
    if 'chunks' in doc_data:
        num_chunks = len(doc_data['chunks'])
        del doc_data['chunks']
        chunks_removed += num_chunks
        print(f"  - {doc_id}: removed {num_chunks} chunks")

# Save cleaned registry
print("\nSaving cleaned registry...")
with open(registry_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

new_size = registry_path.stat().st_size
print(f"\n✅ Done!")
print(f"New file size: {new_size:,} bytes")
print(f"Reduction: {100 - (new_size / registry_path.stat().st_size * 100):.1f}%")
print(f"Total chunks removed: {chunks_removed}")
