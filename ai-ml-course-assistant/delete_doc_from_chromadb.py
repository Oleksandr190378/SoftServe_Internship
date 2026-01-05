"""
Delete specific document from ChromaDB collections.

Usage: python delete_doc_from_chromadb.py arxiv_1409_3215
"""

import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path("data/chroma_db")

def delete_document(doc_id: str):
    """Delete all chunks and images for a document from ChromaDB."""
    
    print(f"\n🗑️  Deleting document: {doc_id}")
    print("=" * 60)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    # Delete from text_chunks
    print("\n📄 Text Chunks Collection:")
    try:
        text_store = Chroma(
            collection_name="text_chunks",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR / "text_chunks")
        )
        
        # Get all IDs for this document
        collection = text_store._collection
        all_data = collection.get(where={"doc_id": doc_id})
        chunk_ids = all_data["ids"]
        
        if chunk_ids:
            print(f"  Found {len(chunk_ids)} chunks to delete")
            collection.delete(ids=chunk_ids)
            print(f"  ✅ Deleted {len(chunk_ids)} chunks")
        else:
            print(f"  ℹ️  No chunks found for {doc_id}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Delete from image_captions
    print("\n🖼️  Image Captions Collection:")
    try:
        image_store = Chroma(
            collection_name="image_captions",
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR / "image_captions")
        )
        
        # Get all IDs for this document
        collection = image_store._collection
        all_data = collection.get(where={"doc_id": doc_id})
        image_ids = all_data["ids"]
        
        if image_ids:
            print(f"  Found {len(image_ids)} images to delete")
            collection.delete(ids=image_ids)
            print(f"  ✅ Deleted {len(image_ids)} images")
        else:
            print(f"  ℹ️  No images found for {doc_id}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Deletion complete!")
    print("\nNext step: Re-run pipeline to re-index with new metadata:")
    print(f"  python run_pipeline.py process --doc-id {doc_id} --force")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_doc_from_chromadb.py <doc_id>")
        print("Example: python delete_doc_from_chromadb.py arxiv_1409_3215")
        sys.exit(1)
    
    doc_id = sys.argv[1]
    delete_document(doc_id)
