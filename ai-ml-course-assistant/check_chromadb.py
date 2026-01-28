"""Quick ChromaDB health check."""
import chromadb
from pathlib import Path

CHROMA_DIR = Path("data/chroma_db")

def check_chromadb():
    """Check ChromaDB collections status."""
    print("🔍 ChromaDB Health Check\n" + "="*50)
    
    if not CHROMA_DIR.exists():
        print("❌ ChromaDB directory does not exist!")
        return
    
    try:
        # Initialize client
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        print(f"✅ Connected to ChromaDB at {CHROMA_DIR}\n")
        
        # List all collections
        collections = client.list_collections()
        print(f"📊 Found {len(collections)} collection(s):\n")
        
        for col in collections:
            print(f"Collection: {col.name}")
            print(f"  ID: {col.id}")
            
            try:
                # Try to count items (this will fail if index corrupted)
                count = col.count()
                print(f"  ✅ Items: {count}")
                
                # Try to peek (this also tests read operations)
                sample = col.peek(limit=1)
                print(f"  ✅ Can read data: {len(sample['ids'])} sample(s)")
                
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)[:100]}")
                print(f"     This collection is likely CORRUPTED")
            
            print()
        
        # Summary
        print("="*50)
        if len(collections) == 2:
            print("✅ Expected collections found (text_chunks, image_captions)")
        else:
            print(f"⚠️  Expected 2 collections, found {len(collections)}")
        
    except Exception as e:
        print(f"❌ FATAL ERROR connecting to ChromaDB:")
        print(f"   {str(e)}")
        print("\n🔧 Recommendation: ChromaDB is corrupted, needs re-indexing")

if __name__ == "__main__":
    check_chromadb()
