#!/usr/bin/env python3
"""
Find where ChromaDB actually stores the vectors
"""

import sqlite3
import numpy as np
import os
import glob

db_path = "../app_colette/mm_index/chroma.sqlite3"  # Change this to your database path

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 80)
print("COMPREHENSIVE VECTOR SEARCH")
print("=" * 80)

# Get collection info
cursor.execute("SELECT id, name, dimension FROM collections LIMIT 1;")
collection_id, collection_name, dimension = cursor.fetchone()

print(f"\nCollection: {collection_name}")
print(f"Collection ID: {collection_id}")
print(f"Expected dimension: {dimension}")

# Check all possible locations in the database
print("\n" + "=" * 80)
print("CHECKING ALL DATABASE TABLES FOR VECTORS")
print("=" * 80)

tables_to_check = [
    ('embeddings_queue', 'vector', 'topic'),
    ('embeddings', 'seq_id', 'segment_id'),
]

for table, vector_col, filter_col in tables_to_check:
    try:
        print(f"\n{table} table:")
        # Try with collection_id
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {filter_col} = ?", (collection_id,))
        count = cursor.fetchone()[0]
        print(f"  Rows matching collection: {count}")
        
        if count > 0:
            cursor.execute(f"SELECT {vector_col} FROM {table} WHERE {filter_col} = ? LIMIT 1", (collection_id,))
            vec_data = cursor.fetchone()[0]
            if vec_data:
                if isinstance(vec_data, bytes):
                    print(f"  ✓ Found BLOB data: {len(vec_data)} bytes")
                    try:
                        vec = np.frombuffer(vec_data, dtype=np.float32)
                        print(f"  ✓ Parsed as float32: {len(vec)} dimensions")
                        if len(vec) == dimension:
                            print(f"  ✓✓ DIMENSION MATCH! This is the correct location.")
                    except:
                        pass
                else:
                    print(f"  Data type: {type(vec_data)}")
    except Exception as e:
        print(f"  Error: {e}")

# Check segments to find the actual vector segment
print("\n" + "=" * 80)
print("CHECKING SEGMENTS")
print("=" * 80)

cursor.execute("""
    SELECT id, type, scope
    FROM segments
    WHERE collection = ?
""", (collection_id,))

segments = cursor.fetchall()
print(f"\nFound {len(segments)} segments for this collection:")
for seg_id, seg_type, scope in segments:
    print(f"\n  Segment ID: {seg_id}")
    print(f"  Type: {seg_type}")
    print(f"  Scope: {scope}")
    
    # Check if there are embeddings for this segment
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE segment_id = ?", (seg_id,))
    emb_count = cursor.fetchone()[0]
    print(f"  Embeddings count: {emb_count}")
    
    if 'vector' in seg_type.lower() or 'hnsw' in seg_type.lower():
        print(f"  ⚠️ This is a VECTOR segment - vectors may be stored in files!")

# Check for file-based storage
print("\n" + "=" * 80)
print("CHECKING FOR FILE-BASED VECTOR STORAGE")
print("=" * 80)

# Look for chroma data directory
possible_dirs = [
    './chroma',
    './chroma_data',
    './',
    '../',
]

for base_dir in possible_dirs:
    if os.path.exists(base_dir):
        # Look for segment directories
        for seg_id, seg_type, scope in segments:
            seg_dir = os.path.join(base_dir, seg_id)
            if os.path.exists(seg_dir):
                print(f"\n✓ Found segment directory: {seg_dir}")
                files = os.listdir(seg_dir)
                print(f"  Files: {files}")
                
                # Look for common vector file formats
                for f in files:
                    filepath = os.path.join(seg_dir, f)
                    if os.path.isfile(filepath):
                        size = os.path.getsize(filepath)
                        print(f"    {f}: {size} bytes")
                        
                        # Check if size matches expected vector storage
                        expected_size = dimension * 4  # float32
                        if size % expected_size == 0:
                            num_vectors = size // expected_size
                            print(f"      -> Could contain {num_vectors} vectors of dimension {dimension}")

# Try direct access through ChromaDB client
print("\n" + "=" * 80)
print("TRYING CHROMADB CLIENT ACCESS")
print("=" * 80)

try:
    import chromadb
    
    # Get the directory of the sqlite file
    db_dir = os.path.dirname(os.path.abspath(db_path))
    
    print(f"Connecting to ChromaDB at: {db_dir}")
    client = chromadb.PersistentClient(path=db_dir)
    
    collections = client.list_collections()
    print(f"Collections: {[c.name for c in collections]}")
    
    if collections:
        collection = client.get_collection(collection_name)
        count = collection.count()
        print(f"\nCollection '{collection_name}' has {count} documents")
        
        if count > 0:
            # Try to get one document with embedding
            result = collection.get(limit=1, include=['embeddings', 'documents', 'metadatas'])
            
            if result['embeddings']:
                emb = result['embeddings'][0]
                print(f"\n✓✓✓ SUCCESS via ChromaDB Client!")
                print(f"  Embedding shape: {len(emb)}")
                print(f"  First 5 values: {emb[:5]}")
                print(f"\n  → Use ChromaDB Python client to access embeddings!")
                print(f"     This is the recommended approach.")
            
except ImportError:
    print("  ChromaDB not installed - install with: pip install chromadb")
except Exception as e:
    print(f"  Error: {e}")

conn.close()

print("\n" + "=" * 80)
print("SEARCH COMPLETE")
print("=" * 80)