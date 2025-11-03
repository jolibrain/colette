#!/usr/bin/env python3
"""
Diagnostic script to find where embeddings are actually stored
"""

import sqlite3
import numpy as np
import struct
import pickle

db_path = "../app_colette/mm_index/chroma.sqlite3"  # Change this to your database path

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 80)
print("FINDING EMBEDDINGS IN CHROMADB")
print("=" * 80)

# Get first collection
cursor.execute("SELECT id, name, dimension FROM collections LIMIT 1;")
collection_id, collection_name, dimension = cursor.fetchone()

print(f"\nCollection: {collection_name}")
print(f"Collection ID: {collection_id}")
print(f"Expected dimension: {dimension}")

# Check all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]
print(f"\nAll tables: {tables}")

# Examine embeddings table structure
print("\n" + "=" * 80)
print("EMBEDDINGS TABLE STRUCTURE")
print("=" * 80)
cursor.execute("PRAGMA table_info(embeddings);")
emb_columns = cursor.fetchall()
print("Columns in 'embeddings' table:")
for col in emb_columns:
    print(f"  {col[1]:20s} {col[2]:15s}")

# Get a sample row with ALL columns
cursor.execute("""
    SELECT e.*
    FROM embeddings e
    JOIN segments s ON e.segment_id = s.id
    WHERE s.collection = ?
    LIMIT 1
""", (collection_id,))

sample_row = cursor.fetchone()
column_names = [desc[0] for desc in cursor.description]

print(f"\nSample row from embeddings table:")
for col_name, value in zip(column_names, sample_row):
    val_type = type(value).__name__
    if isinstance(value, bytes):
        print(f"  {col_name:20s}: BLOB ({len(value)} bytes)")
    elif isinstance(value, (int, float, str)):
        print(f"  {col_name:20s}: {value} (type: {val_type})")
    else:
        print(f"  {col_name:20s}: {val_type}")

# Check if there's an embedding_fulltext_search or vector table
print("\n" + "=" * 80)
print("SEARCHING FOR VECTOR STORAGE")
print("=" * 80)

for table in tables:
    if 'vector' in table.lower() or 'embedding' in table.lower():
        print(f"\nTable: {table}")
        cursor.execute(f"PRAGMA table_info({table});")
        cols = cursor.fetchall()
        for col in cols:
            print(f"  {col[1]:20s} {col[2]:15s}")
        
        # Check for BLOB columns
        cursor.execute(f"SELECT * FROM {table} LIMIT 1;")
        if cursor.description:
            row = cursor.fetchone()
            if row:
                for col_name, value in zip([d[0] for d in cursor.description], row):
                    if isinstance(value, bytes):
                        print(f"  -> Found BLOB in column '{col_name}': {len(value)} bytes")

# Check segments table
if 'segments' in tables:
    print("\n" + "=" * 80)
    print("SEGMENTS TABLE")
    print("=" * 80)
    cursor.execute("PRAGMA table_info(segments);")
    seg_columns = cursor.fetchall()
    print("Columns in 'segments' table:")
    for col in seg_columns:
        print(f"  {col[1]:20s} {col[2]:15s}")
    
    cursor.execute("SELECT * FROM segments WHERE collection = ? LIMIT 1;", (collection_id,))
    if cursor.description:
        seg_row = cursor.fetchone()
        if seg_row:
            print(f"\nSample segment:")
            for col_name, value in zip([d[0] for d in cursor.description], seg_row):
                if isinstance(value, bytes):
                    print(f"  {col_name:20s}: BLOB ({len(value)} bytes)")
                else:
                    print(f"  {col_name:20s}: {value}")
                    print(f"  {col_name:20s}: BLOB ({len(value)} bytes)")
                # else:
                #     print(f"  {col_name:20s}: {value}")

# Look for a separate vector or embedding_vectors table
print("\n" + "=" * 80)
print("CHECKING FOR SEPARATE VECTOR STORAGE")
print("=" * 80)

# ChromaDB might store vectors in a separate optimized structure
# Let's check if there's a connection through embedding_id
cursor.execute("""
    SELECT e.id, e.embedding_id
    FROM embeddings e
    JOIN segments s ON e.segment_id = s.id
    WHERE s.collection = ?
    LIMIT 1
""", (collection_id,))

emb_id, embedding_id = cursor.fetchone()
print(f"\nEmbedding ID: {emb_id}")
print(f"Embedding reference ID: {embedding_id}")

# Check if there's a table that uses this embedding_id
for table in tables:
    try:
        # Try to find a column that might match embedding_id
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [c[1] for c in cursor.fetchall()]
        
        if any('embed' in col.lower() or 'vector' in col.lower() or 'id' in col.lower() for col in cols):
            cursor.execute(f"SELECT * FROM {table} WHERE id = ? OR embedding_id = ? LIMIT 1;", (embedding_id, embedding_id))
            row = cursor.fetchone()
            if row:
                print(f"\nFound match in table '{table}':")
                for col_name, value in zip([d[0] for d in cursor.description], row):
                    if isinstance(value, bytes):
                        print(f"  {col_name:20s}: BLOB ({len(value)} bytes)")
                        # Try to deserialize
                        try:
                            vec = np.frombuffer(value, dtype=np.float32)
                            if len(vec) == dimension:
                                print(f"    ✓ Successfully parsed as float32 with correct dimension!")
                                print(f"    First 5 values: {vec[:5]}")
                        except:
                            pass
                    else:
                        print(f"  {col_name:20s}: {value}")
    except:
        pass

conn.close()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)