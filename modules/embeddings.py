import json
from pathlib import Path

def build_index(chunks,folder='vector_store'):
    import faiss
    from sentence_transformers import SentenceTransformer
    if not chunks: raise ValueError('No document chunks')
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vectors=model.encode([c['text'] for c in chunks],normalize_embeddings=True).astype('float32')
    index=faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    p=Path(folder); p.mkdir(exist_ok=True,parents=True)
    faiss.write_index(index,str(p/'index.faiss'))
    (p/'metadata.json').write_text(json.dumps(chunks,indent=2),encoding='utf-8')
    return model,index,chunks

def load_index(folder='vector_store'):
    import faiss
    from sentence_transformers import SentenceTransformer
    p=Path(folder)
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),faiss.read_index(str(p/'index.faiss')),json.loads((p/'metadata.json').read_text())
