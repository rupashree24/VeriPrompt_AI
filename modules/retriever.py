class Retriever:
    def __init__(self,model,index,chunks): self.model,self.index,self.chunks=model,index,chunks
    def search(self,query,k=5):
        v=self.model.encode([query],normalize_embeddings=True).astype('float32')
        scores,ids=self.index.search(v,min(k,len(self.chunks)))
        return [{**self.chunks[i],'similarity':float(s)} for s,i in zip(scores[0],ids[0]) if i>=0]
    def similarity(self,a,b):
        v=self.model.encode([a,b],normalize_embeddings=True)
        return float(v[0]@v[1])

class LocalRetriever:
    """Offline lexical baseline, not a semantic embedding model."""
    def __init__(self,chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks=chunks; self.model=TfidfVectorizer(); self.matrix=self.model.fit_transform([c['text'] for c in chunks])
    def search(self,query,k=5):
        scores=(self.matrix@self.model.transform([query]).T).toarray().ravel()
        return [{**self.chunks[i],'similarity':float(scores[i])} for i in scores.argsort()[::-1][:k]]
    def similarity(self,a,b):
        v=self.model.transform([a,b]); return float((v[0]@v[1].T).toarray()[0,0])
