from pathlib import Path

def load_documents(folder):
    pages=[]
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower()=='.pdf':
            import fitz
            with fitz.open(p) as doc:
                pages.extend(dict(text=x.get_text(),source=p.name,page=i+1) for i,x in enumerate(doc))
        elif p.suffix.lower() in ('.txt','.md'):
            pages.append(dict(text=p.read_text(encoding='utf-8'),source=p.name,page=1))
    return pages

def create_chunks(pages,size=120,overlap=25):
    if not 0<=overlap<size: raise ValueError('Invalid overlap')
    chunks=[]
    for p in pages:
        words=p['text'].split()
        for i in range(0,len(words),size-overlap):
            chunks.append({**p,'text':' '.join(words[i:i+size]),'chunk_id':f'C{len(chunks)+1}'})
            if i+size>=len(words): break
    return chunks
