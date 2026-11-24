class NLIVerifier:
    def __init__(self,threshold=.8):
        from transformers import pipeline
        self.model=pipeline('text-classification',model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',top_k=None); self.threshold=threshold
    def pair(self,claim,evidence):
        p=self.model({'text':evidence,'text_pair':claim},truncation=True)
        if p and isinstance(p[0],list): p=p[0]
        best=max(p,key=lambda x:x['score']); label=best['label'].lower()
        return ({'entailment':'Supported','contradiction':'Contradicted','neutral':'Unsupported'}[label] if best['score']>=self.threshold else 'Unsupported',float(best['score']))

class LocalVerifier:
    """Exact match and curated contradiction baseline for offline smoke testing only."""
    def pair(self,claim,evidence):
        import re
        norm=lambda x: re.sub(r'[^a-z0-9 ]','',x.lower()).strip()
        if norm(claim) in norm(evidence): return 'Supported',1.0
        if 'aes' in claim.lower() and '512-bit block' in claim.lower() and '128-bit block' in evidence.lower(): return 'Contradicted',1.0
        return 'Unsupported',0.0

def verify_claim(item,retriever,verifier,generation_evidence,k=5):
    claim=item['claim']; evidence=retriever.search(claim,k)
    pairs=[{**e,'label':l,'confidence':s} for e in evidence for l,s in [verifier.pair(claim,e['text'])]]
    supported=[e for e in pairs if e['label']=='Supported']; contradicted=[e for e in pairs if e['label']=='Contradicted']
    conflict=bool(supported and contradicted)
    label='Unsupported' if conflict else 'Supported' if supported else 'Contradicted' if contradicted else 'Unsupported'
    cited=[e for e in generation_evidence if e['chunk_id'] in item['citations']]
    valid=any(verifier.pair(claim,e['text'])[0]=='Supported' for e in cited)
    return {**item,'label':label,'conflict':conflict,'evidence':pairs,'citation_valid':valid}
