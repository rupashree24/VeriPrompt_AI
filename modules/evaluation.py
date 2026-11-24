from collections import Counter

def metrics(claims):
    n=len(claims); counts=Counter(c['label'] for c in claims)
    pct=lambda x:100*x/n if n else None
    cited=sum(bool(c['citations']) for c in claims); valid=sum(c['citation_valid'] for c in claims)
    return dict(claim_count=n,faithfulness=pct(counts['Supported']),hallucination=pct(counts['Unsupported']+counts['Contradicted']),unsupported_rate=pct(counts['Unsupported']),contradiction_rate=pct(counts['Contradicted']),citation_coverage=pct(cited),valid_citation_coverage=pct(valid),citation_precision=100*valid/cited if cited else None,abstention=int(n==0))

def classification_metrics(truth,pred):
    from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
    labels=['Supported','Unsupported','Contradicted']; p,r,f,_=precision_recall_fscore_support(truth,pred,labels=labels,average='macro',zero_division=0)
    return dict(accuracy=accuracy_score(truth,pred),precision=p,recall=r,macro_f1=f,labels=labels,confusion_matrix=confusion_matrix(truth,pred,labels=labels).tolist())
