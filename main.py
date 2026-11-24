import argparse,json,time
from modules.preprocessing import load_documents,create_chunks
from modules.retriever import Retriever,LocalRetriever
from modules.claim_verifier import NLIVerifier,LocalVerifier,verify_claim
from modules.claim_extractor import extract_claims,local_claims
from modules.prompts import STRATEGIES,generate_answer
from modules.evaluation import metrics,classification_metrics

class Engine:
    def __init__(self,demo=False,folder='data/documents'):
        self.demo=demo; chunks=create_chunks(load_documents(folder))
        if demo: self.retriever=LocalRetriever(chunks); self.verifier=LocalVerifier(); self.llm=None
        else:
            from modules.embeddings import build_index
            from modules.llm import GeminiLLM
            self.retriever=Retriever(*build_index(chunks)); self.verifier=NLIVerifier(); self.llm=GeminiLLM()
    def run(self,question,strategy,answer=None,references=None):
        start=time.perf_counter(); before=self.llm.tokens if self.llm else 0
        evidence=self.retriever.search(question); supplied=evidence if strategy in STRATEGIES[3:] else []
        if answer is None:
            if self.demo: answer='AES uses a 128-bit block size. [C1]\nAES uses a 512-bit block size.\nAES is used by 90 percent of organizations.'
            else: answer=generate_answer(self.llm,question,strategy,evidence)
        items=local_claims(answer) if self.demo else extract_claims(answer,self.llm)
        claims=[verify_claim(c,self.retriever,self.verifier,supplied) for c in items]
        m=metrics(claims); m['answer_relevance']=self.retriever.similarity(question,answer); m['latency_seconds']=time.perf_counter()-start; m['tokens']=self.llm.tokens-before if self.llm else None
        m['answer_completeness']=sum(any(self.verifier.pair(r,c['claim'])[0]=='Supported' for c in claims) for r in references)/len(references) if references else None
        return dict(question=question,strategy=strategy,answer=answer,claims=claims,metrics=m,demo=self.demo)
    def benchmark_verifier(self,rows):
        pred=[verify_claim({'claim':r['claim'],'citations':[]},self.retriever,self.verifier,[])['label'] for r in rows]
        return classification_metrics([r['label'] for r in rows],pred)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--demo',action='store_true'); p.add_argument('--folder',default='data/documents'); p.add_argument('--question',default='What is AES?'); a=p.parse_args()
    print(json.dumps(Engine(a.demo,a.folder).run(a.question,STRATEGIES[4]),indent=2))
