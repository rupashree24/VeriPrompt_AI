from main import Engine
from modules.preprocessing import create_chunks
from modules.evaluation import metrics

def test_demo():
    r=Engine(True).run('What is AES?','Evidence-constrained RAG'); assert [c['label'] for c in r['claims']]==['Supported','Contradicted','Unsupported']; assert r['metrics']['claim_count']==3

def test_empty(): assert metrics([])['faithfulness'] is None

def test_chunking(): assert len(create_chunks([{'text':'a b c d e','source':'x','page':1}],3,1))==2
