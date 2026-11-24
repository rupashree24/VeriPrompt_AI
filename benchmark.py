import argparse,json
from pathlib import Path
import pandas as pd
from main import Engine
from modules.prompts import STRATEGIES

def run(engine,questions,repeats=1):
    if engine.demo: raise ValueError('Offline fixture cannot compare actual prompting strategies')
    rows=[]; details=[]
    for q in questions:
        for strategy in STRATEGIES:
            for repeat in range(repeats):
                result=engine.run(q['question'],strategy,references=q['reference_claims']); details.append(result)
                rows.append(dict(id=q['id'],strategy=strategy,repeat=repeat,answerable=q['answerable'],**result['metrics']))
    return pd.DataFrame(rows),details
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repeats',type=int,default=1); p.add_argument('--folder',default='data/documents'); a=p.parse_args()
    e=Engine(folder=a.folder); df,details=run(e,json.loads(Path('data/questions.json').read_text()),a.repeats)
    Path('results').mkdir(exist_ok=True); df.to_csv('results/evaluation_results.csv',index=False); Path('results/details.json').write_text(json.dumps(details,indent=2)); Path('results/verifier.json').write_text(json.dumps(e.benchmark_verifier(json.loads(Path('data/ground_truth.json').read_text())),indent=2)); print(df.groupby('strategy').mean(numeric_only=True))
