import json
from pathlib import Path
import streamlit as st
import pandas as pd
from main import Engine
from modules.prompts import STRATEGIES
from benchmark import run
st.set_page_config(page_title='VeriPrompt AI',layout='wide'); st.title('VeriPrompt AI'); st.caption('Claim-level fact checking and prompt evaluation')
demo=st.sidebar.checkbox('Offline fixture mode',True); folder=st.sidebar.text_input('Corpus directory','data/documents')
@st.cache_resource
def engine(demo,folder): return Engine(demo,folder)
page=st.sidebar.radio('Page',['Knowledge base','Fact checker','Prompt comparison','Evaluation analytics'])
if demo: st.warning('Offline mode uses a fixed answer, lexical retrieval and limited rule checks. Not an LLM experiment.')
try:
    if page=='Knowledge base':
        from modules.preprocessing import load_documents,create_chunks
        chunks=create_chunks(load_documents(folder)); st.write(f'{len(chunks)} chunks'); st.dataframe(chunks)
        if st.button('Reload corpus'): st.cache_resource.clear(); st.rerun()
    elif page=='Fact checker':
        q=st.text_input('Question','What is AES?'); strategy=st.selectbox('Strategy',STRATEGIES); custom=st.text_area('Optional answer to verify (offline: one claim per line)')
        if st.button('Verify'):
            r=engine(demo,folder).run(q,strategy,custom or None); st.write(r['answer']); m=r['metrics']; cols=st.columns(3)
            for col,key in zip(cols,['faithfulness','hallucination','claim_count']): col.metric(key,m[key])
            for c in r['claims']:
                with st.expander(f"{c['label']}: {c['claim']}"):
                    st.write('Citation valid:',c['citation_valid']); st.write('Conflicting evidence:',c['conflict']); st.dataframe(c['evidence'])
            st.json(m)
    elif page=='Prompt comparison':
        q=st.text_input('Question','What is AES?')
        if st.button('Compare six strategies',disabled=demo):
            results=[engine(False,folder).run(q,s) for s in STRATEGIES]; df=pd.DataFrame([dict(strategy=r['strategy'],**r['metrics']) for r in results]); st.dataframe(df); st.bar_chart(df.set_index('strategy')[['faithfulness','hallucination']]); st.json(results)
    else:
        if st.button('Evaluate verifier'): st.json(engine(demo,folder).benchmark_verifier(json.loads(Path('data/ground_truth.json').read_text())))
        if st.button('Run starter benchmark',disabled=demo):
            df,details=run(engine(False,folder),json.loads(Path('data/questions.json').read_text())); st.session_state['results']=df
        if 'results' in st.session_state: df=st.session_state['results']
        elif Path('results/evaluation_results.csv').exists(): df=pd.read_csv('results/evaluation_results.csv')
        else: df=None
        if df is not None:
            st.dataframe(df); st.bar_chart(df.groupby('strategy')[['faithfulness','hallucination']].mean()); st.download_button('Download CSV',df.to_csv(index=False),'evaluation_results.csv')
except Exception as e: st.error(str(e))
