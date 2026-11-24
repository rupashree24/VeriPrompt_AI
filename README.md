# VeriPrompt AI
Claim-level fact checking and comparative prompt engineering using RAG.

## Quick start (Python 3.11)
From the repository root:
```bash
python -m venv .venv
```
Windows PowerShell: `.venv\Scripts\Activate.ps1` (or use `.venv\Scripts\python.exe` directly). Linux/macOS: `source .venv/bin/activate`.
```bash
pip install -r requirements.txt
python main.py --demo
python -m pytest -q
streamlit run app.py
```
Offline fixture mode needs no API key or model downloads. It uses TF-IDF, exact matching and a curated AES contradiction rule. Its fixed answer is intentionally erroneous and independent of the question; do not use it for scientific prompt comparisons. For a smaller offline install: `pip install streamlit pandas scikit-learn pytest`.

## Live mode
Copy `.env.example` to `.env` and set your Gemini API key and available model ID. Obtain a key from https://aistudio.google.com/apikey. Never commit `.env`. First live run downloads embedding and NLI models; CPU is supported but may be slow. API usage may incur charges.
```bash
python main.py --question "What is the AES block size?"
streamlit run app.py
```
Uncheck Offline fixture mode in the dashboard. Select a strategy or enter your own answer for verification.

## Corpus and datasets
`data/documents` contains three manually authored short summaries with source links, not verbatim full standards. `data/questions.json` contains six starter questions with reference claims and answerability; `data/ground_truth.json` contains five manually authored verification examples. These are smoke-test fixtures, not a publication-grade benchmark. Download public full standards with `python collect.py`, then use `--folder data/full_documents` or change the dashboard corpus directory. Review downloaded documents and revise annotations to match the chosen corpus. Add PDF/TXT/MD files as needed. For research, expand to 100–200 questions and independent human annotations; separate calibration and held-out test sets. RAGTruth (https://github.com/ParticleMedia/RAGTruth) is an optional external benchmark requiring adaptation of its span labels; it is not bundled or automatically evaluated.

## Architecture and phases
| Phase | Implementation |
|---|---|
| Corpus | collect.py, data/ |
| Preprocessing | modules/preprocessing.py: page extraction, overlapping chunks and source IDs |
| Embeddings | modules/embeddings.py: MiniLM, normalized vectors, persisted FAISS index and metadata |
| Retrieval | modules/retriever.py: cosine similarity search |
| LLM | modules/llm.py: Gemini API, token accounting |
| Prompting | modules/prompts.py: zero-shot, few-shot, CoT-style, standard RAG, evidence-constrained RAG, two-pass self-verification RAG |
| Claim extraction | modules/claim_extractor.py: JSON atomic claims and citations |
| Verification | modules/claim_verifier.py: independent claim retrieval, DeBERTa NLI, conflict flags and citation validation |
| Metrics | modules/evaluation.py and main.py |
| Experiment | benchmark.py |
| Dashboard | app.py: corpus, fact checker, comparison and analytics |

## Evaluation
```bash
python benchmark.py --repeats 3
```
Runs all six configurations on the same questions with the same generator and verifier. Results are written to results/evaluation_results.csv, details.json and verifier.json. Start with one repetition to check API cost. No results are fabricated. The dashboard can run a smaller experiment and export CSV.

Faithfulness = supported / total claims; hallucination proxy = (unsupported + contradicted) / total; contradiction and unsupported rates are reported separately. Citation coverage counts claims with citations; valid coverage counts claims with at least one supporting citation from supplied generation context; citation precision divides valid cited claims by cited claims. Claim rates are percentages, undefined for zero claims. Answer relevance is embedding cosine similarity (TF-IDF in demo); completeness is fraction of reference claims entailed by at least one extracted claim. Zero extracted claims is an abstention proxy, not a definitive refusal classifier. Latency includes extraction and verification; tokens include generator and extractor API usage. Verifier evaluation reports accuracy, macro precision, recall, F1 and confusion matrix. Unsupported does not mean false. Conflicting evidence is conservatively marked unsupported and flagged. Threshold 0.8 is a starting heuristic, not calibrated confidence. Pairwise NLI may miss multi-hop evidence. Compare abstention by answerability and report cost, completeness and uncertainty; use human evaluation to validate automatic scores. The same corpus is used for all strategies, but only RAG strategies receive context during generation. Few-shot examples must be separated from held-out questions for formal experiments.

## GitHub
Create an empty repository named VeriPrompt-AI on GitHub, then run:
```bash
git init
git add .
git commit -m "Initial VeriPrompt AI implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/VeriPrompt-AI.git
git push -u origin main
```
## Limitations
Offline tests exercise the local fixture only. Live Gemini, model downloads, FAISS and Streamlit browser execution require validation on your machine. The dashboard reads a local corpus directory rather than providing file upload. The system is an academic prototype, not a guarantee of factual truth. Do not submit fixture scores as experimental findings.
