"""Download public standards into a separate corpus directory."""
import requests
from pathlib import Path
SOURCES={'rfc8446.txt':'https://www.rfc-editor.org/rfc/rfc8446.txt','fips197.pdf':'https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf','rfc9110.txt':'https://www.rfc-editor.org/rfc/rfc9110.txt'}
if __name__=='__main__':
    p=Path('data/full_documents'); p.mkdir(exist_ok=True)
    for name,url in SOURCES.items():
        r=requests.get(url,timeout=60); r.raise_for_status(); (p/name).write_bytes(r.content); print(name,url)
