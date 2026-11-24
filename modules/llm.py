import os
class GeminiLLM:
    def __init__(self):
        from dotenv import load_dotenv
        from google import genai
        load_dotenv(); key=os.getenv('GEMINI_API_KEY')
        if not key: raise ValueError('Set GEMINI_API_KEY in .env')
        self.client=genai.Client(api_key=key); self.model=os.getenv('GEMINI_MODEL','gemini-2.5-flash'); self.tokens=0
    def generate(self,prompt,json_output=False):
        config={'temperature':0}
        if json_output: config['response_mime_type']='application/json'
        r=self.client.models.generate_content(model=self.model,contents=prompt,config=config)
        self.tokens+=getattr(r.usage_metadata,'total_token_count',0) or 0
        if not r.text: raise RuntimeError('Empty model response')
        return r.text
