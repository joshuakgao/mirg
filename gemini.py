import google.generativeai as genai
import os

class Gemini():
    def __init__(self):
        genai.configure(api_key=os.environ['GEMINI_KEY'])
        self.model = genai.GenerativeModel(model_name='gemini-1.5-flash')

    def query(self, query):
        response = self.model.generate_content(query)
        return response.text