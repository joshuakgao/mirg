import google.generativeai as genai
import os

class Gemini():
    def __init__(self):
        genai.configure(api_key=open("gemini_key.txt").read())
        self.model = genai.GenerativeModel(model_name='gemini-1.5-flash')

    def query(self, query):
        response = self.model.generate_content(query)
        return response.text