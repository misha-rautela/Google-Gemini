#Using VSCode project and the env file for saving API KEY.

from dotenv import load_dotenv
load_dotenv() ## loading env variables

#Using streamlit library for quick web page view and interaction
import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## function to load Gemini Pro model and get response back

model=genai.GenerativeModel('gemini-1.5-pro')
def get_gemini_response():
    response = model.generate_content(question)
    #print("response", response.text )
    return response.text

##initializing the streamlit app

st.set_page_config(page_title="Conversational LLM App")

st.header("Gemini Application")
input=st.text_input("Input: ",key="input")
submit=st.button("Ask the question")

if submit:
    
    response=get_gemini_response(input)
    st.subheader("The Response is")
    st.write(response)
