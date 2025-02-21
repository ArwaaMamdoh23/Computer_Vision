import streamlit as st
import requests
from nbconvert import PythonExporter
from nbformat import read
from io import StringIO

def load_notebook(notebook_url):
    response = requests.get(notebook_url)
    return response.text

st.title("Deploy Colab Notebook with Streamlit")

notebook_url = 'https://raw.githubusercontent.com/ArwaaMamdoh23/Computer_Vision/main/Computer_Vision.ipynb'

notebook_content = load_notebook(notebook_url)

notebook_node = read(StringIO(notebook_content), as_version=4)

python_exporter = PythonExporter()
source, _ = python_exporter.from_notebook_node(notebook_node)

st.code(source, language="python")

st.write("Displaying Notebook Content:")
st.markdown(f"[Your Notebook on GitHub](https://github.com/ArwaaMamdoh23/Computer_Vision/blob/main/Computer_Vision.ipynb)")
