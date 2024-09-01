import streamlit as st
from openai import OpenAI
from transformers import pipeline

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it"
)

qa_model = pipeline("question-answering")

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management


# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)", type=("txt", "md")
)

# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    # Process the uploaded file and question.
    context = uploaded_file.read().decode()
    # Pass the inputs to the pipeline
    outputs = qa_model(question=question, context=context)

    # Access and print the answer
    stream = outputs['answer']

    # Stream the response to the app using `st.write_stream`.
    st.write(stream)
