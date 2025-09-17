from openai import AzureOpenAI
import streamlit as st
import os
from dotenv import load_dotenv



st.title("ChatGPT-like clone")


endpoint = os.getenv("ENDPOINT")
model_name = "gpt-4o"
deployment = "gpt-4o"

subscription_key=os.getenv("SUBSCRIPTION_KEY")

api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "azure_model" not in st.session_state:
    st.session_state["azure_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["azure_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=False,
        )
        response=st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})