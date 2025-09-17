from openai import AzureOpenAI
import streamlit as st

st.title("ChatGPT-like clone")
st.set_page_config(page_title="Azure OpenAI App", initial_sidebar_state="collapsed")


endpoint = st.session_state.defaultConfig["endpoint"]
model_name = st.session_state.defaultConfig["model"]
deployment = st.session_state.defaultConfig["deployment"]
subscription_key=st.session_state.defaultConfig["subscription_key"]
api_version = st.session_state.defaultConfig["api_version"]

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "azure_model" not in st.session_state:
    st.session_state["azure_model"] = model_name

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
            stream=True,
        )
        response=st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})