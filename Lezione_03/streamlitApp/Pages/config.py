import streamlit as st
from openai import AzureOpenAI
from tenacity import retry
from tenacity import stop_after_attempt




st.title("⚙️ Configurazione Azure OpenAI")
@retry(stop=stop_after_attempt(7),before_sleep=lambda retry_state: st.warning(f"Retrying... {retry_state.attempt_number}/7"))
def try_to_connect(config):
    client = AzureOpenAI(
        api_version=config["api_version"],
        azure_endpoint=config["endpoint"],
        api_key=config["subscription_key"],
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        max_tokens=50,
        temperature=1.0,
        top_p=1.0,
        model=config["deployment"]
    )
    #alert on streamlit
    st.info("Connection successful!")
    return True, "Connection successful!"


st.session_state.defaultConfig= {
    "endpoint": "https://primarisorsa-ai-1509.cognitiveservices.azure.com/",
    "deployment": "gpt-4o",
    "model": "gpt-4o",
    "api_version": "2024-12-01-preview",
    "subscription_key": "",
}

with st.form("cfg-form"):
    endpoint = st.text_input("Endpoint", value=st.session_state.defaultConfig["endpoint"])
    deployment = st.text_input("Deployment", value=st.session_state.defaultConfig["deployment"])
    model = st.text_input("Model", value=st.session_state.defaultConfig["model"])
    api_version = st.text_input("API Version", value=st.session_state.defaultConfig["api_version"])
    subscription_key = st.text_input("Subscription Key", type="password", value=st.session_state.defaultConfig["subscription_key"])
    submitted = st.form_submit_button("Save Configuration")
    if submitted:
        if not subscription_key:
            st.error("Subscription Key is required")
        else:
            st.session_state.defaultConfig["endpoint"] = endpoint
            st.session_state.defaultConfig["deployment"] = deployment
            st.session_state.defaultConfig["model"] = model
            st.session_state.defaultConfig["api_version"] = api_version
            st.session_state.defaultConfig["subscription_key"] = subscription_key
            #try to connect
            try:
                success, message = try_to_connect(st.session_state.defaultConfig)
            except Exception as e:
                st.error(f"Connection failed")
                success = False
            if success:
                st.success(message)
                st.switch_page("Pages/chatbot.py")
            
            
            
            

        
        


