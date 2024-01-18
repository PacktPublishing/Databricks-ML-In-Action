# Here is an example how your App can look to deploy using Gradio. 
# You can host this Gradio app anywhere!
# At the time of this book creaiton Databricks Lakehouse Apps were not in Public preview. 
# We have selected Hugging Face Spaces to host a simple container with our Gradio App using
# Databricks Model Serving Endpoints to orchestrate everything. 

import itertools
import gradio as gr
import requests
import os
from gradio.themes.utils import sizes


def respond(message, history):

    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    # Set your secrets under HF secrets for your Space 
    local_token = os.getenv('API_TOKEN')
    local_endpoint = os.getenv('API_ENDPOINT')

    if local_token is None or local_endpoint is None:
        return "ERROR missing env variables"

    # Add your API token to the headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {local_token}'
    }

    # Pay attenction to what your model expects as schema format 
    q = {"inputs": [message]}
    try:
        response = requests.post(
            local_endpoint, json=q, headers=headers, timeout=100)
        response_data = response.json()
        response_data=response_data["predictions"][0]

    except Exception as error:
        response_data = f"ERROR status_code: {type(error).__name__}"

    return response_data


theme = gr.themes.Soft(
    text_size=sizes.text_sm,radius_size=sizes.radius_sm, spacing_size=sizes.spacing_sm,
)


demo = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
    textbox=gr.Textbox(placeholder="Ask me a question",
                       container=False, scale=7),
    title="Ml in Action LLM RAG demo - Chat with Mixtral from Foundational Databricks model serving endpoint",
    description="This chatbot is a demo example for the llm chatbot. <br>This content is provided as a LLM RAG educational example, without support. It is using Mixtral, can hallucinate and should not be used as production content.<br>Please do not use this for production.",
    examples=[["Can LLM's impact wages and how ?"],
              ["Will AI impact work forces in the US ? "],
              ["Can LLM's impact wages and how ?"],],
    cache_examples=False,
    theme=theme,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

if __name__ == "__main__":
    demo.launch()