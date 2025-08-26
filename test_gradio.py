from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

def ask(
    user_prompt: str,
) -> str | None:
    """
    Sends a prompt to the local LLM server and prints the response.
    """
    print("-" * 40)
    print(f"PROMPT: {user_prompt}")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
        )
        return response.text

    except Exception as e:
        return f"An error occurred: {e}"

import gradio as gr

demo = gr.Interface(
    fn=ask,
    inputs=["text"],
    outputs=gr.Markdown(),
    title="Local LLM Chat Interface",
    description="Interact with a local LLM server using a simple chat interface.",
    examples=[
        ["Explain the theory of relativity."],
        ["Write a short poem about the sea."],
        ["Write a Python function to calculate Fibonacci numbers."],
        ["Summarize the causes of World War I."],
        ["What are the top attractions in Paris?"],
    ],
    flagging_dir="./flagging_data",
    allow_flagging="manual",
    # show time elapsed for each call
    # live=True,
    theme=gr.themes.Soft(),
    # theme=gr.themes.Ocean,
)

demo.launch()
