from google import genai
from google.genai import types
import gradio as gr

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

def ask(
    user_prompt: str,
    include_thoughts: bool = True,
    thinking_budget: int = -1
) -> tuple[str, str]:
    """
    Sends a prompt to the local LLM server and prints the response.
    """
    print("-" * 50)
    print(f"PROMPT: {user_prompt}")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                # thinking_config=types.ThinkingConfig(thinking_budget=1024)
                # Turn off thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=0)
                # Turn on dynamic thinking:
                # thinking_config=types.ThinkingConfig(thinking_budget=-1)
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget,
                    include_thoughts=include_thoughts
                )
            ),
        )
        # return response.text
        thought = "\n**Thought**\n\n"
        answer = "\n**Response**\n\n"

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thought += part.text
                else:
                    answer += part.text
        return thought, answer

    except Exception as e:
        raise RuntimeError(f"Error generating content: {e}")


demo = gr.Interface(
    fn=ask,
    inputs=[
        gr.Textbox(label="User Prompt"),
        gr.Radio(choices=[True, False], label="Include Thoughts", type="value"),
        gr.Radio(choices=[-1, 0, 1024, 4096], label="Thinking Budget", type="value")
    ],
    outputs=[gr.Markdown(), gr.Markdown()],
    title="Local LLM Chat Interface",
    description="Interact with a local LLM server using a simple chat interface.",
    examples=[
        ["Explain the theory of relativity."],
        ["Write a short poem about the sea."],
        ["Write a Python function to calculate Fibonacci numbers."],
        ["Summarize the causes of World War I."],
        ["What are the top attractions in Paris?"],
    ],
    # flagging_dir="./flagging_data",
    # allow_flagging="manual",
    # show time elapsed for each call
    # live=True,
    # theme=gr.themes.Soft(),
    # theme=gr.themes.Ocean,
)

demo.launch()
