import gradio as gr
from transformers import pipeline

# Load AI Model (DeepSeek or Any Preferred Model)
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to Answer Questions
def answer_question(context, question):
    result = qa_model(question=question, context=context)
    return result["answer"]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# AI Tutor: Ask Questions from Your Study Materials")
    
    subject = gr.Textbox(label="Enter Subject/Topic")
    context = gr.Textbox(label="Paste Text from Your Slides/PDF")
    question = gr.Textbox(label="Ask a Question")

    answer_btn = gr.Button("Get Answer")
    answer_output = gr.Textbox(label="AI's Answer")

    answer_btn.click(answer_question, inputs=[context, question], outputs=answer_output)

# Run the App
if __name__ == "__main__":
    demo.launch()
