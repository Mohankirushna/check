import os
import faiss
import gradio as gr
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ========== Step 1: Extract Q&A Pairs (MCQ style with bold answers) ==========
def extract_qna_from_pdfs(folder_path):
    qa_pairs = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ The folder '{folder_path}' does not exist. Please check the path.")

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    words = page.extract_words(extra_attrs=["fontname"], use_text_flow=True)

                    current_question = []
                    found_answer = None

                    for word in words:
                        text = word['text'].strip()
                        font = word['fontname'].lower()

                        if 'bold' not in font:
                            current_question.append(text)
                        elif current_question:
                            found_answer = text
                            full_question = ' '.join(current_question).strip()
                            qa_pairs.append((full_question, found_answer))
                            current_question = []
                            found_answer = None
    return qa_pairs

# ========== Step 2: Build FAISS Index ==========
def build_faiss_index_from_qna(qa_pairs):
    questions = [q for q, a in qa_pairs]
    answers = [a for q, a in qa_pairs]
    embeddings = embedding_model.encode(questions)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, questions, answers

# ========== Step 3: Keyword Search Function ==========
def create_qa_function(questions, answers):
    def search_keyword(keyword):
        keyword = keyword.lower()
        matched_qna = []

        for q, a in zip(questions, answers):
            if keyword in q.lower():
                matched_qna.append(f"<b>Q:</b> {q}<br><b>A:</b> <b>{a}</b>")

        if not matched_qna:
            return "❌ No relevant Q&A pairs found for the given keyword."

        return "<br><br>".join(matched_qna)
    return search_keyword

# ========== Step 4: Setup ==========
# Use local 'pdfs' folder or environment variable
pdf_folder = os.getenv("PDF_FOLDER_PATH", os.path.join(os.path.dirname(__file__), "pdfs"))

print("🔍 Extracting Q&A pairs from PDFs...")
qa_pairs = extract_qna_from_pdfs(pdf_folder)
print(f"✅ Extracted {len(qa_pairs)} Q&A pairs.")

# Preview few Q&A
for i, (q, a) in enumerate(qa_pairs[:5]):
    print(f"{i+1}. Q: {q}\n   A: {a}")

# Build index
faiss_index, questions, answers = build_faiss_index_from_qna(qa_pairs)
qa_func = create_qa_function(questions, answers)

# ========== Step 5: Gradio Interface ==========
iface = gr.Interface(
    fn=qa_func,
    inputs=gr.Textbox(lines=2, placeholder="Enter a keyword from your PDFs..."),
    outputs=gr.HTML(),
    title="📄 MCQ PDF Keyword Q&A Search",
    description="Search MCQs from PDF files based on a keyword. Answers are extracted from bold text.",
)
iface.launch()
