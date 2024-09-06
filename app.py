import os
import io
import re
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import moviepy.editor as mp
import mimetypes
import pdfplumber
import docx2txt
import ffmpeg
import subprocess
from openai import OpenAI
import google.generativeai as genai

#======================== Setup LLM Models =================================

def gemini_llm_api(context , api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return "An error occurred while generating the answer."



def openai_llm_api(context, api_key):
    # Configure the OpenAI API key
    client = OpenAI(api_key=api_key)
    try:
        # Call the OpenAI model (GPT-3.5-turbo in this case) to generate content
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context}
        ])
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating content: {e}")
        return "An error occurred while generating the answer."

# ============================All functions================================

# Generate Email 
def generate_email(subject, reasons , api_service, api_key):
    content_query =f"""
            Write a professional email based on the following details:
            - **Subject:** {subject}
            - **Ke Reasons:{reasons} // Just conceptually not full correctness
            -complete a proper email

            The email should be [mention the tone, e.g., formal, friendly, persuasive, etc.], and include appropriate greetings, sign-offs, and any necessary details or calls to action. Ensure the content is concise, clear, and aligned with the topic provided.
            """
    if api_service=="Google API(Gemini)":
        return gemini_llm_api(content_query , api_key)
    else:
        return openai_llm_api(content_query, api_key)


#Generate Text Summary
def generate_summary(text , api_service , api_key):
    content_query = f"""
        Summarize the following text:
        - **Text:** {text}
        - **Summary Requirements:**
          - Provide a concise summary that captures the main points and key details.
          - The summary should be as brief as possible while still covering the essential content of the original text.
          - Ensure that the summary is clear, coherent, and maintains the essential meaning of the original text.

        The goal is to provide a clear and accurate representation of the main content of the original text, without specifying a fixed length.
    """
    if api_service=="Google API(Gemini)":
        return gemini_llm_api(content_query , api_key)
    else:
        return openai_llm_api(content_query, api_key)


# Document Query
def answer_question(document_text, question , api_service , api_key):
    print("inside LLM : " , question , document_text)
    # Construct the prompt for the LLM
    prompt = f"""
    You are an AI assistant. Given the following text from a document, answer the question as accurately as possible.
    
    **Document Text:**
    {document_text}
    
    **Question:**
    {question}
    
    **Answer:**
    """

    if api_service=="Google API(Gemini)":
        return gemini_llm_api(prompt , api_key)
    else:
        return openai_llm_api(prompt, api_key)

# Generate Article
def generate_article_blog(topic, key_points, api_service, api_key , length=50):
    tone="informative"
    content_query = f"""
        Write a detailed article or blog post on the following topic:
        - **Topic:** {topic}
        - **Key Points to Cover:**
          {key_points}
        - The article should be written in a {tone} tone and should aim to educate, inform, or entertain the audience.
        - Length: The article should be approximately {length}.

        Include an introduction that hooks the reader, well-structured body paragraphs, and a conclusion that summarizes the key points or provides a call to action. Ensure the content is engaging, comprehensive, and aligned with the topic provided.
    """

    if api_service=="Google API(Gemini)":
        return gemini_llm_api(content_query, api_key)
    else:
        return openai_llm_api(content_query, api_key)


# Generate Text Summary from different types of files
def read_pdf(file):
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        # Regular expression to remove URLs
        text_without_links = re.sub(r'https?://\S+', '', text)
        return text_without_links
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return "Error reading PDF file."

def read_docx(file):
    try:
        text = docx2txt.process(io.BytesIO(file.read()))
        # Regular expression to remove URLs
        text_without_links = re.sub(r'https?://\S+', '', text)
        return text_without_links
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return "Error reading DOCX file."

def read_txt(file):
    try:
        text = file.read().decode('utf-8')
        # Regular expression to remove URLs
        text_without_links = re.sub(r'https?://\S+', '', text)
        return text_without_links
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return "Error reading TXT file."

def document_to_text(uploaded_file):
    if uploaded_file is None:
        return "No file uploaded."

    file_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0]

    # Determine the file type and read accordingly
    if file_type == 'application/pdf':
        return read_pdf(uploaded_file)
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return read_docx(uploaded_file)
    elif file_type == 'text/plain':
        return read_txt(uploaded_file)
    else:
        return "Unsupported file type."

def mp3_to_wav(mp3_audio):
    """
    Convert an MP3 audio stream to WAV format in-memory.
    """
    audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_audio.read()))
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def extract_text_from_audio(audio_stream, is_wav=False):
    """
    Split the large audio file into chunks and apply speech recognition on each chunk.
    """
    if not is_wav:
        # Convert MP3 to WAV if the input is not WAV
        wav_audio = mp3_to_wav(audio_stream)
    else:
        wav_audio = io.BytesIO(audio_stream.read())

    sound = AudioSegment.from_wav(wav_audio)

    # Debug: Print audio details
    print(f"Audio Length: {len(sound)} ms")
    print(f"Audio dBFS: {sound.dBFS}")

    chunks = split_on_silence(
        sound,
        min_silence_len=1000,
        silence_thresh=sound.dBFS - 10,
        keep_silence=500,
    )

    print(f"Number of chunks: {len(chunks)}")  # Debug print

    recognizer = sr.Recognizer()
    whole_text = []

    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_io = io.BytesIO()
        audio_chunk.export(chunk_io, format="wav")
        chunk_io.seek(0)

        with sr.AudioFile(chunk_io) as source:
            audio_listened = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_listened)
                text = f"{text.capitalize()}."
                whole_text.append(text)
            except sr.UnknownValueError:
                print(f"Chunk {i}: Audio unintelligible")  # Debug print
                continue
            except sr.RequestError as e:
                print(f"Chunk {i}: Request error {e}")  # Debug print
                continue

    result_text = ' '.join(whole_text)
    print(f"Final Extracted Text: {result_text}")  # Debug print
    return result_text


def generate_audio_summary(audio_file , api_service, api_key):
    try:
        # Check if the file is WAV by its extension or any other logic
        is_wav = audio_file.name.lower().endswith('.wav')
        text = extract_text_from_audio(audio_file, is_wav=is_wav)
        print("Extracted Text:", text)
        return generate_summary(text , api_service, api_key)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


#==================== Streamlit APP =============================
# Combined Custom CSS for Project and Task Titles
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        margin-top: -50px;  /* Moves the title closer to the top */
        text-align: center;
        color: #ffff;
    }
    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
        text-align: center;
        color: #ff11ff;
    }
    .task-title {
        font-size: 32px;
        font-weight: bold;
        color: #1abc9c;  /* Change this to your desired color */
        text-align: left;  /* Adjust alignment if needed */
        margin-bottom: 20px;
    }
    
    </style>
""", unsafe_allow_html=True)

# Project Header or Title
st.markdown("<h1 class='main-title'>LLM-Based Multi-Task Web Application</h1>", unsafe_allow_html=True)

# Add a subtitle or description below the title
st.markdown("<h3 class='subtitle'>An AI-powered app for text extraction, summarization, and more</h3>", unsafe_allow_html=True)


# Sidebar section for API Key Selection and Task Selection
st.sidebar.title("Configuration")

# Selector for choosing between Gemini and ChatGPT API
api_service = st.sidebar.selectbox("Select API Service", ["Google API(Gemini)", "OpenAI (ChatGPT)"])

# Input for the selected API key
if api_service == "OpenAI (ChatGPT)":
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
elif api_service == "Google API(Gemini)":
    api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

st.sidebar.title("LLM Task Selector")
task = st.sidebar.selectbox("Choose a task", ["Text Extractor" ,"Email Writing", "Article Writing",   "Text Summarization",   "Document Q&A", "Audio Summarization"])

if task == "Text Extractor":
    st.markdown("<h2 class='task-title'>Text Extractor</h2>", unsafe_allow_html=True)  # Custom styled title

    document = st.file_uploader("Upload Document", type=['pdf', 'docx', 'txt'])
    
    if st.button("Get Text"):
        extracted_text = document_to_text(document)
        st.write(extracted_text)  # Display the extracted text

elif task == "Email Writing":
    st.markdown("<h2 class='task-title'>Email Writing</h2>", unsafe_allow_html=True)  # Custom styled title
    
    # Input with placeholder for Subject and Key Reasons
    subject = st.text_input("Subject", placeholder="Enter the email subject")
    reasons = st.text_area("Key Reasons", placeholder="Enter key points or reasons for the email")
    
    if st.button("Generate Email"):
        email = generate_email(subject, reasons, api_service, api_key)
        st.write(email)  # Display the generated email

elif task == "Article Writing":
    st.markdown("<h2 class='task-title'>Article/Blog Writing</h2>", unsafe_allow_html=True)  # Custom styled title
    
    # Input with placeholder for Topic and Keywords
    topic = st.text_input("Topic", placeholder="Enter the topic for the article/blog")
    keywords = st.text_area("Keywords", placeholder="Enter keywords (comma separated)")
    
    length = st.slider("Length (words)", 100, 2000)
    
    # Placeholder for the article output
    article_placeholder = st.empty()  # Empty placeholder
    
    if st.button("Generate Article"):
        article = generate_article_blog(topic, keywords, api_service, api_key, length)
        article_placeholder.write(article)  # Display the generated article

elif task == "Text Summarization":
    st.markdown("<h2 class='task-title'>Text Summarization</h2>", unsafe_allow_html=True)  # Custom styled title
    
    # Text input with placeholder for summarization
    text = st.text_area("Text to summarize", placeholder="Enter the text you want summarized")
    
    if st.button("Summarize"):
        summary = generate_summary(text, api_service, api_key)
        st.write(summary)  # Display the summary

elif task == "Document Q&A":
    st.markdown("<h2 class='task-title'>Document Q&A</h2>", unsafe_allow_html=True)  # Custom styled title
    
    # File uploader for document and placeholder for question
    document = st.file_uploader("Upload Document", type=['pdf', 'docx', 'txt'])
    
    if document:
        extracted_text = document_to_text(document)
        st.write("Document text has been extracted.")
    
    question = st.text_input("Ask a question", placeholder="Enter your question related to the document")
    
    if st.button("Get Answer"):
        answer = answer_question(extracted_text, question, api_service, api_key)
        st.write(answer)  # Display the answer

elif task == "Audio Summarization":
    
    st.markdown("<h2 class='task-title'>Audio Summarization</h2>", unsafe_allow_html=True)  # Custom styled title
    
    # File uploader for audio
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if audio_file and st.button("Audio Summarize"):
        summary = generate_audio_summary(audio_file, api_service, api_key)
        st.write(summary)  # Display the audio summary

else:
    st.write("Invalid selection. Please choose a task.")

# Footer section
def footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2c3e50;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        font-family: 'Arial', sans-serif;
    }
    .footer a {
        color: #f39c12;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        <p>Developed by <strong>Rasel Meya</strong> | Contact: <a href="mailto:raselmeya2194@gmail.com">raselmeya2194@gmail.com</a> |
        GitHub: <a href="https://github.com/raselmeya94">github.com/raselmeya94</a> |
        LinkedIn: <a href="https://linkedin.com/in/raselmeya">linkedin.com/in/raselmeya</a></p>
    </div>
    """, unsafe_allow_html=True)

# Call the footer function at the end of your app
footer()
