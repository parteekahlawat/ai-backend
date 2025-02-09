from flask import Flask, request, jsonify, send_file
# from happytransformer import HappyTextToText, TTSettings
from flask_cors import CORS
import os
import assemblyai as aai
import re
from docx import Document
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
import plotly.express as px
# import plotly.io as pio
# from flask_cors import cross_origin
# from PIL import Image
# import io
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
from dotenv import load_dotenv
import json
# Load environment variables from .env file
load_dotenv()
# Initialize the model
# happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
# args = TTSettings(num_beams=500, min_length=1)

import google.generativeai as genai
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(
    history=[
        {"role": "user", "parts": "Hello"},
        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    ]
)

# response = chat.send_message("what is html", stream=True)
# for chunk in response:
#     print(chunk.text, end="", flush=True) 

@app.route('/add-event', methods=['POST'])
def extract_event_details():
    # model = genai.GenerativeModel("gemini-1.5-flash")
    data = request.json
    user_input = data.get("text", "")

    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    # 🔥 Ask Gemini to structure the response as JSON
    prompt = f"""
    Extract event details from the following user input and return in JSON format only:
    User Input: "{user_input}"
    
    Expected JSON format:
    {{
        "id": unique number in string,
        "title": "Event Title",
        "description": "Short event description",
        "date": "YYYY-MM-DDTHH:MM:SS.000Z",
    }}
    """
    
    response = model.generate_content(prompt)
    # response =''
    print(response.text)
    try:
        # event_data = json.loads(response.text)  # Convert to JSON
        extracted_text = response.text.strip()

        # ✅ Extract only the JSON part using regex
        json_match = re.search(r"\{.*\}", extracted_text, re.DOTALL)

        # if not json_match:
        #     return jsonify({"error": "Failed to extract JSON from response"}), 500

        event_data = json.loads(json_match.group(0)) 
        return jsonify(event_data)
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse JSON from Gemini response"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/correct-grammar', methods=['POST'])
def correct_grammar():
    data = request.json
    text = data.get('text', '')
    voiceInput = data.get('voiceInput', 0)
    print("correct grammer called")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Add the prefix "grammar: " before the text
    # result = happy_tt.generate_text(f"grammar: {text}", args=args)
    if voiceInput==1:
        prompt = f"""
    Please check the grammer and spelling mistakes in the text given and return only the corrected sentence
    Text - {text}
    """
    else:
        prompt=f"""
        
        """
    result = model.generate_content(prompt)
    return jsonify({"corrected_text": result.text})

@app.route('/process-audio', methods=['POST'])
def process_audio():
    # return jsonify({"message": "Endpoint working!"})
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Save uploaded file
    audio_file = request.files['audio']
    file_path = os.path.join("uploads", "audio.wav")
    os.makedirs("uploads", exist_ok=True)  # Ensure uploads folder exists
    audio_file.save(file_path)

    try:
        aai.settings.api_key = os.getenv("api_key")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)
        print(transcript.text)
        return jsonify({"response": transcript.text})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup the saved file
        if os.path.exists(file_path):
            os.remove(file_path)
            
   
   
def extract_text_from_file(file):
    """Extract text from different file formats."""
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension == "txt":
        return file.read().decode('utf-8')

    elif file_extension == "csv":
        df = pd.read_csv(file)
        return df.to_string(index=False)  # Convert CSV to readable text

    elif file_extension == "pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    elif file_extension == "docx":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_extension == "html":
        soup = BeautifulSoup(file.read(), "html.parser")
        return soup.get_text()

    else:
        return None  # Unsupported file type

@app.route('/data-summarizer', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    extracted_text = extract_text_from_file(file)

    user_choice = request.form.get('choice')  # 'summary' or 'query'
      # User's question if they choose 'query'

    prompt =''
    if extracted_text is None:
        return jsonify({'error': 'Unsupported file type'}), 400

    if user_choice == 'summary':
        prompt = f"""
    Please Summarize the given text - {extracted_text}
    """
    elif user_choice == 'query':
        user_query = request.form.get('query', '')
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        prompt = f"""
    From the input text give me the answer to my query
    Input text - {extracted_text}
    
    Query - {user_query}
    """
    else:
        return jsonify({'error': 'Invalid choice'}), 400

    
    response = model.generate_content(prompt)
    # response =''
    print(response.text)

    return jsonify({'content': response.text.split()})

@app.route('/generate-content', methods=['POST'])
def generate_content():
    data = request.json
    promptType = data.get('promptType', '')
    additional = data.get('additional', '')
    voiceInput = data.get('voiceInput', 0)
    print("content generator called")
    prompt=''

    
    # Add the prefix "grammar: " before the text
    # result = happy_tt.generate_text(f"grammar: {text}", args=args)
    if(voiceInput==1) :
        prompt = f"""
    Generator the content for me on the given query and additional details given below
    Query:{promptType}
    additional details - {additional}
    """
    else:
        prompt=f"""
        
        """
    
    result = model.generate_content(prompt)
    return jsonify({"generatedContent": result.text})


@app.route('/generate-graph', methods=['POST'])
def generate_graph():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    print("1st")
    file_data = data.get('data', [])
    x_column = data.get('xColumn').strip()
    y_column = data.get('yColumn').strip()
    graph_type = data.get('graphType').strip()

    if not file_data or not x_column or not y_column or not graph_type:
        return jsonify({"error": "Missing required parameters"}), 400
    
    df = pd.DataFrame(file_data)
    fig = None
    print("2nd")
    if graph_type == "Bar":
        fig = px.bar(df, x=x_column, y=y_column, title=f"{graph_type} Chart")
    elif graph_type == "Line":
        fig = px.line(df, x=x_column, y=y_column, title=f"{graph_type} Chart")
    elif graph_type == "Scatter":
        fig = px.scatter(df, x=x_column, y=y_column, title=f"{graph_type} Chart")
    elif graph_type == "Histogram":
        fig = px.histogram(df, x=x_column, y=y_column, title=f"{graph_type} Chart")
    elif graph_type == "Box Plot":
        fig = px.box(df, x=x_column, y=y_column, title=f"{graph_type}")
    elif graph_type == "Violin Plot":
        fig = px.violin(df, x=x_column, y=y_column, title=f"{graph_type}")
    elif graph_type == "Pie Chart":
        fig = px.pie(df, names=x_column, values=y_column, title=f"{graph_type}")
    print("3rd")
    if fig:
        graph_path = "graph_created.html"  # Save as PNG
        print("4th")

        # if not os.path.exists("images"):
        #     os.mkdir("images")
        # Generate image bytes in PNG format
        # img_bytes = fig.to_image(format="png", width=800, height=600)

        # # Convert the bytes to an image and save as PNG
        # image = Image.open(io.BytesIO(img_bytes))
        # image.save(graph_path, "PNG")
        fig.write_html(graph_path)

        print("5th")

        # Send the file as a response
        return send_file(graph_path)
    
    return jsonify({"error": "Invalid graph type"}), 400
    
@app.route("/")
def home():
    print("Orato Running parteekkkkk........")
    return "Hello, Flask on Azure!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
