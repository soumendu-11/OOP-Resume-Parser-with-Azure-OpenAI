# =========================
# Imports
# =========================
import os
import time
import json
import re
import base64
import fitz  # PyMuPDF
import docx
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from openai import AzureOpenAI  # Azure OpenAI SDK

# =========================
# Load environment variables
# =========================
load_dotenv()

azure_api_key = os.getenv('AZURE_API_KEY')
azure_endpoint = os.getenv('AZURE_ENDPOINT')
deployment_name = os.getenv('DEPLOYMENT_NAME')
api_version = os.getenv('API_VERSION')

required_vars = {
    'AZURE_API_KEY': azure_api_key,
    'AZURE_ENDPOINT': azure_endpoint,
    'DEPLOYMENT_NAME': deployment_name,
    'API_VERSION': api_version
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

print("✅ Azure OpenAI Configuration Loaded")

# =========================
# CV Parser Prompt (Markdown)
# =========================
CV_PARSER_PROMPT = """
You are an expert CV parser. Your task is to extract all structured information from the given CV text.
Follow these instructions carefully:

1. Extract all visible text from the CV.
2. Maintain the correct markdown format.
3. If there are any tables, convert them into proper markdown table format using pipes `|` and dashes `-`.
4. Preserve all information, but do not hallucinate or guess missing data.
5. Output your response exactly in the sections below.
"""

# =========================
# Resume JSON Extraction Prompt
# =========================
EXTRACTION_PROMPT = """
You are an expert resume parser.
Extract ONLY the following fields from the resume text and return a valid JSON object strictly in this format:

{
  "name": string,
  "email": string,
  "skills": [string]
}

Rules:
- "name" should be the person's full name (if multiple found, choose the most likely).
- "email" should be a valid email address.
- "skills" should be a list of key skills (technical or soft skills).
- Do not add any explanations or extra text — only return pure JSON.
"""

# =========================
# Functions
# =========================

def pdf_page_to_base64(pdf_path, page_num, zoom=5, max_zoom=8, target_width=2500):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    rect = page.rect
    width, height = rect.width, rect.height
    scale_factor = target_width / max(width, height)
    adaptive_zoom = min(max_zoom, max(zoom, scale_factor))
    mat = fitz.Matrix(adaptive_zoom, adaptive_zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    base64_img = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    doc.close()
    return base64_img

def extract_text_fallback_pdf(pdf_path, page_num):
    try:
        doc = fitz.open(pdf_path)
        text = doc.load_page(page_num).get_text()
        doc.close()
        return text
    except Exception as e:
        return f"[FALLBACK ERROR: {str(e)}]"

def extract_text_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"[ERROR READING DOCX: {str(e)}]"

def process_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    full_content = []

    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        openai_api_version=api_version,
        openai_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        temperature=0
    )

    if file_ext == ".pdf":
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            print(f"Processing {os.path.basename(file_path)} - Page {page_num+1}/{len(doc)}...")
            page_text = extract_text_fallback_pdf(file_path, page_num)
            response = llm([HumanMessage(content=CV_PARSER_PROMPT + "\n\n" + page_text)])
            page_content = response.content.strip() if response.content else "[ERROR: Empty GPT response]"
            full_content.append({
                "page": page_num + 1,
                "content": page_content,
                "file_name": os.path.splitext(os.path.basename(file_path))[0]
            })
            time.sleep(1)
        doc.close()

    elif file_ext == ".docx":
        print(f"Processing DOCX file: {os.path.basename(file_path)}...")
        docx_text = extract_text_docx(file_path)
        response = llm([HumanMessage(content=CV_PARSER_PROMPT + "\n\n" + docx_text)])
        page_content = response.content.strip() if response.content else "[ERROR: Empty GPT response]"
        full_content.append({
            "page": 1,
            "content": page_content,
            "file_name": os.path.splitext(os.path.basename(file_path))[0]
        })

    else:
        print(f"⚠️ Unsupported file type: {file_ext}")

    return full_content

def save_extracted_content(extracted_pages, file_name, output_dir):
    output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# CV EXTRACTION REPORT\n\n")
        f.write(f"**Source File:** {file_name}\n\n")
        f.write(f"**Total Pages:** {len(extracted_pages)}\n\n")
        f.write(f"**Extraction Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n")
        for page_data in extracted_pages:
            f.write(f"## PAGE {page_data['page']} - {file_name}\n\n---\n\n")
            f.write(page_data["content"])
            f.write(f"\n\n[END OF PAGE {page_data['page']}]\n\n---\n\n")
    return output_file

# Initialize Azure OpenAI Client for JSON extraction
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version=api_version
)

def extract_resume_info(text):
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("⚠️ Model did not return valid JSON. Attempting fix...")
            json_text = re.search(r'\{.*\}', content, re.DOTALL)
            return json.loads(json_text.group(0)) if json_text else {"name": None, "email": None, "skills": []}
    except Exception as e:
        print(f"❌ Error extracting info: {e}")
        return {"name": None, "email": None, "skills": []}

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    input_dir = "/Users/soumendusekharbhattacharjee/Documents/DATA-SCIENCE/Policy_Reporter/data/Input_data"
    output_dir = "/Users/soumendusekharbhattacharjee/Documents/DATA-SCIENCE/Policy_Reporter/data/Output_data"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".pdf", ".docx"))]

    if not files:
        print("⚠️ No PDF or DOCX files found in the input directory.")
    else:
        print(f"📁 Found {len(files)} file(s) to process.\n")

        for i, file_name in enumerate(files, 1):
            file_path = os.path.join(input_dir, file_name)
            print(f"\n🚀 [{i}/{len(files)}] Processing: {file_name}")

            try:
                # Step 1: Extract Markdown from PDF/DOCX
                extracted_pages = process_file(file_path)
                md_file = save_extracted_content(extracted_pages, file_name, output_dir)
                print(f"💾 Saved Markdown: {md_file}")

                # Step 2: Extract structured JSON info from Markdown
                with open(md_file, "r", encoding="utf-8") as f:
                    text = f.read()

                parsed_data = extract_resume_info(text)
                json_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".json")
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(parsed_data, f, indent=2, ensure_ascii=False)

                print(f"💾 Saved JSON: {json_file}\n")

            except Exception as e:
                print(f"❌ Critical error processing {file_name}: {e}")

        print("\n🎉 All files processed successfully!")

