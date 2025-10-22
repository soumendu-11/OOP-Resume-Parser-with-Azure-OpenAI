"""
Resume Parsing Framework using Azure OpenAI GPT-4o
Optimized for long PDFs with chunking + fallback base64 image extraction
"""

# =========================
# Imports
# =========================
import os
import json
import re
import time
import base64
import fitz
import docx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import AzureOpenAI

# =========================
# Azure Configuration
# =========================
load_dotenv()

azure_api_key = os.getenv('AZURE_API_KEY')
azure_endpoint = os.getenv('AZURE_ENDPOINT')
deployment_name = os.getenv('DEPLOYMENT_NAME')  # GPT-4o deployment
api_version = os.getenv('API_VERSION')

if not all([azure_api_key, azure_endpoint, deployment_name, api_version]):
    raise EnvironmentError("Missing required Azure OpenAI environment variables.")

client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version=api_version
)

# =========================
# Prompts
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
- "name" should be the person's full name.
- "email" should be a valid email address.
- "skills" should be a list of key skills.
- Do not add any explanations or extra text — only return pure JSON.
"""

# =========================
# Abstract Base Classes
# =========================
class FileParser(ABC):
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        pass

# =========================
# Concrete File Parsers
# =========================
class PDFParser(FileParser):
    def extract_text(self, file_path: str) -> str:
        """Extract all text from PDF pages; fallback to base64 image OCR if empty"""
        doc = fitz.open(file_path)
        full_text = ""
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text().strip()
            if not text:
                # fallback base64 image extraction
                text = self.extract_text_from_image(file_path, i)
            full_text += text + "\n"
        doc.close()
        return full_text.strip()

    def extract_text_from_image(self, pdf_path, page_num, zoom=2, target_width=1200):
        """Fallback: Convert PDF page to base64 image and run GPT-4o"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        rect = page.rect
        scale_factor = target_width / max(rect.width, rect.height)
        adaptive_zoom = max(zoom, scale_factor)
        mat = fitz.Matrix(adaptive_zoom, adaptive_zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        base64_img = base64.b64encode(img_bytes).decode("utf-8")
        doc.close()

        return self.run_gpt40_with_base64(base64_img)

    def run_gpt40_with_base64(self, base64_img, retry_count=3):
        """Send base64 image to GPT-4o for text extraction"""
        for attempt in range(retry_count):
            try:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "user", "content": f"[IMAGE BASE64]{base64_img}"}
                    ],
                    temperature=0
                )
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} failed: {e}")
                time.sleep(1)
        return "[ERROR: Failed to extract text]"

class WordParser(FileParser):
    def extract_text(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]).strip()

# =========================
# ResumeData Data Class
# =========================
@dataclass
class ResumeData:
    name: str
    email: str
    skills: list

# =========================
# Azure-based Field Extractor
# =========================
class AzureFieldExtractor:
    """Uses GPT-4o to extract JSON fields from resume text"""
    def __init__(self, client, deployment_name):
        self.client = client
        self.deployment_name = deployment_name

    def extract_json_fields(self, text: str) -> ResumeData:
        # Chunk text if too long
        max_chunk_length = 50000  # 50k chars per chunk
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        combined_data = {"name": "", "email": "", "skills": []}

        for chunk in chunks:
            prompt = EXTRACTION_PROMPT + f"\n\nResume Text:\n{chunk}"
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    # Merge results
                    if data.get("name") and not combined_data["name"]:
                        combined_data["name"] = data["name"]
                    if data.get("email") and not combined_data["email"]:
                        combined_data["email"] = data["email"]
                    if data.get("skills"):
                        combined_data["skills"].extend([s for s in data["skills"] if s not in combined_data["skills"]])
                except json.JSONDecodeError:
                    continue

        return ResumeData(**combined_data)

# =========================
# ResumeParserFramework
# =========================
class ResumeParserFramework:
    def __init__(self):
        self.parsers = {
            ".pdf": PDFParser(),
            ".docx": WordParser()
        }
        self.extractor = AzureFieldExtractor(client, deployment_name)

    def parse_resume(self, file_path: str) -> ResumeData:
        _, ext = os.path.splitext(file_path)
        parser = self.parsers.get(ext.lower())
        if not parser:
            raise ValueError(f"Unsupported file type: {ext}")
        text = parser.extract_text(file_path)
        if not text.strip():
            raise ValueError(f"Empty text extracted from: {file_path}")
        return self.extractor.extract_json_fields(text)

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    input_dir = "/Users/soumendusekharbhattacharjee/Documents/DATA-SCIENCE/Policy_Reporter/data/Input_data"
    output_dir = "/Users/soumendusekharbhattacharjee/Documents/DATA-SCIENCE/Policy_Reporter/data/Output_data"
    os.makedirs(output_dir, exist_ok=True)

    framework = ResumeParserFramework()
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".pdf", ".docx"))]

    if not files:
        print("⚠️ No resume files found.")
    else:
        print(f"📁 Found {len(files)} resume(s) to process.\n")
        for i, file in enumerate(files, 1):
            file_path = os.path.join(input_dir, file)
            print(f"🚀 [{i}/{len(files)}] Parsing: {file}")
            try:
                resume_data = framework.parse_resume(file_path)
                json_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(resume_data.__dict__, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved structured data → {json_path}\n")
            except Exception as e:
                print(f"❌ Error parsing {file}: {e}\n")

        print("🎉 All resumes processed successfully!")

