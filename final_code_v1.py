"""
Resume Parsing Framework using Azure OpenAI (Prompt-based extraction)

This module extracts structured resume information purely using Azure OpenAI prompts.
"""

# =========================
# Imports
# =========================
import os
import json
import re
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
deployment_name = os.getenv('DEPLOYMENT_NAME')
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
CV_PARSER_PROMPT = """
You are an expert CV parser. Extract all structured information from the given CV text.
Output the data in clear markdown sections as shown below:

### Personal Information
- Name:
- Email:
- Phone:
- Location:
- LinkedIn / GitHub (if any):

### Education
| Degree | Institution | Year | Major/Field |
|---------|--------------|------|--------------|

### Work Experience
| Job Title | Company | Duration | Key Responsibilities |
|------------|----------|-----------|-----------------------|

### Skills
- Technical Skills:
- Soft Skills:

### Certifications (if available)
| Certification | Organization | Year |

### Projects (if mentioned)
| Project | Description | Tools Used |

### Additional Information
[Any extra relevant details found in the CV]

**Note:** Preserve all tables in correct markdown format. Do not hallucinate or guess missing data.
"""

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
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()


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
    """
    Uses Azure OpenAI to extract structured JSON from resume text using prompts.
    """

    def __init__(self, client, deployment_name):
        self.client = client
        self.deployment_name = deployment_name

    def parse_markdown(self, text: str) -> str:
        """Optional: Generate structured CV in markdown using CV_PARSER_PROMPT."""
        prompt = CV_PARSER_PROMPT + f"\n\nResume Text:\n{text}"
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def extract_json_fields(self, text: str) -> ResumeData:
        """Extract only name, email, and skills as JSON using EXTRACTION_PROMPT with robust parsing."""
        prompt = EXTRACTION_PROMPT + f"\n\nResume Text:\n{text}"
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()

        # Extract JSON object from response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in Azure response:\n{content}")

        try:
            data = json.loads(match.group(0))
            return ResumeData(
                name=data.get("name", ""),
                email=data.get("email", ""),
                skills=data.get("skills", [])
            )
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from Azure response:\n{content}")


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

        # Optional: structured markdown version (not used in JSON extraction)
        _ = self.extractor.parse_markdown(text)

        # Extract final JSON fields
        resume_data = self.extractor.extract_json_fields(text)
        return resume_data


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

