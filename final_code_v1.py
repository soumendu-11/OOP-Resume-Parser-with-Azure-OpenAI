"""
Resume Parsing Framework using Azure OpenAI

This module provides an extensible OOP-based architecture to parse resumes (PDF/DOCX),
extract structured text, and identify fields such as Name, Email, and Skills using Azure OpenAI.

Implements:
- Parser abstraction for PDF and Word
- Field extractor abstraction for specific resume fields
- ResumeData class for encapsulating extracted information
- ResumeExtractor orchestrator to manage field extraction
- ResumeParserFramework to combine file parsing and LLM-based field extraction

Author: Soumendu Sekhar Bhattacharjee
Date: 2025-10-22
"""

# =========================
# Imports
# =========================
import os
import re
import json
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
# Abstract Base Classes
# =========================
class FileParser(ABC):
    """Abstract base class for file parsers."""

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract text from the provided file path."""
        pass


class FieldExtractor(ABC):
    """Abstract base class for field extractors."""

    @abstractmethod
    def extract(self, text: str, client: AzureOpenAI, deployment_name: str) -> str:
        """Extract a specific field from resume text using Azure OpenAI."""
        pass


# =========================
# Concrete File Parsers
# =========================
class PDFParser(FileParser):
    """Concrete parser for extracting text from PDF files."""

    def extract_text(self, file_path: str) -> str:
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()


class WordParser(FileParser):
    """Concrete parser for extracting text from Word (.docx) files."""

    def extract_text(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text).strip()


# =========================
# ResumeData Data Class
# =========================
@dataclass
class ResumeData:
    """Data class encapsulating extracted resume fields."""
    name: str
    email: str
    skills: list


# =========================
# Concrete Field Extractors
# =========================
class NameExtractor(FieldExtractor):
    """Extractor for candidate name."""

    def extract(self, text: str, client: AzureOpenAI, deployment_name: str) -> str:
        prompt = f"Extract the candidate's full name from this resume text:\n\n{text}\n\nOnly return the name."
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()


class EmailExtractor(FieldExtractor):
    """Extractor for email address."""

    def extract(self, text: str, client: AzureOpenAI, deployment_name: str) -> str:
        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        return match.group(0) if match else None


class SkillsExtractor(FieldExtractor):
    """Extractor for skills using Azure OpenAI."""

    def extract(self, text: str, client: AzureOpenAI, deployment_name: str) -> list:
        prompt = f"""
        Extract a list of technical and soft skills from this resume text.
        Return only a JSON array of skills. Example: ["Python", "Machine Learning", "Leadership"]
        Text:
        {text}
        """
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()

        try:
            skills = json.loads(content)
            if isinstance(skills, list):
                return skills
        except json.JSONDecodeError:
            # fallback regex if JSON fails
            skills = re.findall(r'\b[A-Z][a-zA-Z0-9+\-# ]{2,}\b', content)
        return skills or []


# =========================
# ResumeExtractor Orchestrator
# =========================
class ResumeExtractor:
    """Coordinates extraction for all resume fields using provided field extractors."""

    def __init__(self, extractors: dict):
        """
        Initialize with a dictionary of extractors.
        Example:
            extractors = {
                "name": NameExtractor(),
                "email": EmailExtractor(),
                "skills": SkillsExtractor()
            }
        """
        self.extractors = extractors

    def extract_all(self, text: str, client: AzureOpenAI, deployment_name: str) -> ResumeData:
        """Run all extractors and return a ResumeData instance."""
        name = self.extractors["name"].extract(text, client, deployment_name)
        email = self.extractors["email"].extract(text, client, deployment_name)
        skills = self.extractors["skills"].extract(text, client, deployment_name)
        return ResumeData(name=name, email=email, skills=skills)


# =========================
# ResumeParserFramework
# =========================
class ResumeParserFramework:
    """
    Orchestrates file parsing and resume field extraction.

    Provides a single interface:
        parse_resume(file_path: str) -> ResumeData
    """

    def __init__(self):
        self.parsers = {
            ".pdf": PDFParser(),
            ".docx": WordParser()
        }
        self.extractor = ResumeExtractor({
            "name": NameExtractor(),
            "email": EmailExtractor(),
            "skills": SkillsExtractor()
        })

    def parse_resume(self, file_path: str) -> ResumeData:
        """Identify file type, parse text, and extract structured resume data."""
        _, ext = os.path.splitext(file_path)
        parser = self.parsers.get(ext.lower())

        if not parser:
            raise ValueError(f"Unsupported file type: {ext}")

        text = parser.extract_text(file_path)
        if not text.strip():
            raise ValueError(f"Empty text extracted from: {file_path}")

        resume_data = self.extractor.extract_all(text, client, deployment_name)
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

