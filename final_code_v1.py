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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
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
# Parser Abstraction
# =========================

class FileParser(ABC):
    """Abstract base class for file parsers"""
    
    @abstractmethod
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse the given file and extract content.
        
        Args:
            file_path (str): Path to the file to parse
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing page content and metadata
        """
        pass

class PDFParser(FileParser):
    """Concrete PDF parser implementation"""
    
    def __init__(self, deployment_name: str, api_version: str, azure_api_key: str, azure_endpoint: str):
        """
        Initialize PDF parser with Azure OpenAI configuration.
        
        Args:
            deployment_name (str): Azure OpenAI deployment name
            api_version (str): Azure OpenAI API version
            azure_api_key (str): Azure OpenAI API key
            azure_endpoint (str): Azure OpenAI endpoint
        """
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_version=api_version,
            openai_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            temperature=0
        )
    
    def pdf_page_to_base64(self, pdf_path: str, page_num: int, zoom: int = 5, max_zoom: int = 8, target_width: int = 2500) -> str:
        """
        Convert PDF page to base64 encoded image.
        
        Args:
            pdf_path (str): Path to PDF file
            page_num (int): Page number to convert
            zoom (int): Zoom level for conversion
            max_zoom (int): Maximum zoom level
            target_width (int): Target width for image
            
        Returns:
            str: Base64 encoded image string
        """
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
    
    def extract_text_fallback_pdf(self, pdf_path: str, page_num: int) -> str:
        """
        Extract text from PDF page using fallback method.
        
        Args:
            pdf_path (str): Path to PDF file
            page_num (int): Page number to extract text from
            
        Returns:
            str: Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = doc.load_page(page_num).get_text()
            doc.close()
            return text
        except Exception as e:
            return f"[FALLBACK ERROR: {str(e)}]"
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse PDF file and extract structured content.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            List[Dict[str, Any]]: List of page content dictionaries
        """
        full_content = []
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            print(f"Processing {os.path.basename(file_path)} - Page {page_num+1}/{len(doc)}...")
            page_text = self.extract_text_fallback_pdf(file_path, page_num)
            response = self.llm([HumanMessage(content=CV_PARSER_PROMPT + "\n\n" + page_text)])
            page_content = response.content.strip() if response.content else "[ERROR: Empty GPT response]"
            full_content.append({
                "page": page_num + 1,
                "content": page_content,
                "file_name": os.path.splitext(os.path.basename(file_path))[0]
            })
            time.sleep(1)
        doc.close()
        return full_content

class DOCXParser(FileParser):
    """Concrete DOCX parser implementation"""
    
    def __init__(self, deployment_name: str, api_version: str, azure_api_key: str, azure_endpoint: str):
        """
        Initialize DOCX parser with Azure OpenAI configuration.
        
        Args:
            deployment_name (str): Azure OpenAI deployment name
            api_version (str): Azure OpenAI API version
            azure_api_key (str): Azure OpenAI API key
            azure_endpoint (str): Azure OpenAI endpoint
        """
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_version=api_version,
            openai_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            temperature=0
        )
    
    def extract_text_docx(self, docx_path: str) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            docx_path (str): Path to DOCX file
            
        Returns:
            str: Extracted text content
        """
        try:
            doc = docx.Document(docx_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            return f"[ERROR READING DOCX: {str(e)}]"
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse DOCX file and extract structured content.
        
        Args:
            file_path (str): Path to DOCX file
            
        Returns:
            List[Dict[str, Any]]: List of page content dictionaries
        """
        full_content = []
        print(f"Processing DOCX file: {os.path.basename(file_path)}...")
        docx_text = self.extract_text_docx(file_path)
        response = self.llm([HumanMessage(content=CV_PARSER_PROMPT + "\n\n" + docx_text)])
        page_content = response.content.strip() if response.content else "[ERROR: Empty GPT response]"
        full_content.append({
            "page": 1,
            "content": page_content,
            "file_name": os.path.splitext(os.path.basename(file_path))[0]
        })
        return full_content

class ParserFactory:
    """Factory class to create appropriate parsers based on file type"""
    
    @staticmethod
    def create_parser(file_ext: str, deployment_name: str, api_version: str, azure_api_key: str, azure_endpoint: str) -> FileParser:
        """
        Create appropriate parser based on file extension.
        
        Args:
            file_ext (str): File extension (.pdf or .docx)
            deployment_name (str): Azure OpenAI deployment name
            api_version (str): Azure OpenAI API version
            azure_api_key (str): Azure OpenAI API key
            azure_endpoint (str): Azure OpenAI endpoint
            
        Returns:
            FileParser: Appropriate parser instance
            
        Raises:
            ValueError: If file extension is not supported
        """
        if file_ext == ".pdf":
            return PDFParser(deployment_name, api_version, azure_api_key, azure_endpoint)
        elif file_ext == ".docx":
            return DOCXParser(deployment_name, api_version, azure_api_key, azure_endpoint)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

# =========================
# Field Extractor Abstraction
# =========================

class FieldExtractor(ABC):
    """Abstract base class for field extractors"""
    
    @abstractmethod
    def extract(self, text: str) -> Any:
        """
        Extract specific field from text.
        
        Args:
            text (str): Input text to extract from
            
        Returns:
            Any: Extracted field value
        """
        pass

class NameExtractor(FieldExtractor):
    """Concrete name extractor implementation"""
    
    def extract(self, text: str) -> Optional[str]:
        """
        Extract name from text.
        
        Args:
            text (str): Input text to extract name from
            
        Returns:
            Optional[str]: Extracted name or None if not found
        """
        # Using the original logic - rely on the LLM for extraction
        return None

class EmailExtractor(FieldExtractor):
    """Concrete email extractor implementation"""
    
    def extract(self, text: str) -> Optional[str]:
        """
        Extract email from text.
        
        Args:
            text (str): Input text to extract email from
            
        Returns:
            Optional[str]: Extracted email or None if not found
        """
        # Using the original logic - rely on the LLM for extraction
        return None

class SkillsExtractor(FieldExtractor):
    """Concrete skills extractor implementation"""
    
    def extract(self, text: str) -> List[str]:
        """
        Extract skills from text.
        
        Args:
            text (str): Input text to extract skills from
            
        Returns:
            List[str]: List of extracted skills
        """
        # Using the original logic - rely on the LLM for extraction
        return []

# =========================
# Resume Data Class
# =========================

@dataclass
class ResumeData:
    """Data class to encapsulate resume fields"""
    name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = None
    
    def __post_init__(self):
        """Initialize skills list if not provided."""
        if self.skills is None:
            self.skills = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ResumeData to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of resume data
        """
        return {
            "name": self.name,
            "email": self.email,
            "skills": self.skills
        }

# =========================
# Resume Extraction Coordinator
# =========================

class ResumeExtractor:
    """Orchestrates extraction for all fields and outputs ResumeData instance"""
    
    def __init__(self, client: AzureOpenAI, deployment_name: str):
        """
        Initialize ResumeExtractor with Azure OpenAI client.
        
        Args:
            client (AzureOpenAI): Azure OpenAI client instance
            deployment_name (str): Azure OpenAI deployment name
        """
        self.client = client
        self.deployment_name = deployment_name
    
    def extract(self, text: str) -> ResumeData:
        """
        Extract resume data from text using LLM.
        
        Args:
            text (str): Input text to extract resume data from
            
        Returns:
            ResumeData: Extracted resume data
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            try:
                parsed_data = json.loads(content)
            except json.JSONDecodeError:
                print("⚠️ Model did not return valid JSON. Attempting fix...")
                json_text = re.search(r'\{.*\}', content, re.DOTALL)
                parsed_data = json.loads(json_text.group(0)) if json_text else {"name": None, "email": None, "skills": []}
            
            return ResumeData(
                name=parsed_data.get("name"),
                email=parsed_data.get("email"),
                skills=parsed_data.get("skills", [])
            )
        except Exception as e:
            print(f"❌ Error extracting info: {e}")
            return ResumeData()

# =========================
# Framework Orchestration
# =========================

class ResumeParserFramework:
    """Combines a FileParser and a ResumeExtractor"""
    
    def __init__(self, file_parser: FileParser, resume_extractor: ResumeExtractor):
        """
        Initialize ResumeParserFramework with parser and extractor.
        
        Args:
            file_parser (FileParser): File parser instance
            resume_extractor (ResumeExtractor): Resume extractor instance
        """
        self.file_parser = file_parser
        self.resume_extractor = resume_extractor
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process file and extract content.
        
        Args:
            file_path (str): Path to file to process
            
        Returns:
            List[Dict[str, Any]]: List of extracted page content
        """
        return self.file_parser.parse(file_path)
    
    def extract_resume_info(self, text: str) -> ResumeData:
        """
        Extract resume information from text.
        
        Args:
            text (str): Input text to extract resume info from
            
        Returns:
            ResumeData: Extracted resume data
        """
        return self.resume_extractor.extract(text)

# =========================
# Utility Functions
# =========================

def save_extracted_content(extracted_pages: List[Dict[str, Any]], file_name: str, output_dir: str) -> str:
    """
    Save extracted content to markdown file.
    
    Args:
        extracted_pages (List[Dict[str, Any]]): List of extracted page content
        file_name (str): Original file name
        output_dir (str): Output directory path
        
    Returns:
        str: Path to saved markdown file
    """
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

# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    """
    Main execution function for resume parsing framework.
    
    Processes PDF and DOCX files from input directory, extracts resume information,
    and saves results as markdown and JSON files in output directory.
    """
    input_dir = "/Users/soumendusekharbhattacharjee/Documents/DATA-SCIENCE/Policy_Reporter/data/Input_data"
    output_dir = "/Users/soumendusekharbhattacharjee/Documents/DATA-SCIENCE/Policy_Reporter/data/Output_data"
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".pdf", ".docx"))]

    if not files:
        print("⚠️ No PDF or DOCX files found in the input directory.")
    else:
        print(f"📁 Found {len(files)} file(s) to process.\n")

        # Initialize Azure OpenAI Client
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=api_version
        )

        # Create resume extractor
        resume_extractor = ResumeExtractor(client, deployment_name)

        for i, file_name in enumerate(files, 1):
            file_path = os.path.join(input_dir, file_name)
            print(f"\n🚀 [{i}/{len(files)}] Processing: {file_name}")

            try:
                # Get file extension and create appropriate parser
                file_ext = os.path.splitext(file_name)[1].lower()
                parser = ParserFactory.create_parser(file_ext, deployment_name, api_version, azure_api_key, azure_endpoint)
                
                # Create framework
                framework = ResumeParserFramework(parser, resume_extractor)
                
                # Step 1: Extract Markdown from PDF/DOCX
                extracted_pages = framework.process_file(file_path)
                md_file = save_extracted_content(extracted_pages, file_name, output_dir)
                print(f"💾 Saved Markdown: {md_file}")

                # Step 2: Extract structured info from Markdown
                with open(md_file, "r", encoding="utf-8") as f:
                    text = f.read()

                # Use the framework to extract resume data
                resume_data = framework.extract_resume_info(text)
                
                # Save as JSON
                json_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".json")
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(resume_data.to_dict(), f, indent=2, ensure_ascii=False)

                print(f"💾 Saved JSON: {json_file}\n")

            except Exception as e:
                print(f"❌ Critical error processing {file_name}: {e}")

        print("\n🎉 All files processed successfully!")
