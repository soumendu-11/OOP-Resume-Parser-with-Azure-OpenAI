# OOP Resume Parser with Azure OpenAI

This project provides a modular, object-oriented framework to parse resumes in PDF or DOCX formats and extract structured information such as **Name**, **Email**, and **Skills** using **Azure OpenAI** (GPT-4o or similar). The framework is designed for scalability, extensibility, and maintainability.

---

## Features

- **Parser Abstraction**: Easily extendable file parsers for PDF, Word, and more.
- **Field Extractor Abstraction**: Extract fields like Name, Email, and Skills individually.
- **ResumeData Class**: Encapsulates all extracted fields into a structured dataclass.
- **Resume Extraction Coordinator**: Orchestrates field extraction for accurate structured output.
- **Framework Orchestration**: Combines file parsing and field extraction with a single interface.

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/oop-resume-parser.git
cd oop-resume-parser
```
```
2. Install required dependencies:

pip install -r requirements.txt
```
Install required dependencies:

pip install -r requirements.txt


Create a .env file in the project root and add your Azure OpenAI credentials:

AZURE_API_KEY=your_azure_api_key
AZURE_ENDPOINT=your_azure_endpoint
DEPLOYMENT_NAME=your_model_deployment_name
API_VERSION=your_api_version


Place your resumes (PDF or DOCX) in the Input_data folder. The extracted JSON files will be saved in the Output_data folder.

Run the parser with:

python final_code_v1.py

Output

For each resume file, a JSON file with the same base name will be created in Output_data, containing:

{
  "name": "Full Name",
  "email": "email@example.com",
  "skills": ["Skill1", "Skill2", "Skill3"]
}

Folder Structure
.
├── Input_data/           # Place your PDF/DOCX resumes here
├── Output_data/          # Parsed JSON files will be saved here
├── final_code_v1.py      # Main parser framework
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

