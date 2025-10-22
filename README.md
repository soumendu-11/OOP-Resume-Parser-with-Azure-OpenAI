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
