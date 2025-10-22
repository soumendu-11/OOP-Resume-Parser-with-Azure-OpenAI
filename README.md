# Resume Parsing Framework using Azure OpenAI

This project provides an **extensible, prompt-driven resume parsing framework** using **Azure OpenAI (GPT-4o)**. It parses resumes in PDF or DOCX format, extracts structured information such as **Name**, **Email**, and **Skills**, and outputs the results as JSON files.

---

## Features

- Parse **PDF** and **Word (.docx)** resumes.
- Extract key fields:
  - `name`
  - `email`
  - `skills`
- Optional: Generate full structured CV in **Markdown** format.
- Fully **prompt-driven** using Azure OpenAI; **no regex used**.
- Robust JSON parsing to handle minor formatting inconsistencies.
- Deterministic results using `temperature=0`.

---


## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/oop-resume-parser.git
cd oop-resume-parser
```

2. Install required dependencies:
```
pip install -r requirements.txt
```
3. Run the parser:
```
python final_code_v1.py

```
