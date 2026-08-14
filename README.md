# Construct AI — Structural Drawing Reader to AlloyDB Workflow (`beta_salitsteels`)

An end-to-end AI workflow built with **Google Agent Development Kit (ADK)** and **Gemini 3.6 Flash Vision** that extracts structural schedule tables and 2D diagram element quantities from PDF drawings, formats the structured data, and commits the records into an **AlloyDB PostgreSQL** database.

---

## 🏗️ System Architecture

```text
beta_salitsteels/
├── .env                    # Unified environment configuration
├── .env.template           # Template for environment variables
├── main.py                 # Primary entry point for the end-to-end workflow
├── runner_extract_test.py  # Standalone extraction runner
├── agents/
│   ├── extractor_agent/    # PDF Vision Loader, Schedule Extractor, & JSON Exporter
│   │   ├── agent.py
│   │   ├── tools.py
│   │   ├── field_extraction_agent.py
│   │   ├── json_exporter.py
│   │   └── pdf_vision_loader.py
│   └── database_agent/     # AlloyDB ADK Database Agent & Connection Tools
│       ├── agent.py
│       ├── tools.py
│       ├── db.py
│       └── models.py
└── tests/                  # Pytest regression suite
```

---

## ⚡ Prerequisites & Setup

### 1. Requirements
- **Python 3.10+** (tested up to Python 3.14)
- **Google Cloud AlloyDB** instance with Auth Proxy running on `127.0.0.1:5432`

### 2. Install Dependencies
Install all required Python packages into your active environment:

```powershell
pip install -r requirements.txt
```

### 3. Start AlloyDB Auth Proxy (Cloud Connectivity)
Before running the database workflow, start the Google Cloud AlloyDB Auth Proxy in a separate terminal window to connect your local machine securely to your Cloud AlloyDB instance:

```powershell
# Standard Proxy command (or using C:\Users\daksh\bin\alloydb-auth-proxy.exe)
alloydb-auth-proxy "projects/salitsteel-502008/locations/asia-south1/clusters/slait-steel/instances/primary"

# Or if connecting via Public IP:
alloydb-auth-proxy "projects/salitsteel-502008/locations/asia-south1/clusters/slait-steel/instances/primary" --public-ip
```

---

## 🔑 Environment Configuration

Copy `.env.template` to `.env` in the root directory and configure your credentials:

```powershell
cp .env.template .env
```

Set the following variables in `.env`:

```env
# Gemini API Keys
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash

# AlloyDB Database Connection (via Auth Proxy)
ALLOYDB_HOST=127.0.0.1
ALLOYDB_PORT=5432
ALLOYDB_DATABASE=postgres
ALLOYDB_USER=postgres
ALLOYDB_PASSWORD=your_alloydb_password_here
```

---

## 🗄️ Database Table Setup

Before running the workflow for the first time, ensure the target table `structural_footing_schedule` exists in AlloyDB. You can initialize it using this Python command:

```powershell
python -c "import psycopg; from agents.database_agent.db import get_connection; conn = get_connection(); conn.execute('CREATE TABLE IF NOT EXISTS structural_footing_schedule (footing_type VARCHAR(50) PRIMARY KEY, length VARCHAR(50) NOT NULL, width VARCHAR(50) NOT NULL, depth VARCHAR(50) NOT NULL, longitudinal_bars TEXT, lateral_bars TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);'); conn.commit(); print('Table structural_footing_schedule ready!')"
```

---

## 🚀 Execution Commands

### 1. Run Full End-to-End Workflow (Recommended)
Run the main orchestrator workflow to extract PDF data, send the JSON payload to the Database Agent, and stage results:

```powershell
# Run with default PDF path
python main.py

# Or specify a custom PDF file path
python main.py --pdf "C:\path\to\your\structural_drawing.pdf"
```

### 2. Run Standalone Extractor Agent
To extract structural tables and print the formatted grid without committing to the database:

```powershell
python runner_extract_test.py --pdf "C:\path\to\your\structural_drawing.pdf"
```

### 3. Run Pytest Suite
Run the automated regression test suite to verify table parsing, element pairing, and rotation invariance:

```powershell
pytest tests/
```

---

## 📊 Summary of Outputs

- **`output_table.json`**: Formatted JSON output of all extracted structural elements, dimensions, QTY counts, and rebar specifications.
- **AlloyDB Table (`structural_footing_schedule`)**: Parameterized UPSERT containing footing type, length, width, depth, longitudinal bars, and lateral bars.
