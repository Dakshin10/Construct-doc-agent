# Construct AI — Structural Drawing Reader to AlloyDB Workflow (`beta_salitsteels`)

An end-to-end AI workflow built with **Google Agent Development Kit (ADK)** and **Gemini 3.6 Flash / Flash Lite** that extracts structural schedule tables and 2D diagram element quantities from PDF drawings, parses geometric and schedule attributes, formats structured payloads, and commits records into an **AlloyDB PostgreSQL** database.

---

## 📁 Detailed Directory & Workflow Architecture

```text
beta_salitsteels/
├── main.py                         # ADK Workflow Orchestrator (Extractor Agent -> Database Agent)
├── runner_extract_test.py          # Standalone PDF extraction & tabular CLI execution runner
├── requirements.txt                # Core Python package dependencies
├── requirements_gcp.txt            # GCP & AlloyDB specific package requirements
├── output_table.json               # Extracted output payload (JSON format)
├── output_table.csv                # Formatted CSV dump of extracted structural schedules
├── .env                            # Active environment configuration (API keys, DB host)
├── .env.template                   # Configuration template
├── agents/                         # Google ADK Multi-Agent System
│   ├── extractor_agent/            # Extractor & Vision Parser Agent module
│   │   ├── agent.py                # Root orchestrator agent definition (`reader_to_database_workflow_agent`)
│   │   ├── tools.py                # ADK Tool definitions (`extract_pdf_drawing_data`, `send_to_database_agent`)
│   │   ├── field_extraction_agent.py # Full heuristic + Vision extraction engine (Schedule tables & Plan marks)
│   │   ├── pdf_vision_loader.py    # PyMuPDF / PDF renderer & Gemini Vision prompt generator
│   │   └── json_exporter.py       # Exporter utility for CSV & JSON payload formatting
│   └── database_agent/             # AlloyDB Database Persistence Agent module
│       ├── agent.py                # Database root agent definition (`alloydb_insert_agent`)
│       ├── tools.py                # ADK Database tools (`insert_foundation_element_records_batch`, etc.)
│       ├── db.py                   # AlloyDB connection manager & psycopg cursor utilities
│       └── models.py               # Pydantic data schemas for foundation records validation
└── tests/                          # Automated Pytest suite and geometric/parsing probe tools
    ├── test_extraction_pipeline.py # End-to-end pipeline unit tests
    ├── test_rotation_invariance.py # Tests for bounding box normalization across PDF page rotations
    ├── test_geometric_bearing.py   # Spatial bearing & mark-to-geometry association tests
    ├── probe_schedule_debug.py     # Diagnostic probe for structural schedule table bounds
    ├── probe_pairing_debug.py      # Diagnostic probe for plan instances mark-to-shape pairing
    └── probe_geometry.py           # Geometric bounding box analysis tool
```

---

## 🤖 ADK Multi-Agent Workflow Details

The system operates as a coordinated **2-Agent Workflow** executed via Google ADK (`InMemoryRunner`):

### 1. Extractor Agent (`agents/extractor_agent`)
- **`agent.py`**: Defines `root_agent` (`reader_to_database_workflow_agent`). Orchestrates drawing extraction and transfers structured payloads to the database agent.
- **`tools.py`**: Wraps the extraction engine (`extract_pdf_drawing_data`) and database bridge (`send_to_database_agent`).
- **`field_extraction_agent.py`**: Core intelligence layer containing multi-stage parsing logic:
  - **Schedule Table Extraction**: Identifies footing/column/wall schedule tables (Marks, Length, Width, Depth, Top/Bottom Reinforcement).
  - **Plan Element Quantity (QTY) Count**: Scans 2D drawing vector paths and text marks, pairing marks to geometric bounding shapes (footings, columns, grade beams) with spatial bearing offsets and rotation invariance.
  - **Fallback Gemini Vision Extraction**: Utilizes Gemini 3.6 Flash Vision when high-complexity schedules require multimodal visual comprehension.
- **`pdf_vision_loader.py`**: Converts PDF pages into high-DPI images and structures multi-modal prompts for Gemini Vision API calls.
- **`json_exporter.py`**: Merges schedule properties with 2D plan counts, exporting standardized results into `output_table.json` and `output_table.csv`.

### 2. Database Agent (`agents/database_agent`)
- **`agent.py`**: Defines `root_agent` (`alloydb_insert_agent`).
- **`tools.py`**: Provides single record (`insert_foundation_element_record`) and batch insertion (`insert_foundation_element_records_batch`) capabilities into AlloyDB PostgreSQL.
- **`db.py`**: Manages PostgreSQL connections using `psycopg` (v3) over AlloyDB Auth Proxy with auto-reconnect and schema setup (`salit_steel_foundation_diagram`).
- **`models.py`**: Validates record data schema (Type, Category, Dimensions, Reinforcement, Quantity, Source PDF).

---

## ⚡ Setup & Environment Configuration

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file from `.env.template`:
```env
# Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash

# AlloyDB PostgreSQL Database Connection
ALLOYDB_HOST=127.0.0.1
ALLOYDB_PORT=5432
ALLOYDB_DATABASE=postgres
ALLOYDB_USER=postgres
ALLOYDB_PASSWORD=your_alloydb_password_here
```

### 3. AlloyDB Auth Proxy
Start the Cloud AlloyDB Auth Proxy in a separate terminal:
```powershell
alloydb-auth-proxy "projects/salitsteel-502008/locations/asia-south1/clusters/slait-steel/instances/primary" --public-ip
```

---

## 🚀 Execution & Usage

### 1. Full ADK End-to-End Workflow (Extractor Agent -> Database Agent)
Run the orchestrator pipeline to process the drawing PDF and insert records into AlloyDB:
```powershell
python main.py --pdf "path/to/structural_drawing.pdf"
```

### 2. Standalone Extraction Runner (Extract & Generate JSON/CSV Output)
Run extraction only without inserting records into AlloyDB:
```powershell
python runner_extract_test.py --pdf "path/to/structural_drawing.pdf"
```

### 3. Run Automated Tests & Diagnostic Probes
Execute Pytest suite for extraction accuracy, spatial pairing, and rotation invariance:
```powershell
# Run full test suite
pytest tests/

# Run targeted rotation invariance test
pytest tests/test_rotation_invariance.py

# Run schedule probe diagnostic tool
python tests/probe_schedule_debug.py
```

---

## 📊 Workflow Outputs & Persistence

- **`output_table.json`**: Structured array of footing schedule objects including dimensions, rebar specs, and quantity counts.
- **`output_table.csv`**: Tabular CSV export of extracted schedules for spreadsheet review.
- **AlloyDB Table (`salit_steel_foundation_diagram`)**: UPSERT record persistence containing:
  - `element_type` (e.g. `F1`, `F2`, `SF1`)
  - `category` (e.g. `Footing Schedule`)
  - `length_mm`, `width_mm`, `depth_mm`
  - `primary_reinforcement`, `secondary_reinforcement`
  - `plan_instances` (Extracted QTY count from 2D diagram)
  - `source_pdf` (Filename of drawing)

