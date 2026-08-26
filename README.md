# FinAlze OCR PoC

Banking financial statement digitization platform for balance sheets, P&L statements, and cash flow statements.

## Quick Start

```bash
# Activate virtual environment
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run sample extraction
python src/pipeline.py --input Samples/bilan1.docx --output output/
```

## Project Structure

```
OCR/
├── context/              # Research documents and plans
├── Samples/              # Test documents (7 files)
├── src/                  # Source code
│   ├── pipeline.py       # Main orchestrator
│   ├── preprocessing/    # Language detection, doc-type classifier
│   ├── extraction/       # OCR engines (Docling, Marker, Surya)
│   ├── accounting/       # NER, schema mapping, validation
│   └── hitl/             # Streamlit validation UI
├── schemas/              # Pandera schemas for IFRS/NCT/SYSCOHADA
├── output/               # Extraction results
├── tests/                # Unit and integration tests
└── requirements.txt      # Python dependencies
```

## Supported Accounting Standards

- **IFRS** — International Financial Reporting Standards
- **NCT** — Normes Comptables Tunisiennes
- **SYSCOHADA** — Système Comptable OHADA

## Tech Stack

| Component | Library |
|-----------|---------|
| DOCX Processing | Marker |
| Table Extraction | Docling + TableFormer |
| OCR Engine | EasyOCR (via Docling) |
| Layout Analysis | Surya |
| DataFrame Validation | Pandera |
| Record Validation | Pydantic |
| HITL UI | Streamlit |

## Phase 0 Status

- [ ] Project structure created
- [ ] Dependencies installed
- [ ] Sample document processing verified
- [ ] Pandera schemas defined
- [ ] Document type classifier prototype

---

**Version**: 0.1.0  
**Status**: Phase 0 — Setup
