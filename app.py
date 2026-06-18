import os
# Vercel-compatible matplotlib setup
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, render_template, make_response
from google import genai
from google.genai import types
import fitz
import pandas as pd
import json
import re
import io
import zipfile
import uuid
import base64
from datetime import datetime
from jinja2 import Template

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-key-for-session")

MODEL_NAME = "gemini-2.5-flash-lite"

# ----------------------------- VERCEL LIMITATIONS NOTE -----------------------------
# VERCEL HOBBY LIMITATION: Serverless functions on the Hobby tier have a strict 10-second timeout. 
# Generating long academic sections (which involves sequential Gemini calls) requires 20-60 seconds.
# This deployment will timeout on Vercel Hobby. Deploy on Vercel Pro and use the maxDuration=60 
# setting configured in vercel.json.
# Additionally, in-memory state (SESSION_STORE) resets on serverless cold starts. 

SESSION_STORE = {}

def get_session_id():
    session_id = request.cookies.get('session_id')
    if not session_id or session_id not in SESSION_STORE:
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = init_state()
    return session_id

def get_state():
    return SESSION_STORE[get_session_id()]

def init_state():
    return {
        "project": {
            "title": "", "domain": "", "keywords": "", "abstract_notes": "",
            "journal": "IEEE", "columns": "Single Column", "page_limit": 8,
            "char_target": 30000,
            "authors": [{"Name": "", "Affiliation": "", "Email": "", "ORCID": "", "Corresponding": False}],
            "equipment": [{"Equipment Name": "", "Model": "", "Manufacturer": ""}],
        },
        "citation_style": "IEEE",
        "export_include_figures": True,
        "export_include_charts": True,
        "fact_lock": True,
        "papers": [],
        "figures": [],
        "charts": [],
        "sections": {},
        "validation_report": None,
        "latex_output": "",
        "processed_pdf_names": []
    }

# ----------------------------- GEMINI -----------------------------

def call_gemini(prompt, max_tokens=700, temperature=0.25):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return None
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"Gemini call failed: {e}")
        return None

def parse_json_safe(text):
    if not text:
        return None
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None

# ----------------------------- PDF EXTRACTION -----------------------------

def extract_pdf_text(file_bytes, max_chars=12000):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) >= max_chars:
            break
    doc.close()
    return text[:max_chars]

EXTRACTION_PROMPT = """Extract structured metadata from the research paper text below.
Return valid JSON only, no markdown, no commentary.
Do not infer. Do not invent. Use only document content. Use empty string if unknown.
JSON schema:
{{"title":"","authors":"","year":"","doi":"","abstract":"","methodology":"","dataset":"","equipment":"","results":"","limitations":"","future_work":""}}

DOCUMENT TEXT:
{text}
"""

def extract_pdf_data(file_bytes, filename):
    raw_text = extract_pdf_text(file_bytes)
    prompt = EXTRACTION_PROMPT.format(text=raw_text)
    response = call_gemini(prompt, max_tokens=600, temperature=0.0)
    data = parse_json_safe(response)
    if data is None:
        return None
    data["source_file"] = filename
    for k in ["title", "authors", "year", "doi", "abstract", "methodology",
              "dataset", "equipment", "results", "limitations", "future_work"]:
        data.setdefault(k, "")
    return data

# ----------------------------- BIBTEX -----------------------------

def generate_bibtex(papers):
    entries = []
    used_keys = set()
    for p in papers:
        first_author = (p.get("authors") or "Unknown").split(",")[0].split(" and ")[0].strip()
        last_name = first_author.split()[-1] if first_author else "Unknown"
        year = re.sub(r"\D", "", str(p.get("year", ""))) or "n_d"
        base_key = re.sub(r"[^A-Za-z0-9]", "", f"{last_name}{year}") or "ref"
        key = base_key
        i = 1
        while key in used_keys:
            i += 1
            key = f"{base_key}{i}"
        used_keys.add(key)
        entries.append(
            "@article{%s,\n  title={%s},\n  author={%s},\n  year={%s},\n  doi={%s},\n  note={%s}\n}\n" % (
                key, p.get("title", ""), p.get("authors", ""), p.get("year", ""),
                p.get("doi", ""), p.get("source_file", "")
            )
        )
    return "\n".join(entries)

# ----------------------------- SECTIONS -----------------------------

SECTION_LIST = ["Abstract", "Introduction", "Literature Review", "Methodology",
                "Results", "Discussion", "Conclusion"]

FACT_LOCK_CLAUSE = (
    "Use only explicitly supplied information. "
    "If information is unavailable write exactly [DATA NOT PROVIDED]. Never infer.\n"
)

def build_context_block(state):
    proj = state["project"]
    authors_txt = "; ".join(a.get("Name", "") for a in proj["authors"] if a.get("Name"))
    equipment_txt = "; ".join(
        f"{e.get('Equipment Name','')} ({e.get('Manufacturer','')} {e.get('Model','')})"
        for e in proj["equipment"] if e.get("Equipment Name")
    )
    papers_txt = "\n".join(
        f"- {p.get('title','')} ({p.get('year','')}) by {p.get('authors','')}: "
        f"method={p.get('methodology','')}; results={p.get('results','')}; "
        f"limitations={p.get('limitations','')}"
        for p in state["papers"]
    )
    charts_txt = "; ".join(c.get("name", "") for c in state["charts"])
    return {
        "title": proj["title"], "domain": proj["domain"], "keywords": proj["keywords"],
        "notes": proj["abstract_notes"], "authors": authors_txt, "equipment": equipment_txt,
        "papers": papers_txt or "[NONE EXTRACTED]", "charts": charts_txt or "[NONE]",
    }

def build_section_prompt(section, ctx, fact_lock):
    base = FACT_LOCK_CLAUSE if fact_lock else "Use only the supplied information; never invent references, results, or datasets.\n"
    common = (
        f"Paper title: {ctx['title']}\nDomain: {ctx['domain']}\nKeywords: {ctx['keywords']}\n"
        f"Notes: {ctx['notes']}\nAuthors: {ctx['authors']}\n"
    )
    if section == "Abstract":
        return base + common + "Write a concise academic abstract (150-200 words) summarizing the work."
    if section == "Introduction":
        return base + common + "Write an Introduction section motivating the research domain and contribution."
    if section == "Literature Review":
        return (base + common +
                f"Related papers:\n{ctx['papers']}\n\n"
                "First output a markdown table with columns Author | Year | Contribution | Limitation "
                "using only the related papers above. Then write '---' then a narrative literature review "
                "(2-3 paragraphs) referencing only these papers.")
    if section == "Methodology":
        return base + common + f"Equipment used:\n{ctx['equipment']}\n\nWrite a Methodology section describing the approach and listed equipment only."
    if section == "Results":
        return base + common + f"Available chart/data names: {ctx['charts']}\n\nWrite a Results section referencing only the listed charts/data. Do not invent numeric outcomes not present."
    if section == "Discussion":
        return base + common + f"Related papers:\n{ctx['papers']}\n\nWrite a Discussion comparing findings only against the related papers listed above."
    if section == "Conclusion":
        return base + common + "Write a Conclusion summarizing the work. Introduce no new claims beyond prior sections."
    return base + common

def generate_section_logic(state, section):
    ctx = build_context_block(state)
    prompt = build_section_prompt(section, ctx, state["fact_lock"])
    text = call_gemini(prompt, max_tokens=550, temperature=0.3)
    if text:
        state["sections"][section] = text
    return text

# ----------------------------- VALIDATION -----------------------------

RISKY_PHRASES = ["clearly", "obviously", "all studies show", "everyone agrees", "always", "never fails"]

def validate_paper(state):
    success, warning, error = [], [], []
    sections = state["sections"]
    papers = state["papers"]

    for s in SECTION_LIST:
        if not sections.get(s, "").strip():
            error.append(f"Empty section: {s}")
        else:
            success.append(f"{s} generated")

    if not papers:
        error.append("No references extracted")
    else:
        success.append(f"{len(papers)} reference(s) extracted")

    missing_doi = [p.get("title", p.get("source_file", "unknown")) for p in papers if not p.get("doi")]
    if missing_doi:
        warning.append(f"Missing DOI for {len(missing_doi)} paper(s)")

    for s in ["Literature Review", "Discussion"]:
        text = sections.get(s, "")
        if text and not re.search(r"\[\d+\]|\(\w+,?\s*\d{4}\)", text):
            warning.append(f"{s} has no detectable citation markers")

    dnp_count = sum(text.count("[DATA NOT PROVIDED]") for text in sections.values())
    if dnp_count:
        warning.append(f"{dnp_count} place(s) marked [DATA NOT PROVIDED]")

    for s, text in sections.items():
        lower = text.lower()
        for phrase in RISKY_PHRASES:
            if phrase in lower:
                warning.append(f"Possible unsupported claim phrase '{phrase}' in {s}")

    report = {"success": success, "warning": warning, "error": error}
    state["validation_report"] = report
    return report

# ----------------------------- LATEX -----------------------------

def escape_latex(text):
    if not text:
        return ""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(k) for k in replacements))
    return pattern.sub(lambda m: replacements[m.group(0)], text)

JOURNAL_DOCCLASS = {
    "IEEE": r"\documentclass[conference]{IEEEtran}",
    "Springer": r"\documentclass{svjour3}",
    "Elsevier": r"\documentclass[review]{elsarticle}",
    "ACM": r"\documentclass[sigconf]{acmart}",
    "Nature": r"\documentclass{article}",
    "Taylor & Francis": r"\documentclass[12pt]{article}",
    "Custom": r"\documentclass[12pt]{article}",
}

LATEX_TEMPLATE = Template(r"""
{{ docclass }}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{cite}
\title{ {{ title }} }
\author{ {{ authors }} }
\begin{document}
\maketitle

\begin{abstract}
{{ abstract }}
\end{abstract}

\section{Introduction}
{{ introduction }}

\section{Literature Review}
{{ literature_review }}

\section{Methodology}
{{ methodology }}

\section{Results}
{{ results }}

\section{Discussion}
{{ discussion }}

\section{Conclusion}
{{ conclusion }}

\bibliographystyle{ {{ bib_style }} }
\bibliography{references}

\end{document}
""")

CITATION_STYLE_MAP = {"IEEE": "ieeetr", "APA": "apalike", "MLA": "plain", "Chicago": "plain"}

def build_latex_logic(state):
    proj = state["project"]
    sections = state["sections"]
    authors_str = " \\and ".join(
        escape_latex(a.get("Name", "")) for a in proj["authors"] if a.get("Name")
    ) or "Author Name"
    docclass = JOURNAL_DOCCLASS.get(proj["journal"], JOURNAL_DOCCLASS["Custom"])
    bib_style = CITATION_STYLE_MAP.get(state["citation_style"], "plain")
    rendered = LATEX_TEMPLATE.render(
        docclass=docclass,
        title=escape_latex(proj["title"]) or "Untitled Paper",
        authors=authors_str,
        abstract=escape_latex(sections.get("Abstract", "[DATA NOT PROVIDED]")),
        introduction=escape_latex(sections.get("Introduction", "[DATA NOT PROVIDED]")),
        literature_review=escape_latex(sections.get("Literature Review", "[DATA NOT PROVIDED]")),
        methodology=escape_latex(sections.get("Methodology", "[DATA NOT PROVIDED]")),
        results=escape_latex(sections.get("Results", "[DATA NOT PROVIDED]")),
        discussion=escape_latex(sections.get("Discussion", "[DATA NOT PROVIDED]")),
        conclusion=escape_latex(sections.get("Conclusion", "[DATA NOT PROVIDED]")),
        bib_style=bib_style,
    )
    state["latex_output"] = rendered
    return rendered

# ----------------------------- EXPORT ZIP -----------------------------

def build_zip_logic(state):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        latex_content = state["latex_output"] if state["latex_output"] else build_latex_logic(state)
        z.writestr("main.tex", latex_content)
        z.writestr("references.bib", generate_bibtex(state["papers"]))
        
        if state.get("export_include_figures", True):
            for fig in state["figures"]:
                if "bytes" in fig:
                    z.writestr(f"figures/{fig['name']}", fig["bytes"])
        
        if state.get("export_include_charts", True):
            for chart in state["charts"]:
                if "png_bytes" in chart:
                    z.writestr(f"figures/{chart['name']}.png", chart["png_bytes"])
                if "csv_string" in chart:
                    z.writestr(f"tables/{chart['name']}_data.csv", chart["csv_string"].encode("utf-8"))
        
        if state["project"]["equipment"]:
            eq_df = pd.DataFrame(state["project"]["equipment"])
            z.writestr("tables/equipment.csv", eq_df.to_csv(index=False))
        
        readme = (
            "Research Paper LaTeX Builder Export\n\n"
            "Upload this entire project to Overleaf as a New Project > Upload Project.\n"
            "main.tex is the entry point. references.bib holds bibliography entries.\n"
            "figures/ contains uploaded images and exported charts.\n"
            "tables/ contains supporting CSV data.\n"
        )
        z.writestr("README.md", readme)
    
    buf.seek(0)
    return buf

# ----------------------------- FLASK ROUTES -----------------------------

@app.route("/")
def index():
    session_id = get_session_id()
    resp = make_response(render_template("index.html"))
    resp.set_cookie('session_id', session_id)
    return resp

@app.route("/api/state", methods=["GET"])
def api_get_state():
    state = get_state()
    # Strip binary data for UI transport
    clean_state = {**state}
    clean_state["figures"] = [{k: v for k, v in f.items() if k != 'bytes'} for f in state["figures"]]
    clean_state["charts"] = [{k: v for k, v in c.items() if k not in ['png_bytes']} for c in state["charts"]]
    return jsonify(clean_state)

@app.route("/api/state", methods=["POST"])
def api_update_state():
    state = get_state()
    data = request.json
    for key, value in data.items():
        if key in ["figures", "charts"]: continue # Handled by specific endpoints
        state[key] = value
    return jsonify({"status": "ok"})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    session_id = get_session_id()
    SESSION_STORE[session_id] = init_state()
    return jsonify({"status": "ok"})

@app.route("/api/extract_pdfs", methods=["POST"])
def api_extract_pdfs():
    state = get_state()
    files = request.files.getlist("files")
    results = []
    
    for f in files:
        if f.filename not in state["processed_pdf_names"]:
            file_bytes = f.read()
            record = extract_pdf_data(file_bytes, f.filename)
            if record:
                state["papers"].append(record)
                state["processed_pdf_names"].append(f.filename)
                results.append(record)
    return jsonify({"processed": len(results), "papers": state["papers"]})

@app.route("/api/bibtex", methods=["GET"])
def api_bibtex():
    state = get_state()
    bibtex_str = generate_bibtex(state["papers"])
    buf = io.BytesIO(bibtex_str.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, download_name="references.bib", as_attachment=True, mimetype="text/plain")

@app.route("/api/upload_figure", methods=["POST"])
def api_upload_figure():
    state = get_state()
    files = request.files.getlist("figures")
    for img in files:
        existing_names = [f["name"] for f in state["figures"]]
        if img.filename not in existing_names:
            file_bytes = img.read()
            b64 = base64.b64encode(file_bytes).decode('utf-8')
            mime = img.mimetype
            state["figures"].append({
                "name": img.filename, 
                "bytes": file_bytes, 
                "b64": f"data:{mime};base64,{b64}",
                "caption": "", 
                "label": "", 
                "placement": "h"
            })
    
    clean_figures = [{k: v for k, v in f.items() if k != 'bytes'} for f in state["figures"]]
    return jsonify({"figures": clean_figures})

@app.route("/api/generate_chart", methods=["POST"])
def api_generate_chart():
    state = get_state()
    csv_file = request.files.get("csv")
    x_col = request.form.get("x_col")
    y_col = request.form.get("y_col")
    chart_type = request.form.get("chart_type")
    chart_name = request.form.get("chart_name")
    
    if not csv_file: return jsonify({"error": "No CSV file"}), 400
    
    csv_bytes = csv_file.read()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    
    fig, ax = plt.subplots()
    if chart_type == "Line": ax.plot(df[x_col], df[y_col])
    elif chart_type == "Bar": ax.bar(df[x_col], df[y_col])
    elif chart_type == "Scatter": ax.scatter(df[x_col], df[y_col])
    elif chart_type == "Histogram": ax.hist(df[x_col])
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(chart_name)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    
    png_bytes = buf.getvalue()
    b64 = base64.b64encode(png_bytes).decode('utf-8')
    b64_src = f"data:image/png;base64,{b64}"
    
    chart_obj = {
        "name": chart_name,
        "type": chart_type,
        "x_col": x_col,
        "y_col": y_col,
        "png_bytes": png_bytes,
        "b64": b64_src,
        "csv_string": df.to_csv(index=False)
    }
    state["charts"].append(chart_obj)
    
    return jsonify({
        "name": chart_name,
        "type": chart_type,
        "b64": b64_src
    })

@app.route("/api/parse_csv_columns", methods=["POST"])
def api_parse_csv_columns():
    csv_file = request.files.get("csv")
    if not csv_file: return jsonify({"error": "No file"}), 400
    df = pd.read_csv(io.BytesIO(csv_file.read()))
    return jsonify({"columns": list(df.columns)})

@app.route("/api/generate_section", methods=["POST"])
def api_generate_section():
    state = get_state()
    section = request.json.get("section")
    text = generate_section_logic(state, section)
    return jsonify({"section": section, "text": text})

@app.route("/api/validate", methods=["POST"])
def api_validate():
    state = get_state()
    report = validate_paper(state)
    return jsonify(report)

@app.route("/api/build_latex", methods=["POST"])
def api_build_latex():
    state = get_state()
    latex = build_latex_logic(state)
    return jsonify({"latex": latex})

@app.route("/api/download_zip", methods=["GET"])
def api_download_zip():
    state = get_state()
    buf = build_zip_logic(state)
    return send_file(buf, download_name="project.zip", as_attachment=True, mimetype="application/zip")

if __name__ == "__main__":
    app.run(debug=True, port=5000)