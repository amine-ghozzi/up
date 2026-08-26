import streamlit as st
from pathlib import Path
import tempfile
import os

from pipeline import FinAlzePipeline


st.set_page_config(page_title="FinAlze Level 1 POC", layout="wide")

st.title("FinAlze — Level 1 (Lecture & Extraction) POC")

uploaded = st.file_uploader("Déposez un document (PDF, PNG, JPG, TIFF)", type=["pdf", "png", "jpg", "jpeg", "tiff"])

if uploaded:
    st.info(f"Fichier reçu: {uploaded.name} — taille {uploaded.size} bytes")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    st.progress(0)
    st.write("Lancement du pipeline Level 1 (Docling)...")
    pipeline = FinAlzePipeline()

    try:
        result = pipeline.process_document(tmp_path)
    except Exception as e:
        st.error(f"Erreur d'exécution du pipeline: {e}")
        os.unlink(tmp_path)
        st.stop()

    st.progress(100)

    # If pipeline returned a REJECTED metadata, show error and audit detail
    meta = getattr(result, "metadata", {}) or {}
    if meta.get("pipeline_status") == "REJECTED":
        st.error("Document non reconnu comme état financier — traitement interrompu")
        st.json(meta.get("classification"))
        st.write("Raison:", meta.get("reject_reason"))
    else:
        st.success("Extraction initiale terminée — voir extraits ci-dessous")
        st.write("Tier used:", result.tier_used)

        # Scores 1-5
        scores = result.confidence_details.get("scores") if isinstance(result.confidence_details, dict) else None
        if scores:
            cols = st.columns(5)
            keys = ["score1", "score2", "score3", "score4", "score5"]
            labels = ["Qualité OCR", "Identification", "Extraction", "Mapping/Valid.", "Confiance globale"]
            for c, k, lab in zip(cols, keys, labels):
                value = scores.get(k) if scores else None
                c.metric(label=lab, value=f"{value}/100" if value is not None else "—")
        else:
            st.write("QCS score:", result.qcs_score)

        st.write("Confidence details:")
        st.json(result.confidence_details)

        # Document image + bounding boxes (first page)
        bboxes = meta.get("bboxes") or []
        try:
            from PIL import Image, ImageDraw
            import io
            import fitz

            img = None
            suffix = Path(uploaded.name).suffix.lower()
            if suffix == ".pdf":
                try:
                    pdf = fitz.open(str(tmp_path))
                    page = pdf[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.open(io.BytesIO(pix.tobytes()))
                except Exception:
                    img = None
            else:
                try:
                    img = Image.open(tmp_path)
                except Exception:
                    img = None

            if img and bboxes:
                draw = ImageDraw.Draw(img)
                w, h = img.size
                for box in bboxes:
                    # Support two formats: normalized dict (x0,y0,x1,y1)
                    # or raw {'bbox': (x0,y0,x1,y1), 'page': n}
                    if "bbox" in box:
                        bx = box.get("bbox")
                        try:
                            x0, y0, x1, y1 = bx
                            # If coordinates appear normalized (<=1), scale
                            if max(x0, y0, x1, y1) <= 1.0:
                                x0 = int(x0 * w)
                                x1 = int(x1 * w)
                                y0 = int(y0 * h)
                                y1 = int(y1 * h)
                            else:
                                x0 = int(x0)
                                x1 = int(x1)
                                y0 = int(y0)
                                y1 = int(y1)
                        except Exception:
                            continue
                        label = box.get("label", "")
                    else:
                        x0 = int(box.get("x0", 0) * w)
                        y0 = int(box.get("y0", 0) * h)
                        x1 = int(box.get("x1", 1) * w)
                        y1 = int(box.get("y1", 1) * h)
                        label = box.get("label", "")

                    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                    if label:
                        draw.text((x0 + 4, y0 + 4), label, fill="red")
                st.image(img, caption="Document (première page) — zones détectées")
            elif img:
                st.image(img, caption="Document (première page)")
        except Exception:
            pass

        # Mirror table view: for each extracted table, show document snippet and structured rows
        if result.tables:
            st.write("Tables extraites (miroir document ↔ données)")
            for i, tbl in enumerate(result.tables):
                st.markdown(f"**Table {i+1} — {len(tbl)} ligne(s)**")
                left, right = st.columns([1, 2])
                with left:
                    st.write("Extrait brut (JSON):")
                    st.json(tbl)
                with right:
                    st.write("Vue structurée:")
                    try:
                        import pandas as pd
                        df = pd.DataFrame(tbl)
                        st.dataframe(df)
                    except Exception:
                        st.write(tbl)

    # cleanup
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
