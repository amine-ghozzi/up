import pytest
from pathlib import Path
from PIL import Image
import io

from pipeline import FinAlzePipeline, PipelineConfig


def make_dummy_image(path: Path):
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    img.save(path)


def test_score_aggregation_with_weights(tmp_path, monkeypatch):
    # Prepare pipeline with custom weights
    cfg = PipelineConfig()
    cfg.score_weights = {"s1": 0.1, "s2": 0.2, "s3": 0.3, "s4": 0.4}
    p = FinAlzePipeline(config=cfg)

    # Monkeypatch classification to pass
    monkeypatch.setattr(p, "_classify_document_strict", lambda path: {"category": "Bilan", "confidence": 0.9})

    # Create a dummy image file
    img_path = tmp_path / "img.jpg"
    make_dummy_image(img_path)

    # Monkeypatch DoclingExtractor.extract_from_image to return a fake result
    from extraction.docling_extractor import TableExtractionResult

    fake_res = TableExtractionResult(
        tables=[[]],
        text="dummy",
        confidence=0.8,
        ocr_score=0.8,
        layout_score=0.7,
        corrections_made=0,
        ocr_grade="GOOD",
        layout_grade="GOOD",
        parse_grade="FAIR",
        low_grade="FAIR",
        metadata={"bboxes": []},
    )

    import extraction.docling_extractor as de
    monkeypatch.setattr(de.DoclingExtractor, "extract_from_image", lambda self, image: fake_res)

    # Run pipeline on image
    res = p.process_document(img_path)

    # Scores should be present
    scores = res.confidence_details.get("scores")
    assert scores is not None
    # compute expected weighted score5 manually
    s1 = res.qcs_score if res.qcs_score is not None else 0.0
    s2 = 0.9
    s3 = 0.8
    s4 = 0.5  # validation fallback used in pipeline if none
    expected = int(round((0.1 * s1 + 0.2 * s2 + 0.3 * s3 + 0.4 * s4) * 100))
    assert scores["score5"] == expected
