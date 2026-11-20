"""PDF report of one analysis (ReportLab), styled like the web app.

``build_pdf`` returns the PDF as bytes so Streamlit can offer it through
``st.download_button`` without writing temporary files. Page 1 is a one-page summary (patient input,
top-3 table, contribution chart, key takeaway, disclaimer); the following content adds the explanation
details, counterfactuals, care information and follow-up questions.
"""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.engine import AnalysisResult
from src.explain import plot_local_explanation
from src.redflags import emergency_banner

DISCLAIMER = (
    "RogMitra.AI is an academic project for learning explainable AI. It is not a medical device and does not "
    "provide a diagnosis. Please consult a qualified healthcare professional."
)
ACCENT = colors.HexColor("#D9602A")
INK = colors.HexColor("#2B1D17")
MUTED = colors.HexColor("#6B5A50")
PANEL = colors.HexColor("#F6F1EC")
LINE = colors.HexColor("#E3D6CB")
AMBER_BG = colors.HexColor("#FFF3D6")
AMBER_LINE = colors.HexColor("#E6B84A")


def _safe(text: object) -> str:
    """Escape XML and replace characters the built-in PDF fonts cannot draw."""
    s = str(text).replace("→", "->").replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return escape(s.encode("latin-1", "replace").decode("latin-1"))


def _chart_png(result: AnalysisResult) -> bytes | None:
    """Render the contribution chart for the top disease as PNG bytes (white background for print)."""
    if not result.top or not (result.supports or result.against):
        return None
    import matplotlib.pyplot as plt

    fig = plot_local_explanation(result.supports, result.against, result.top[0].disease)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _sized_image(png: bytes, width: float) -> Image:
    """Image scaled to ``width`` keeping its aspect ratio."""
    iw, ih = ImageReader(io.BytesIO(png)).getSize()
    return Image(io.BytesIO(png), width=width, height=width * ih / iw)


def _takeaway(result: AnalysisResult) -> str:
    """One-sentence summary built from the numbers (no free text generation)."""
    top = result.top[0]
    if result.is_uncertain:
        return (f"The model is unsure. Its closest match is {top.disease} ({top.prob:.0%}), but the confidence is too low "
                "for a reliable answer. Please consult a doctor.")
    level = "High" if top.prob >= 0.7 else "Moderate" if top.prob >= 0.45 else "Low"
    names = [c.symptom for c in result.supports[:3]]
    why = (" The prediction is mainly supported by " + ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1] + "."
           if names else "")
    return f"{level} likelihood of {top.disease} ({top.prob:.0%}).{why}"


def _header_footer(canvas, doc) -> None:  # noqa: ANN001 - reportlab callback signature
    """Thin accent bar at the top and a footer on every page."""
    w, h = A4
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.rect(0, h - 0.45 * cm, w, 0.45 * cm, stroke=0, fill=1)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(2 * cm, 1.0 * cm, "RogMitra.AI - academic project, not a medical device, not a diagnosis.")
    canvas.drawRightString(w - 2 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(result: AnalysisResult, patient: dict, remedy: dict | None = None) -> bytes:
    """Build the report. ``patient`` may contain ``age`` and ``sex`` (display only, never used by the model)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.8 * cm, title="RogMitra.AI prediction report", author="RogMitra.AI")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13.5, textColor=INK)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10.5, textColor=MUTED)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4)
    cap = ParagraphStyle("cap", parent=body, fontSize=8.5, textColor=MUTED, leading=11)
    brand = ParagraphStyle("brand", parent=body, fontName="Times-Bold", fontSize=24, leading=26, textColor=INK)
    sub = ParagraphStyle("sub", parent=body, fontSize=13, leading=16, textColor=MUTED)
    right = ParagraphStyle("right", parent=small, alignment=2, fontSize=8.5, leading=12)
    alert = ParagraphStyle("alert", parent=body, textColor=colors.white, fontSize=10, leading=14)
    W = A4[0] - 4 * cm

    def panel(rows: list, bg=PANEL, border=LINE, pad: int = 8) -> Table:
        """Shaded box. ``rows`` is a list of one-item rows; all items are stacked inside a single padded cell."""
        cell: list = []
        for r in rows:
            cell += [*r, Spacer(1, 3)]
        t = Table([[cell[:-1]]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.8, border),
                               ("LEFTPADDING", (0, 0), (-1, -1), pad + 2), ("RIGHTPADDING", (0, 0), (-1, -1), pad + 2),
                               ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad)]))
        return t

    # ---------------- header
    head = Table([[[Paragraph('RogMitra<font color="#D9602A">.AI</font>', brand), Paragraph("Prediction Report", sub)],
                   Paragraph(f"Generated: {datetime.now():%d %b %Y, %H:%M}<br/>For academic use only", right)]],
                 colWidths=[W * 0.62, W * 0.38])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 1.2, ACCENT),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story: list = [head, Spacer(1, 10)]

    if result.red_flags:
        lines = _safe(emergency_banner(result.red_flags)).replace("\n", "<br/>")
        story.append(panel([[Paragraph(lines, alert)]], bg=colors.HexColor("#B71C1C"), border=colors.HexColor("#B71C1C")))
        story.append(Spacer(1, 8))

    # ---------------- patient input
    facts = []
    if patient.get("age"):
        facts.append(f"Age: {patient['age']}")
    if patient.get("sex") and patient["sex"] != "Prefer not to say":
        facts.append(f"Sex: {patient['sex']}")
    rows = [[Paragraph("<b>Patient Input</b>", body)],
            [Paragraph("<b>Symptoms:</b> " + _safe(", ".join(result.selected) or "none"), body)]]
    if facts:
        rows.append([Paragraph(_safe("   ".join(facts)) + "  (context only; the model does not use age or sex)", cap)])
    story += [panel(rows), Spacer(1, 10)]

    if result.top:
        # ---------------- predictions table + chart side by side
        if result.is_uncertain:
            story.append(panel([[Paragraph(_safe(f"{result.uncertainty_message} " + " ".join(result.uncertainty_reasons)), body)]],
                               bg=AMBER_BG, border=AMBER_LINE))
            story.append(Spacer(1, 8))
        rows = [["#", "Disease", "Prob.", "Range"]] + [
            [str(i), t.disease, f"{t.prob:.0%}", f"{t.low:.0%}-{t.high:.0%}"] for i, t in enumerate(result.top, 1)]
        tbl = Table([[Paragraph(f"<b>{_safe(c)}</b>" if r == 0 else _safe(c), body) for c in row] for r, row in enumerate(rows)],
                    colWidths=[0.7 * cm, 2.8 * cm, 1.5 * cm, 2.5 * cm])
        tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PANEL), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 4)]))
        left_cell = [Paragraph("<b>Predictions (Top-3)</b>", body), Spacer(1, 4), tbl,
                     Spacer(1, 3), Paragraph("Probabilities are temperature-calibrated; the range is the 5th-95th percentile of "
                                             "30 Monte-Carlo-Dropout passes.", small)]
        png = _chart_png(result)
        right_cell = ([Paragraph("<b>Symptom Contribution</b>", body), Spacer(1, 4), _sized_image(png, 7.6 * cm),
                       Spacer(1, 2), Paragraph("Integrated Gradients; green supports, red argues against.", small)]
                      if png else [Paragraph("No symptom contribution available.", small)])
        two = Table([[left_cell, right_cell]], colWidths=[W * 0.50, W * 0.50])
        two.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.8, LINE),
                                 ("LINEAFTER", (0, 0), (0, 0), 0.8, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 8),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story += [two, Spacer(1, 10)]

        # ---------------- key takeaway + disclaimer
        story.append(panel([[Paragraph("<b>Key Takeaway</b>", body)], [Paragraph(_safe(_takeaway(result)), body)]]))
        story.append(Spacer(1, 8))
        story.append(panel([[Paragraph("<b>Academic project, not a medical diagnosis.</b> " + _safe(
            "Please consult a doctor for proper evaluation and treatment."), body)]], bg=AMBER_BG, border=AMBER_LINE))

        # ---------------- details
        story.append(Paragraph("Why this prediction?", h2))
        story.append(Paragraph(_safe(result.explanation_text), body))
        story.append(Paragraph(_safe(result.explanation_footnote), small))

        if result.counterfactuals:
            cf_rows = [["If you did not have...", "New top prediction", f"{result.top[0].disease} before -> after"]]
            for c in result.counterfactuals[:6]:
                after = "-" if c["target_after"] is None else f"{c['target_after']:.0%}"
                cf_rows.append([c["symptom"], c["new_top"] if c["changes_prediction"] else "unchanged", f"{c['target_before']:.0%} -> {after}"])
            cf = Table([[Paragraph(f"<b>{_safe(x)}</b>" if r == 0 else _safe(x), body) for x in row] for r, row in enumerate(cf_rows)],
                       colWidths=[W * 0.36, W * 0.30, W * 0.34])
            cf.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PANEL), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                                    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story += [KeepTogether([Paragraph("What if a symptom were absent?", h2), cf])]

        useful = [f for f in result.followups if f["expected_entropy_reduction"] > 0.01]
        if useful:
            story.append(Paragraph("Helpful follow-up questions", h2))
            for f in useful[:3]:
                story.append(Paragraph(_safe(f"Do you also have {f['symptom']}? If yes, the model would lean towards {f['top_if_yes']}."),
                                       body, bulletText="•"))

        if result.description:
            story += [Paragraph(f"About {_safe(result.top[0].disease)}", h2), Paragraph(_safe(result.description), body)]
        if result.precautions:
            story.append(Paragraph("Precautions", h2))
            story += [Paragraph(_safe(p), body, bulletText="•") for p in result.precautions]
        if remedy:
            story.append(Paragraph("Home care and when to see a doctor", h2))
            story += [Paragraph(_safe(r), body, bulletText="•") for r in remedy["remedies"]]
            story.append(Spacer(1, 4))
            story.append(panel([[Paragraph("<b>See a doctor if:</b> " + _safe(remedy["when_to_see_doctor"]), body)]],
                               bg=colors.HexColor("#FDEDE4"), border=ACCENT))
            story.append(Paragraph(_safe(f"Source: {remedy['source']} (summary prepared for this project; "
                                         f"{'reviewed against the source' if remedy.get('verified') else 'review pending'})."), small))
    story += [Spacer(1, 12), Paragraph(_safe(DISCLAIMER), small)]
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return buf.getvalue()
