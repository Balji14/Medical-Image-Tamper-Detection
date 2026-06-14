"""Generate a downloadable PDF clinical report using reportlab."""
import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


def _color_verdict(verdict: str):
    return colors.HexColor("#00c853") if "AUTHENTIC" in verdict else colors.HexColor("#d50000")


def generate_pdf(
    patient_name: str,
    patient_id: str,
    scan_type: str,
    filename: str,
    verdict: str,
    similarity: float,
    roi_ratio: float,
    metrics: dict,
    username: str,
) -> bytes | None:
    """Return PDF bytes or None if reportlab is not installed."""
    if not REPORTLAB_OK:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=18, textColor=colors.HexColor("#002147"),
                                  spaceAfter=6)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  fontSize=10, textColor=colors.HexColor("#555555"),
                                  spaceAfter=12)
    head_style  = ParagraphStyle("head", parent=styles["Heading2"],
                                  fontSize=12, textColor=colors.HexColor("#002147"),
                                  spaceBefore=14, spaceAfter=4)
    body_style  = styles["Normal"]

    story = []

    # Header
    story.append(Paragraph("Medical Image Tamper Detection Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; Analyst: {username}",
        sub_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#900000")))
    story.append(Spacer(1, 12))

    # Patient info table
    story.append(Paragraph("Patient Information", head_style))
    patient_data = [
        ["Patient Name", patient_name or "—",   "Patient ID", patient_id or "—"],
        ["Scan Type",    scan_type or "Unknown", "Filename",   filename],
    ]
    pt = Table(patient_data, colWidths=[4 * cm, 6 * cm, 4 * cm, 4 * cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8edf2")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#e8edf2")),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(pt)
    story.append(Spacer(1, 12))

    # Verdict
    story.append(Paragraph("Tamper Detection Verdict", head_style))
    v_color = _color_verdict(verdict)
    verdict_data = [
        [Paragraph(f'<font color="{v_color.hexval()}" size="14"><b>{verdict}</b></font>', body_style),
         f"Hash Similarity: {similarity:.1%}",
         f"ROI Coverage: {roi_ratio:.1%}"],
    ]
    vt = Table(verdict_data, colWidths=[7 * cm, 5 * cm, 5 * cm])
    vt.setStyle(TableStyle([
        ("BOX",     (0, 0), (-1, -1), 1.5, v_color),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("FONTNAME",(0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",(0, 0), (-1, -1), 10),
    ]))
    story.append(vt)
    story.append(Spacer(1, 12))

    # Metrics table
    if metrics:
        story.append(Paragraph("Image Quality Metrics", head_style))
        rows = [["Metric", "Value", "Interpretation"]]
        interp = {
            "PSNR (dB)": "≥ 35 dB → excellent quality",
            "SSIM":      "→ 1.0 = identical",
            "MSE":       "← 0 = no distortion",
            "NCC":       "→ 1.0 = perfect correlation",
            "BER":       "← 0 = no bit errors",
        }
        for k, v in metrics.items():
            rows.append([k, str(v), interp.get(k, "")])
        mt = Table(rows, colWidths=[5 * cm, 4 * cm, 9 * cm])
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#002147")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f5f8fb"), colors.white]),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(mt)
        story.append(Spacer(1, 12))

    # Methodology
    story.append(Paragraph("Methodology", head_style))
    story.append(Paragraph(
        "This analysis follows the Hybrid Residual U-Net++ (HResUNet++) pipeline described in "
        "Sarika &amp; Shankar (2025). The ROI is segmented by the encoder-decoder network; "
        "a Locality-Sensitive Hash (LSH) fingerprint is embedded in the RONI via DWT-LSB "
        "steganography. On verification, the extracted hash is compared bit-by-bit against "
        "the re-generated LSH to produce the similarity score.",
        body_style,
    ))
    story.append(Spacer(1, 8))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))
    disclaimer = ParagraphStyle("disc", parent=styles["Normal"],
                                 fontSize=8, textColor=colors.HexColor("#888888"))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-assisted decision support tool and does "
        "not constitute a clinical diagnosis. All results must be reviewed by a qualified "
        "medical professional before clinical use.",
        disclaimer,
    ))

    doc.build(story)
    return buf.getvalue()
