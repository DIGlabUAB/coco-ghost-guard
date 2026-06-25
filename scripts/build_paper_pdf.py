"""Build the COCO-Ghost project paper PDF.

The project page links to the generated file at docs/assets/coco-ghost-paper.pdf.
This script keeps that artifact reproducible from the checked-in results.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "coco-ghost-paper.pdf"
ASSETS = ROOT / "docs" / "assets"


AUTHORS = (
    "Anusha Kanagala, M.S.; Dhiral Panjwani, M.S.; Stephanie M. Aguilera, B.S.; "
    "Abhiraj Pudhota, M.S.; Kelly Powell, M.B.A.; Danish Murad, M.S.; "
    "Rubin Pillay, M.D.; Leon Jololian, Ph.D.; and Sandeep Bodduluri, M.S., Ph.D."
)


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "authors": ParagraphStyle(
            "Authors",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.5,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#303633"),
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=13.5,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13.5,
            spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Times-Italic",
            fontSize=8.6,
            leading=11,
            textColor=colors.HexColor("#4f5753"),
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.8,
            leading=11,
            spaceAfter=4,
        ),
        "mono": ParagraphStyle(
            "Mono",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.2,
            leading=8.5,
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def image_flowable(filename: str, width: float, caption: str, st: dict, max_height: float = 6.2 * inch) -> KeepTogether:
    path = ASSETS / filename
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    draw_width = width
    draw_height = width * ratio
    if draw_height > max_height:
        draw_height = max_height
        draw_width = max_height / ratio
    img.drawWidth = draw_width
    img.drawHeight = draw_height
    return KeepTogether([img, Spacer(1, 4), Paragraph(caption, st["caption"])])


def results_table(st: dict) -> Table:
    data = [
        ["Model", "Original YES", "Crop YES", "Masked YES", "Ghost rate", "Guard accept"],
        ["qwen2.5vl:7b", "96.0%", "94.5%", "49.5%", "51.6%", "44.5%"],
        ["qwen2.5vl:3b", "88.5%", "80.0%", "43.0%", "47.5%", "30.0%"],
        ["llava:7b", "43.0%", "60.5%", "14.5%", "33.7%", "24.5%"],
    ]
    table = Table(data, colWidths=[1.18 * inch, 0.82 * inch, 0.72 * inch, 0.78 * inch, 0.78 * inch, 0.86 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f2ef")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#173f3a")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c9c5ba")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def bullet_list(items: list[str], st: dict) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, st["body"]), leftIndent=10) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
    )


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    st = styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="COCO-Ghost: Auditing Whether VLM Object Claims Survive Object Removal",
        author=AUTHORS,
    )
    story = []
    story.append(Paragraph("COCO-Ghost: Auditing Whether VLM Object Claims Survive Object Removal", st["title"]))
    story.append(Paragraph(AUTHORS, st["authors"]))

    story.append(Paragraph("Abstract", st["h1"]))
    story.append(
        p(
            "Vision-language models can report objects that are contextually plausible but visually absent. "
            "COCO-Ghost evaluates this behavior with object-level counterfactuals: an object must be recognized "
            "in the original image and in its crop, then the same object is occluded and queried again. If the "
            "model continues to answer YES after clean target removal, the claim is counted as a ghost. On 200 "
            "audited COCO validation instances, three local VLMs retain removed-object claims in 33.7-51.6% of "
            "originally recognized cases. GHOST-Guard is a training-free, black-box abstention rule that accepts "
            "object claims only when they are supported before removal and disappear after removal.",
            st["body"],
        )
    )

    story.append(Paragraph("Core idea", st["h1"]))
    story.append(
        p(
            "The diagnostic asks a deliberately narrow question: if a model says an object is visible, does that "
            "claim depend on local visual evidence or only on scene context? This is useful because context priors "
            "can be semantically reasonable while still being visually unsupported.",
            st["body"],
        )
    )
    story.append(
        bullet_list(
            [
                "Original query: ask whether the target object is present in the full COCO image.",
                "Crop gate: ask the same question on a crop around the target object.",
                "Masked query: replace the target region with padded bounding-box solid local-mean fill and ask again.",
                "Ghost criterion: original YES and masked YES after target removal.",
                "GHOST-Guard accept criterion: original YES, crop YES, and masked NO.",
            ],
            st,
        )
    )

    story.append(Paragraph("Scaled clean-removal results", st["h1"]))
    story.append(
        p(
            "The headline run uses 200 COCO validation instances across 20 object categories and three local "
            "Ollama VLMs. All 200 counterfactuals are labeled clean-removal. The solid bbox intervention is less "
            "naturalistic than blur, but it avoids object-shaped blur that could leak the target silhouette.",
            st["body"],
        )
    )
    story.append(results_table(st))
    story.append(Spacer(1, 8))
    story.append(
        p(
            "Ghost rate is computed among original-YES cases. Guard accept is the fraction of all evaluated cases "
            "accepted after the counterfactual test. Detected ghost claims accepted by GHOST-Guard were 0.0% in "
            "all three model runs.",
            st["small"],
        )
    )

    story.append(Paragraph("Qualitative evidence and artifact audit", st["h1"]))
    story.append(
        image_flowable(
            "success_failure_examples.png",
            6.0 * inch,
            "Figure 1. Representative success and failure cases. A failure means the masked view still receives YES.",
            st,
        )
    )
    story.append(PageBreak())
    story.append(
        image_flowable(
            "model_metric_comparison.png",
            5.6 * inch,
            "Figure 2. Aggregate metrics across three local VLMs on the same 200 clean counterfactuals.",
            st,
        )
    )
    story.append(
        image_flowable(
            "category_model_heatmap.png",
            5.6 * inch,
            "Figure 3. Category-by-model ghost rates reveal object classes with strong context persistence.",
            st,
        )
    )
    story.append(
        image_flowable(
            "removal_method_focused_contact_sheet.png",
            5.6 * inch,
            "Figure 4. Removal-method audit. Bbox-solid occlusion reduces object-shaped residual evidence.",
            st,
        )
    )

    story.append(Paragraph("Interpretation and limitations", st["h1"]))
    story.append(
        p(
            "GHOST-Guard is not a mechanistic explanation and does not repair the underlying model. It is a black-box "
            "reliability wrapper for object-presence claims. In deployed settings, abstentions can be routed to a "
            "detector, a stronger model, another view, or human review.",
            st["body"],
        )
    )
    story.append(
        p(
            "COCO-Ghost is a counterfactual stress test for one interpretable claim type: object presence. It does "
            "not prove that every masked-image YES is a human-level hallucination, and it does not replace broader "
            "VLM robustness evaluation.",
            st["body"],
        )
    )

    story.append(Paragraph("Citation", st["h1"]))
    story.append(
        Paragraph(
            "@misc{cocoghost2026,<br/>"
            "&nbsp;&nbsp;title = {COCO-Ghost: Auditing Whether VLM Object Claims Survive Object Removal},<br/>"
            "&nbsp;&nbsp;author = {Kanagala, Anusha and Panjwani, Dhiral and Aguilera, Stephanie M. and "
            "Pudhota, Abhiraj and Powell, Kelly and Murad, Danish and Pillay, Rubin and Jololian, Leon and "
            "Bodduluri, Sandeep},<br/>"
            "&nbsp;&nbsp;year = {2026},<br/>"
            "&nbsp;&nbsp;url = {https://github.com/DIGlabUAB/coco-ghost-guard}}",
            st["mono"],
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#606864"))
    canvas.drawString(doc.leftMargin, 0.38 * inch, "COCO-Ghost + GHOST-Guard")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    build()
