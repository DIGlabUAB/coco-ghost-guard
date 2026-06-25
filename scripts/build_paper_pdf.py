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
    "Dhiral Panjwani, M.S.; Anusha Kanagala, M.S.; Stephanie M. Aguilera, B.S.; "
    "Abhiraj Pudhota, M.S.; Leon Jololian, Ph.D.; and Sandeep Bodduluri, M.S., Ph.D."
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
        ["Model", "Orig YES", "Crop YES", "Masked YES", "Ghost", "Guard accept", "Guard abstain", "Accepted ghosts"],
        ["qwen2.5vl:7b", "96.0%", "94.5%", "49.5%", "51.6%", "44.5%", "51.5%", "0.0%"],
        ["qwen2.5vl:3b", "88.5%", "80.0%", "43.0%", "47.5%", "30.0%", "58.5%", "0.0%"],
        ["llava:7b", "43.0%", "60.5%", "14.5%", "33.7%", "24.5%", "18.5%", "0.0%"],
    ]
    table = Table(data, colWidths=[1.05 * inch, 0.62 * inch, 0.62 * inch, 0.68 * inch, 0.58 * inch, 0.76 * inch, 0.78 * inch, 0.82 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("LEADING", (0, 0), (-1, -1), 8.8),
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
    story = [
        Paragraph("COCO-Ghost: Auditing Whether VLM Object Claims Survive Object Removal", st["title"]),
        Paragraph(AUTHORS, st["authors"]),
        Paragraph("Abstract", st["h1"]),
        p(
            "Vision-language models can report objects that are contextually plausible but visually absent. "
            "COCO-Ghost evaluates this behavior with object-level counterfactuals: an object must be recognized "
            "in the original image and in its crop, then the same object is occluded and queried again. If the "
            "model continues to answer YES after clean target removal, the claim is counted as a ghost. On 200 "
            "audited COCO validation instances, three local VLMs retain removed-object claims in 33.7-51.6% of "
            "originally recognized cases. GHOST-Guard is a training-free, black-box abstention rule that accepts "
            "object claims only when they are supported before removal and disappear after removal.",
            st["body"],
        ),
    ]

    sections = [
        (
            "1. Introduction",
            [
                "Vision-language models (VLMs) are increasingly used as general visual interfaces: they answer questions, describe scenes, and make object-presence claims from images. A central reliability problem is that these claims can be supported by a mixture of visual evidence and learned scene priors. A model may say that a cup is visible because it detects a cup, but it may also say so because the surrounding sink, table, or countertop makes a cup plausible.",
                "COCO-Ghost studies this failure mode with a narrow counterfactual question: does an object claim survive removal of the object evidence? For each target object, we query the original image, a crop around the target object, and a masked counterfactual image in which the target region is occluded. If the model answers YES after the target has been removed, the object claim is no longer locally supported by the image.",
                "This project makes three contributions: a lightweight object-removal benchmark using audited COCO counterfactuals, a 200-sample clean-removal experiment across three local VLMs, and GHOST-Guard, a black-box abstention rule that rejects object claims that survive target removal.",
            ],
        ),
        (
            "2. Related Work",
            [
                "Object hallucination has a long history in image captioning. Rohrbach et al. introduced the CHAIR metric to measure hallucinated object mentions in generated captions and showed that strong captioning metrics do not necessarily imply faithful object grounding [1]. Large VLMs renewed this concern because instruction-following models can produce fluent visual answers while still mentioning objects inconsistent with the image.",
                "Several benchmarks directly probe object hallucination in VLMs. POPE formulates polling-based object-presence questions to stabilize hallucination evaluation [2]. AMBER extends hallucination evaluation across existence, attribute, and relation dimensions without requiring an LLM judge [3]. COCO-Ghost is complementary: it starts from a known object instance and asks whether a model's own object-presence claim survives a controlled removal of the target evidence.",
                "Mitigation work includes decoding-time approaches such as Visual Contrastive Decoding (VCD), which contrasts predictions under original and distorted images [4], and OPERA, which penalizes attention patterns associated with over-trusting summary tokens [5]. Post-hoc systems such as Woodpecker decompose generated claims and validate them through additional visual questions [6]. GHOST-Guard is closest in spirit to black-box post-hoc validation, but it focuses on one interpretable claim type: object presence under explicit target removal.",
                "COCO-Ghost uses COCO because it provides natural images with object-level instance annotations [7]. The local model suite includes Qwen2.5-VL models [8] and LLaVA-style VLMs [9]. More broadly, segmentation and localization tools such as SAM [10] motivate counterfactual visual editing as a practical way to stress-test whether visual claims remain grounded.",
            ],
        ),
        (
            "3. Methods",
            [
                "For each selected COCO validation instance, the pipeline constructs three views. The original view is the full image. The crop view isolates the target object region. The masked view preserves the surrounding scene while replacing a padded target-object bounding box with a solid local-mean color. The headline experiment uses bbox-solid occlusion rather than blur because blur can preserve object-shaped residuals.",
                "Each view is queried with the same object-presence prompt: whether the target object is visible in the image. The model is constrained to answer YES, NO, or UNSURE, with a short evidence phrase. A ghost is counted when the model answers YES on the original image and also YES on the masked image. GHOST-Guard accepts only the pattern original=YES, crop=YES, masked=NO. If the masked answer remains YES, the guard abstains.",
            ],
        ),
    ]
    for heading, paragraphs in sections:
        story.append(Paragraph(heading, st["h1"]))
        for text in paragraphs:
            story.append(p(text, st["body"]))

    story.append(
        bullet_list(
            [
                "Dataset: 200 COCO validation instances spanning 20 object categories.",
                "Models: qwen2.5vl:7b, qwen2.5vl:3b, and llava:7b through local Ollama inference.",
                "Artifact labels: all 200 headline counterfactuals are labeled clean-removal.",
                "Primary metric: clean Ghost Object Rate among original-YES cases.",
                "Safety metric: detected ghost claims accepted by GHOST-Guard.",
            ],
            st,
        )
    )

    story.append(Paragraph("4. Results", st["h1"]))
    story.append(
        p(
            "Table 1 summarizes the scaled clean-removal run. Qwen2.5-VL 7B recognizes the target in 96.0% of original images and 94.5% of crops, but still answers YES on 49.5% of masked images. Among original-YES cases, this corresponds to a 51.6% clean ghost rate. Qwen2.5-VL 3B shows a similar pattern at 47.5%. LLaVA 7B has lower original recognition, so its 33.7% ghost rate should be interpreted with that caveat.",
            st["body"],
        )
    )
    story.append(results_table(st))
    story.append(Spacer(1, 6))
    story.append(p("Table 1. Scaled clean-removal results. Ghost rate is computed among original-YES cases.", st["caption"]))
    story.append(
        image_flowable(
            "zoomed_evidence_examples.png",
            6.0 * inch,
            "Figure 1. Zoomed evidence examples. The success case flips to NO after bbox-solid removal; the failure case remains YES and is therefore abstained.",
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
    story.append(PageBreak())
    story.append(
        image_flowable(
            "success_failure_examples.png",
            6.0 * inch,
            "Figure 4. Full qualitative contact sheet with success and failure examples from the clean occlusion run.",
            st,
        )
    )
    story.append(
        image_flowable(
            "zoomed_removal_audit.png",
            5.8 * inch,
            "Figure 5. Zoomed removal-method audit. Blur can preserve object-shaped residual evidence; bbox-solid occlusion is less naturalistic but easier to audit.",
            st,
        )
    )

    story.append(Paragraph("5. Discussion", st["h1"]))
    story.append(
        p(
            "The central empirical result is not simply that VLMs hallucinate objects. It is that an object-presence claim can remain stable after the corresponding object evidence has been removed. This makes the failure more specific than generic caption hallucination: the same object claim is tested before and after a targeted counterfactual intervention.",
            st["body"],
        )
    )
    story.append(
        p(
            "GHOST-Guard trades coverage for reliability. It accepted fewer claims than an original-only or crop-only gate, but it accepted 0.0% of detected ghost claims in all three model runs. This is useful in applications where an unsupported object claim should trigger abstention, a detector, a stronger model, another camera view, or human review. It should not be interpreted as repairing the VLM or as explaining model internals.",
            st["body"],
        )
    )

    story.append(Paragraph("6. Limitations", st["h1"]))
    story.append(
        p(
            "COCO-Ghost is intentionally narrow. It evaluates object-presence claims, not attributes, relations, counting, spatial reasoning, or open-ended caption faithfulness. The bbox-solid intervention is conservative with respect to silhouette leakage but can introduce an unnatural patch. The 200-sample run is large enough to reveal a clear failure mode, but larger datasets, additional VLM families, human artifact audits, and self-localized masks would strengthen future versions.",
            st["body"],
        )
    )

    story.append(Paragraph("7. Conclusion", st["h1"]))
    story.append(
        p(
            "COCO-Ghost provides a simple counterfactual test for whether VLM object claims survive object removal. On 200 audited clean-removal COCO examples, local VLMs retained removed-object claims at substantial rates. GHOST-Guard offers a practical black-box abstention rule: trust object claims only when they are locally supported and disappear after target removal.",
            st["body"],
        )
    )

    story.append(Paragraph("References", st["h1"]))
    refs = [
        "A. Rohrbach, L. A. Hendricks, K. Burns, T. Darrell, and K. Saenko. Object Hallucination in Image Captioning. EMNLP, 2018.",
        "Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen. Evaluating Object Hallucination in Large Vision-Language Models. arXiv:2305.10355, 2023.",
        "J. Wang et al. AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation. arXiv:2311.07397, 2023.",
        "S. Leng et al. Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding. arXiv:2311.16922, 2023.",
        "Q. Huang et al. OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation. arXiv:2311.17911, 2023.",
        "S. Yin et al. Woodpecker: Hallucination Correction for Multimodal Large Language Models. arXiv:2310.16045, 2023.",
        "T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. ECCV, 2014.",
        "S. Bai et al. Qwen2.5-VL Technical Report. arXiv:2502.13923, 2025.",
        "H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual Instruction Tuning. NeurIPS, 2023.",
        "A. Kirillov et al. Segment Anything. ICCV, 2023.",
        "J. Li, D. Li, S. Savarese, and S. Hoi. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML, 2023.",
        "D. Dai et al. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. NeurIPS, 2023.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", st["small"]))

    story.append(Paragraph("Citation", st["h1"]))
    story.append(
        Paragraph(
            "@misc{cocoghost2026,<br/>"
            "&nbsp;&nbsp;title = {COCO-Ghost: Auditing Whether VLM Object Claims Survive Object Removal},<br/>"
            "&nbsp;&nbsp;author = {Panjwani, Dhiral and Kanagala, Anusha and Aguilera, Stephanie M. and "
            "Pudhota, Abhiraj and Jololian, Leon and Bodduluri, Sandeep},<br/>"
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
