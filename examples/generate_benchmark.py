#!/usr/bin/env python3
"""
generate_benchmark.py — Generates 5 synthetic PDFs and a ground-truth QA dataset
for the Colette visual RAG retrieval evaluation benchmark (Aether Analytics / VIRA).
"""

import os
import io
import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.colors import HexColor

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Brand colours ─────────────────────────────────────────────────────────────
PRIMARY  = HexColor("#2E4057")
ACCENT   = HexColor("#048A81")
LIGHT_BG = HexColor("#F4F6F9")
ALT_ROW  = HexColor("#E8EDF2")
GRAY     = HexColor("#666666")
WHITE    = colors.white
BLACK    = colors.black

METHODS        = ["AlphaNet", "BetaSearch", "GammaHybrid", "VIRA-v1", "VIRA-v2"]
METHOD_COLORS  = ["#E07A5F",  "#3D405B",    "#81B29A",     "#F2CC8F",  "#048A81"]

# ── Styles ────────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    defs = [
        ("VQCoverTitle",  "Times-Bold",   24, PRIMARY, TA_CENTER, 14, 0,  16),
        ("VQCoverSub",    "Times-Roman",  14, ACCENT,  TA_CENTER,  8, 0,  18),
        ("VQSecHead",     "Times-Bold",   14, PRIMARY, TA_LEFT,    8, 10, 18),
        ("VQBody",        "Times-Roman",  10, BLACK,   TA_JUSTIFY, 6, 0,  14),
        ("VQCaption",     "Times-Italic",  9, GRAY,    TA_CENTER,  6, 0,  12),
        ("VQBullet",      "Times-Roman",  10, BLACK,   TA_LEFT,    4, 0,  14),
        ("VQFinding",     "Times-Roman",  11, BLACK,   TA_LEFT,    8, 0,  15),
        ("VQSmall",       "Times-Roman",   9, BLACK,   TA_LEFT,    4, 0,  13),
    ]
    for name, font, size, color, align, after, before, lead in defs:
        kwargs = dict(fontName=font, fontSize=size, textColor=color,
                      alignment=align, spaceAfter=after, spaceBefore=before,
                      leading=lead)
        if name == "VQBullet":
            kwargs.update(leftIndent=18, bulletIndent=8)
        if name == "VQFinding":
            kwargs.update(leftIndent=22)
        base.add(ParagraphStyle(name, **kwargs))
    return base

STYLES = make_styles()

# ── ReportLab helpers ─────────────────────────────────────────────────────────
def color_bar(w=6.5*inch):
    return Table([[""]], colWidths=[w], rowHeights=[10],
                 style=TableStyle([("BACKGROUND", (0,0), (-1,-1), ACCENT)]))


def make_table(data, col_widths, left_align_cols=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND",    (0,0),  (-1,0),  PRIMARY),
        ("TEXTCOLOR",     (0,0),  (-1,0),  WHITE),
        ("FONTNAME",      (0,0),  (-1,0),  "Times-Bold"),
        ("FONTSIZE",      (0,0),  (-1,0),  9),
        ("ALIGN",         (0,0),  (-1,-1), "CENTER"),
        ("VALIGN",        (0,0),  (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),  (-1,-1), [WHITE, ALT_ROW]),
        ("FONTNAME",      (0,1),  (-1,-1), "Times-Roman"),
        ("FONTSIZE",      (0,1),  (-1,-1), 9),
        ("GRID",          (0,0),  (-1,-1), 0.4, GRAY),
        ("TOPPADDING",    (0,0),  (-1,-1), 5),
        ("BOTTOMPADDING", (0,0),  (-1,-1), 5),
        ("LEFTPADDING",   (0,0),  (-1,-1), 4),
        ("RIGHTPADDING",  (0,0),  (-1,-1), 4),
    ]
    if left_align_cols:
        for c in left_align_cols:
            style_cmds.append(("ALIGN", (c,0), (c,-1), "LEFT"))
    t.setStyle(TableStyle(style_cmds))
    return t


# ── Matplotlib helpers ────────────────────────────────────────────────────────
def fig_to_rl(fig, width=6.5*inch, height=None):
    """Render a matplotlib figure to a ReportLab Image with correct proportional height."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    # ReportLab ignores PNG DPI metadata and treats images as 72 dpi when
    # computing natural size. Always provide both dimensions so it scales correctly.
    from PIL import Image as PILImage
    pil = PILImage.open(buf)
    px_w, px_h = pil.size
    buf.seek(0)
    if height is None:
        height = width * (px_h / px_w)
    return Image(buf, width=width, height=height)


def _setup_fig(w=9, h=4.5, bg="#F4F6F9"):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    return fig, ax


# ════════════════════════════════════════════════════════════════════════════════
# PDF 1 — VTQ-Q1-2025-Architecture
# ════════════════════════════════════════════════════════════════════════════════

def _arch_flowchart():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("#F4F6F9")
    ax.set_facecolor("#F4F6F9")
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")

    boxes = [
        (1.4,  2.0, "Document\nIngestion",    "#4A90D9", "PyMuPDF +\nLayoutDetector"),
        (4.2,  2.0, "Embedding\nEngine",      "#27AE60", "gme-Qwen2-\nVL-7B"),
        (7.0,  2.0, "Hierarchical\nIndex",    "#8E44AD", "ChromaDB +\nTantivy + HDF5"),
        (9.8,  2.0, "Query\nProcessor",       "#E67E22", "Qwen3-VL-7B"),
    ]
    for x, y, label, col, tech in boxes:
        rect = mpatches.FancyBboxPatch((x-1.1, y-0.75), 2.2, 1.5,
                                       boxstyle="round,pad=0.12",
                                       facecolor=col, edgecolor="white",
                                       linewidth=2, alpha=0.92, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y+0.27, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=3)
        ax.text(x, y-0.35, tech, ha="center", va="center",
                fontsize=7.5, color="white", style="italic", zorder=3)

    for i in range(len(boxes)-1):
        x1 = boxes[i][0]+1.1; x2 = boxes[i+1][0]-1.1
        ax.annotate("", xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle="-|>", color="#2E4057",
                                    lw=2.0, mutation_scale=16), zorder=1)

    ax.set_title("Figure 1 — VIRA System Architecture: Data Flow",
                 fontsize=11, fontweight="bold", color="#2E4057", pad=8)
    return fig


def _deployment_diagram():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#F4F6F9")
    ax.set_facecolor("#F4F6F9")
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.5); ax.axis("off")

    tiers = [
        (6.0, "Client Tier",
         ["User Applications", "Web Browser", "Mobile App"],    "#4A90D9"),
        (3.5, "API Tier",
         ["VIRA API Gateway", "LLM Service\n(Qwen3-VL-7B)", "Auth & Rate Limiter"], "#048A81"),
        (1.0, "Storage Tier",
         ["ChromaDB\n(Vectors)", "Tantivy\n(BM25 Index)", "KVStore\n(Metadata)"],   "#8E44AD"),
    ]
    for yc, name, comps, col in tiers:
        bg = mpatches.FancyBboxPatch((0.2, yc-0.9), 11.6, 1.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=col, edgecolor=col,
                                     linewidth=1.5, alpha=0.12)
        ax.add_patch(bg)
        ax.text(0.6, yc+0.62, name, fontsize=11, fontweight="bold",
                color=col, va="center")
        xs = [3.0, 6.0, 9.0]
        for x, comp in zip(xs, comps):
            box = mpatches.FancyBboxPatch((x-1.25, yc-0.6), 2.5, 0.9,
                                          boxstyle="round,pad=0.08",
                                          facecolor=col, edgecolor="white",
                                          linewidth=1.5, alpha=0.85)
            ax.add_patch(box)
            ax.text(x, yc-0.15, comp, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    # Bi-directional arrows between tiers
    for y_lo, y_hi in [(1.9, 2.6), (4.4, 5.1)]:
        ax.annotate("", xy=(6, y_hi), xytext=(6, y_lo),
                    arrowprops=dict(arrowstyle="<->", color="#2E4057", lw=2.0))

    ax.set_title("Figure 2 — VIRA Deployment Architecture: 3-Tier Overview",
                 fontsize=11, fontweight="bold", color="#2E4057", pad=8)
    return fig


def make_pdf1():
    pdf_path = os.path.join(OUTPUT_DIR, "VTQ-Q1-2025-Architecture.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    s = STYLES
    story = []

    # ── P1: Cover ──
    story += [
        color_bar(), Spacer(1, 0.25*inch),
        Paragraph("VIRA Technical Architecture", s["VQCoverTitle"]),
        Paragraph("Q1 2025 Report", s["VQCoverSub"]),
        Paragraph("Aether Analytics", s["VQCoverSub"]),
        Spacer(1, 0.15*inch),
        HRFlowable(width="100%", thickness=2, color=ACCENT),
        Spacer(1, 0.2*inch),
        Paragraph("<i>Visual Intelligence Retrieval Architecture — "
                  "Built for Scale, Designed for Precision</i>", s["VQCoverSub"]),
        Spacer(1, 0.35*inch),
        Paragraph(
            "This report introduces VIRA (Visual Intelligence Retrieval Architecture), "
            "Aether Analytics’ next-generation document retrieval system for visually "
            "rich corpora. The architecture employs a hierarchical indexing strategy that "
            "combines dense visual embeddings, sparse BM25 retrieval, and VLM-guided "
            "navigation, achieving a baseline P@4 of 0.71 in Q1 2025 evaluations with "
            "sub-4-second average query latency and an index throughput of 12 pages/minute.",
            s["VQBody"]),
        PageBreak(),
    ]

    # ── P2: Architecture figure ──
    story += [
        Paragraph("System Architecture", s["VQSecHead"]),
        Spacer(1, 0.1*inch),
        fig_to_rl(_arch_flowchart(), width=6.5*inch),
        Paragraph(
            "Figure 1. VIRA system architecture showing the four main processing "
            "components and the document-to-result data flow.",
            s["VQCaption"]),
        Spacer(1, 0.15*inch),
        Paragraph(
            "The VIRA pipeline processes documents through four sequential stages. "
            "Document Ingestion parses raw inputs and detects layout using PyMuPDF and "
            "LayoutDetector. The Embedding Engine encodes each page into a dense vector "
            "using gme-Qwen2-VL-7B with MROPE position encoding. The Hierarchical Index "
            "organises embeddings into a 3-level hierarchy for efficient retrieval. "
            "Finally, the Query Processor ranks candidates and generates answers using "
            "Qwen3-VL-7B.", s["VQBody"]),
        PageBreak(),
    ]

    # ── P3: Component Specifications table ──
    story += [Paragraph("Component Specifications", s["VQSecHead"]), Spacer(1, 0.1*inch)]
    comp_data = [
        ["Component", "Model / Backend", "Max Input", "Key Parameter", "Notes"],
        ["Document Ingestion", "PyMuPDF +\nLayoutDetector",
         "1000 pages/doc", "page image 1024×1024", "supports PDF, DOCX, PNG"],
        ["Embedding Engine",  "gme-Qwen2-VL-7B",
         "2048 tokens",       "cosine similarity",    "MROPE position encoding"],
        ["Hierarchical Index","ChromaDB +\nTantivy + HDF5",
         "32 topics/doc",     "reranking threshold 0.72", "3-level hierarchy"],
        ["Query Processor",   "Qwen3-VL-7B",
         "4096 tokens",       "default top-k = 4",    "supports hybrid + hindex modes"],
    ]
    story += [
        make_table(comp_data,
                   [1.25*inch, 1.5*inch, 1.1*inch, 1.55*inch, 1.1*inch],
                   left_align_cols=[0,4]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "Table 1. VIRA component specifications. All components benchmarked on A100 80 GB GPU hardware.",
            s["VQCaption"]),
        Spacer(1, 0.15*inch),
        Paragraph(
            "The Embedding Engine uses MROPE (Multi-dimensional Rotary Position Encoding) "
            "for accurate spatial understanding of document layouts. The Hierarchical Index "
            "applies a reranking threshold of 0.72 to prune irrelevant candidates before "
            "returning results to the Query Processor.", s["VQBody"]),
        PageBreak(),
    ]

    # ── P4: Deployment diagram ──
    story += [
        Paragraph("Deployment Architecture", s["VQSecHead"]),
        fig_to_rl(_deployment_diagram(), width=6.5*inch, height=3.6*inch),
        Paragraph(
            "Figure 2. VIRA 3-tier deployment architecture: Client, API, and Storage tiers.",
            s["VQCaption"]),
        Spacer(1, 0.15*inch),
        Paragraph(
            "The deployment follows a standard 3-tier architecture. The Client Tier "
            "encompasses user-facing applications including web browsers and mobile apps. "
            "The API Tier hosts the VIRA API Gateway, LLM Service (Qwen3-VL-7B), and "
            "authentication components. The Storage Tier provides persistence via "
            "ChromaDB for vector embeddings, Tantivy for full-text BM25 indexing, and "
            "a KVStore for document metadata.", s["VQBody"]),
        PageBreak(),
    ]

    # ── P5: Q1 Highlights + Metrics ──
    story += [Paragraph("Q1 2025 Highlights", s["VQSecHead"])]
    for item in [
        "Successfully deployed VIRA architecture to internal staging environment with full document pipeline.",
        "Integrated gme-Qwen2-VL-7B with MROPE position encoding for layout-aware visual retrieval.",
        "Achieved a 3-level hierarchical index supporting up to 32 topic clusters per document.",
        "Indexed 8,432 documents across three test corpora at an average throughput of 12 pages/minute.",
        "Established baseline P@4 of 0.71 on the internal evaluation set, targeting 0.85+ by Q4 2025.",
    ]:
        story.append(Paragraph(f"• {item}", s["VQBullet"]))

    story += [
        Spacer(1, 0.25*inch),
        Paragraph("Q1 Key Metrics", s["VQSecHead"]),
    ]
    metrics = [
        ["Metric", "Value"],
        ["Precision@4 (P@4)",         "0.71"],
        ["Recall@4 (R@4)",            "0.73"],
        ["Mean Reciprocal Rank (MRR)","0.68"],
        ["Average Query Latency",     "3.4 s"],
        ["Index Throughput",          "12 pages/min"],
        ["Documents Indexed",         "8,432"],
    ]
    story.append(make_table(metrics, [3.5*inch, 3.0*inch]))

    doc.build(story)
    return pdf_path


# ════════════════════════════════════════════════════════════════════════════════
# PDF 2 — VTQ-Q2-2025-Benchmarks
# ════════════════════════════════════════════════════════════════════════════════

# Benchmark data used in multiple figures / tables
Q2_DATA = {
    #               P@4-S  R@4-S  P@4-M  R@4-M  Lat
    "AlphaNet":    [0.641, 0.658, 0.683, 0.701, 1.2],
    "BetaSearch":  [0.698, 0.731, 0.724, 0.768, 2.1],
    "GammaHybrid": [0.721, 0.812, 0.756, 0.834, 3.8],
    "VIRA-v1":     [0.773, 0.791, 0.801, 0.812, 4.2],
    "VIRA-v2":     [0.819, 0.843, 0.847, 0.871, 3.9],
}


def _q2_bar_chart():
    """Precision@4 grouped bar chart — DocBench-S and DocBench-M."""
    x = np.arange(len(METHODS))
    w = 0.35
    ps = [Q2_DATA[m][0] for m in METHODS]
    pm = [Q2_DATA[m][2] for m in METHODS]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#F4F6F9"); ax.set_facecolor("#F4F6F9")
    bars1 = ax.bar(x - w/2, ps, w, label="DocBench-S",
                   color=[c+"CC" for c in METHOD_COLORS], edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w/2, pm, w, label="DocBench-M",
                   color=METHOD_COLORS, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=10)
    ax.set_ylim(0.55, 0.92)
    ax.set_ylabel("Precision@4", fontsize=10)
    ax.set_title("Figure 3 — Precision@4 by Method and Dataset", fontsize=11,
                 fontweight="bold", color="#2E4057")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    for b in list(bars1) + list(bars2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    return fig


def _q2_latency_and_scatter():
    """Two side-by-side charts: latency bar + P@4 vs Latency scatter."""
    lats = [Q2_DATA[m][4] for m in METHODS]
    pm   = [Q2_DATA[m][2] for m in METHODS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#F4F6F9")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F4F6F9")

    # Left: latency bar
    bars = ax1.bar(METHODS, lats, color=METHOD_COLORS, edgecolor="white", linewidth=0.8)
    ax1.set_ylabel("Avg Latency (s)", fontsize=10)
    ax1.set_title("Figure 4a — Average Latency", fontsize=10, fontweight="bold", color="#2E4057")
    ax1.set_ylim(0, 5.2)
    ax1.yaxis.grid(True, alpha=0.4); ax1.set_axisbelow(True)
    ax1.tick_params(axis="x", labelsize=9, rotation=15)
    for b, v in zip(bars, lats):
        ax1.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.1f}s",
                 ha="center", va="bottom", fontsize=8)

    # Right: P@4 vs Latency scatter
    for m, col in zip(METHODS, METHOD_COLORS):
        ax2.scatter(Q2_DATA[m][4], Q2_DATA[m][2], color=col, s=100, zorder=3,
                    edgecolors="white", linewidths=1.2, label=m)
        ax2.annotate(m, (Q2_DATA[m][4], Q2_DATA[m][2]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8, color=col)
    ax2.set_xlabel("Avg Latency (s)", fontsize=10)
    ax2.set_ylabel("Precision@4 (DocBench-M)", fontsize=10)
    ax2.set_title("Figure 4b — P@4 vs Latency (DocBench-M)", fontsize=10,
                  fontweight="bold", color="#2E4057")
    ax2.yaxis.grid(True, alpha=0.4); ax2.xaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)
    fig.tight_layout()
    return fig


def make_pdf2():
    pdf_path = os.path.join(OUTPUT_DIR, "VTQ-Q2-2025-Benchmarks.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    s = STYLES
    story = []

    # ── P1: Cover + executive summary ──
    story += [
        color_bar(), Spacer(1, 0.25*inch),
        Paragraph("VIRA Technical Quarterly", s["VQCoverTitle"]),
        Paragraph("Q2 2025 Benchmark Results", s["VQCoverSub"]),
        Paragraph("Aether Analytics", s["VQCoverSub"]),
        Spacer(1, 0.15*inch),
        HRFlowable(width="100%", thickness=2, color=ACCENT),
        Spacer(1, 0.2*inch),
        Paragraph("Executive Summary", s["VQSecHead"]),
        Paragraph(
            "This report presents the Q2 2025 benchmark results for VIRA (Visual Intelligence "
            "Retrieval Architecture) and four competing retrieval systems evaluated on two "
            "standardised document retrieval datasets: DocBench-S (short documents) and "
            "DocBench-M (multi-page documents). VIRA-v2 achieves the highest overall "
            "Precision@4 of 0.847 on DocBench-M, surpassing all baselines. GammaHybrid "
            "demonstrates the best recall at 0.834, while AlphaNet is the fastest system "
            "at 1.2 seconds average latency.", s["VQBody"]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "Five systems were evaluated: AlphaNet (dense retrieval baseline), BetaSearch "
            "(BM25 variant), GammaHybrid (hybrid fusion), VIRA-v1 (hierarchical index, "
            "first generation), and VIRA-v2 (hierarchical index, second generation with "
            "VLM-enriched crops). All experiments were conducted on a held-out test split "
            "of 250 queries per dataset. Results demonstrate consistent improvements from "
            "VIRA’s hierarchical indexing approach over both dense and sparse baselines.",
            s["VQBody"]),
        PageBreak(),
    ]

    # ── P2: Benchmark Results table ──
    story += [Paragraph("Benchmark Results", s["VQSecHead"]), Spacer(1, 0.1*inch)]
    bench = [
        ["Method", "P@4\nDocBench-S", "R@4\nDocBench-S",
         "P@4\nDocBench-M", "R@4\nDocBench-M", "Avg\nLatency (s)"],
    ]
    for m in METHODS:
        v = Q2_DATA[m]
        bench.append([m, f"{v[0]:.3f}", f"{v[1]:.3f}", f"{v[2]:.3f}", f"{v[3]:.3f}", f"{v[4]:.1f}"])
    story += [
        make_table(bench, [1.3*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch],
                   left_align_cols=[0]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "Table 2. Benchmark results across two datasets. Best value per column shown in bold. "
            "All scores are averages over 250 test queries per dataset.",
            s["VQCaption"]),
        Spacer(1, 0.2*inch),
        Paragraph(
            "VIRA-v2 leads on both DocBench-S (P@4=0.819) and DocBench-M (P@4=0.847). "
            "GammaHybrid records the best recall on DocBench-M (R@4=0.834), slightly above "
            "VIRA-v2’s 0.871 — wait, VIRA-v2 leads recall on DocBench-M at 0.871. "
            "AlphaNet remains the fastest system but trails in accuracy by 16+ pp.", s["VQBody"]),
        PageBreak(),
    ]

    # ── P3: Grouped bar chart ──
    story += [
        Paragraph("Precision@4 Comparison", s["VQSecHead"]),
        fig_to_rl(_q2_bar_chart(), width=6.5*inch),
        Paragraph(
            "Figure 3. Grouped bar chart of Precision@4 for all five methods on "
            "DocBench-S (lighter) and DocBench-M (solid). VIRA-v2 leads on both datasets.",
            s["VQCaption"]),
        PageBreak(),
    ]

    # ── P4: Latency + scatter ──
    story += [
        Paragraph("Latency and Accuracy-Efficiency Trade-off", s["VQSecHead"]),
        fig_to_rl(_q2_latency_and_scatter(), width=6.5*inch),
        Paragraph(
            "Figure 4. Left: average latency per method. Right: Precision@4 (DocBench-M) "
            "versus latency — VIRA-v2 occupies the high-accuracy, moderate-latency region.",
            s["VQCaption"]),
        PageBreak(),
    ]

    # ── P5: Key findings + improvement table ──
    story += [Paragraph("Key Findings", s["VQSecHead"])]
    for i, txt in enumerate([
        "VIRA-v2 achieves the highest P@4 (0.847 on DocBench-M), a 24.1% improvement over the AlphaNet baseline (0.683).",
        "GammaHybrid leads in recall (R@4=0.834 on DocBench-M), making it the preferred choice when missing relevant results is costly.",
        "AlphaNet is the fastest system (1.2 s) but pays a significant accuracy penalty; VIRA-v2 (3.9 s) is only 3× slower yet 24 pp more precise.",
        "VIRA-v1 to VIRA-v2 upgrade yields +4.6 pp P@4 on DocBench-M with a 7.1% latency reduction, demonstrating the value of VLM-enriched crop embeddings.",
    ], 1):
        story.append(Paragraph(f"{i}. {txt}", s["VQFinding"]))

    story += [Spacer(1, 0.2*inch), Paragraph("Improvement over AlphaNet Baseline", s["VQSecHead"])]

    # Compute deltas vs AlphaNet
    alpha_p = Q2_DATA["AlphaNet"][2]; alpha_r = Q2_DATA["AlphaNet"][3]; alpha_l = Q2_DATA["AlphaNet"][4]
    imp = [["Method", "Δ P@4 (DocBench-M)", "Δ R@4 (DocBench-M)", "Δ Latency (s)"]]
    for m in METHODS[1:]:
        v = Q2_DATA[m]
        imp.append([m,
                    f"+{v[2]-alpha_p:+.3f}",
                    f"+{v[3]-alpha_r:+.3f}",
                    f"{v[4]-alpha_l:+.1f}"])
    story.append(make_table(imp, [1.5*inch, 1.6*inch, 1.6*inch, 1.6*inch], left_align_cols=[0]))

    doc.build(story)
    return pdf_path


# ════════════════════════════════════════════════════════════════════════════════
# PDF 3 — VTQ-Q3-2025-ErrorAnalysis
# ════════════════════════════════════════════════════════════════════════════════

ERROR_DATA = [
    # type           freq  pct    sev       component           mitigation
    ["Layout Confusion", 387, 31.0, "High",     "Document Ingestion", "multi-scale processing"],
    ["Hallucination",    287, 23.0, "Critical", "Query Processor",    "confidence thresholding"],
    ["Context Overflow", 224, 18.0, "Medium",   "Embedding Engine",   "chunk size reduction"],
    ["OCR Failure",      187, 15.0, "High",     "Document Ingestion", "preprocessing enhancement"],
    ["Retrieval Miss",   162, 13.0, "Medium",   "Hierarchical Index", "query expansion"],
]


def _q3_error_charts():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#F4F6F9")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F4F6F9")

    types = [r[0] for r in ERROR_DATA]
    freqs = [r[1] for r in ERROR_DATA]
    colors_err = ["#E07A5F", "#C0392B", "#3D405B", "#F2CC8F", "#81B29A"]

    # Left: horizontal bar
    y = range(len(types))
    bars = ax1.barh(list(y), freqs, color=colors_err, edgecolor="white", linewidth=0.8)
    ax1.set_yticks(list(y)); ax1.set_yticklabels(types, fontsize=9)
    ax1.set_xlabel("Number of Error Cases", fontsize=10)
    ax1.set_title("Figure 5a — Error Frequency by Type", fontsize=10,
                  fontweight="bold", color="#2E4057")
    ax1.xaxis.grid(True, alpha=0.4); ax1.set_axisbelow(True)
    for b, v in zip(bars, freqs):
        ax1.text(v+4, b.get_y()+b.get_height()/2, str(v),
                 va="center", fontsize=9, fontweight="bold")

    # Right: donut — severity distribution
    sev_map = {"Critical": 0, "High": 0, "Medium": 0}
    for r in ERROR_DATA:
        sev_map[r[3]] += r[1]
    total = sum(sev_map.values())
    labels = [f"{k}\n{v/total*100:.1f}%" for k, v in sev_map.items()]
    wedge_colors = ["#C0392B", "#E07A5F", "#3D405B"]
    wedges, _ = ax2.pie(list(sev_map.values()), labels=labels,
                        colors=wedge_colors, startangle=90,
                        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
                        textprops=dict(fontsize=9))
    ax2.set_title("Figure 5b — Severity Distribution", fontsize=10,
                  fontweight="bold", color="#2E4057")
    fig.tight_layout()
    return fig


def make_pdf3():
    pdf_path = os.path.join(OUTPUT_DIR, "VTQ-Q3-2025-ErrorAnalysis.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    s = STYLES
    story = []

    # ── P1: Cover + summary ──
    story += [
        color_bar(), Spacer(1, 0.25*inch),
        Paragraph("VIRA Technical Quarterly", s["VQCoverTitle"]),
        Paragraph("Q3 2025 — Error Analysis Report", s["VQCoverSub"]),
        Paragraph("Aether Analytics", s["VQCoverSub"]),
        Spacer(1, 0.15*inch),
        HRFlowable(width="100%", thickness=2, color=ACCENT),
        Spacer(1, 0.25*inch),
        Paragraph("Summary", s["VQSecHead"]),
        Paragraph(
            "This report presents a systematic error analysis of 1,247 error cases "
            "identified from the Q2 2025 benchmark evaluation. Errors were manually "
            "reviewed and classified into five categories according to their root cause, "
            "severity, and the VIRA component primarily responsible. The dominant failure "
            "mode is Layout Confusion (387 cases, 31.0%), followed by Hallucination "
            "(287 cases, 23.0%). Proposed mitigations are projected to reduce total error "
            "frequency by an estimated 50–60% when fully deployed in Q3–Q4 2025.",
            s["VQBody"]),
        PageBreak(),
    ]

    # ── P2: Error Taxonomy table ──
    story += [Paragraph("Error Taxonomy", s["VQSecHead"]), Spacer(1, 0.1*inch)]
    tax = [["Error Type", "Frequency", "% of Total", "Severity",
            "Affected Component", "Mitigation Strategy"]]
    for r in ERROR_DATA:
        tax.append([r[0], str(r[1]), f"{r[2]:.1f}%", r[3], r[4], r[5]])
    story += [
        make_table(tax,
                   [1.15*inch, 0.7*inch, 0.75*inch, 0.75*inch, 1.2*inch, 1.45*inch],
                   left_align_cols=[0, 4, 5]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "Table 3. Error taxonomy derived from 1,247 error cases in the Q2 evaluation. "
            "Frequencies sum to 1,247 (387+287+224+187+162).",
            s["VQCaption"]),
        Spacer(1, 0.15*inch),
        Paragraph(
            "Two error types (Layout Confusion and OCR Failure) both affect the Document "
            "Ingestion component, collectively accounting for 46% of all errors. "
            "The single Critical-severity category (Hallucination) is flagged for priority "
            "mitigation via confidence thresholding, already partially deployed in Q3.",
            s["VQBody"]),
        PageBreak(),
    ]

    # ── P3: Error charts ──
    story += [
        Paragraph("Error Frequency and Severity Distribution", s["VQSecHead"]),
        fig_to_rl(_q3_error_charts(), width=6.5*inch),
        Paragraph(
            "Figure 5. Left: horizontal bar chart of error counts by type. "
            "Right: donut chart of severity distribution (Critical 23%, High 46%, Medium 31%).",
            s["VQCaption"]),
        PageBreak(),
    ]

    # ── P4: Case studies ──
    story += [Paragraph("Case Studies", s["VQSecHead"]), Spacer(1, 0.1*inch)]
    cases = [
        ("Case Study 1: Multi-Column Layout Failure",
         "A multi-column table spanning the full page width was misread as two separate "
         "narrow tables due to the column separator being classified as a page boundary. "
         "This affected 78 documents in the Q2 test corpus and is the primary driver of "
         "Layout Confusion errors. Multi-scale processing is expected to resolve this by "
         "re-examining detected table regions at a lower resolution to detect column "
         "continuity before splitting."),
        ("Case Study 2: Cross-Page Table Continuation",
         "When a table occupies more than one page, the header row is not repeated on "
         "continuation pages. VIRA’s Document Ingestion component failed to associate "
         "continuation rows with their header in 43% of wide tables in the evaluation set. "
         "This caused incorrect column attribution and cascading retrieval failures. "
         "A header-propagation heuristic is under development for Q4 deployment."),
        ("Case Study 3: Figure Caption Misattribution",
         "Caption text was attributed to the wrong figure in 61 cases where two figures "
         "appeared in close proximity on the same page. The proximity scoring used by "
         "LayoutDetector assigned the caption to the nearer centroid rather than the "
         "correct figure, resulting in a precision drop of 0.12 on caption-dependent queries. "
         "A directional proximity model incorporating reading-order constraints is proposed."),
    ]
    for title, body in cases:
        story += [
            Paragraph(f"<b>{title}</b>", s["VQBody"]),
            Paragraph(body, s["VQBody"]),
            Spacer(1, 0.1*inch),
        ]
    story.append(PageBreak())

    # ── P5: Mitigation impact table ──
    story += [Paragraph("Mitigation Impact Projections", s["VQSecHead"]), Spacer(1, 0.1*inch)]
    mit = [["Error Type", "Proposed Fix", "Expected Freq.\nReduction", "Implementation\nQuarter"]]
    mit_data = [
        ("Layout Confusion", "multi-scale processing",     "45%", "Q4 2025"),
        ("Hallucination",    "confidence thresholding",    "60%", "Q3 2025"),
        ("Context Overflow", "chunk size reduction",       "70%", "Q3 2025"),
        ("OCR Failure",      "preprocessing enhancement",  "55%", "Q4 2025"),
        ("Retrieval Miss",   "query expansion",            "40%", "Q4 2025"),
    ]
    for row in mit_data:
        mit.append(list(row))
    story += [
        make_table(mit, [1.35*inch, 1.65*inch, 1.35*inch, 1.35*inch],
                   left_align_cols=[0, 1]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "Table 4. Projected error frequency reductions from proposed mitigations, "
            "with target implementation quarters.",
            s["VQCaption"]),
        Spacer(1, 0.15*inch),
        Paragraph(
            "If all mitigations are fully implemented and achieve their projected reductions, "
            "total error count is expected to drop from 1,247 to below 550, corresponding "
            "to a >55% overall error rate reduction by end of Q4 2025.", s["VQBody"]),
    ]

    doc.build(story)
    return pdf_path


# ════════════════════════════════════════════════════════════════════════════════
# PDF 4 — VTQ-Q4-2025-Comparison
# ════════════════════════════════════════════════════════════════════════════════

Q4_DATA = {
    #               P@4    R@4    MRR    NDCG10  F1     pval
    "AlphaNet":    [0.683, 0.701, 0.612, 0.731,  0.692, 0.003],
    "BetaSearch":  [0.724, 0.768, 0.698, 0.812,  0.745, 0.041],
    "GammaHybrid": [0.756, 0.834, 0.743, 0.798,  0.793, 0.089],
    "VIRA-v1":     [0.801, 0.812, 0.721, 0.803,  0.806, 0.112],
    "VIRA-v2":     [0.847, 0.871, 0.739, 0.809,  0.859, None],
}


def _radar_chart():
    cats = ["P@4", "R@4", "MRR", "NDCG@10", "F1"]
    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#F4F6F9"); ax.set_facecolor("#F4F6F9")

    for m, col in zip(METHODS, METHOD_COLORS):
        vals = Q4_DATA[m][:5]
        vals = vals + vals[:1]
        ax.plot(angles, vals, "o-", color=col, linewidth=2, label=m, markersize=5)
        ax.fill(angles, vals, color=col, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=11, fontweight="bold")
    ax.set_ylim(0.55, 0.92)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(["0.60", "0.70", "0.80", "0.90"], fontsize=8, color="#666666")
    ax.yaxis.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title("Figure 6 — Radar Chart: All Metrics by Method", fontsize=11,
                 fontweight="bold", color="#2E4057", pad=20)
    fig.tight_layout()
    return fig


def _ndcg_curves():
    """NDCG@K curves, K=1..10, one line per method."""
    # Anchor at K=10 (from Q4_DATA NDCG@10 column index 3)
    # Build increasing curves using a saturation model:  v(k) = max * (1 - exp(-k/tau))
    anchors = {m: Q4_DATA[m][3] for m in METHODS}  # NDCG@10
    starts  = {"AlphaNet": 0.52, "BetaSearch": 0.62,
                "GammaHybrid": 0.58, "VIRA-v1": 0.60, "VIRA-v2": 0.62}
    ks = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#F4F6F9"); ax.set_facecolor("#F4F6F9")
    for m, col in zip(METHODS, METHOD_COLORS):
        v10 = anchors[m]; v1 = starts[m]
        # linear interp on a compressed scale
        tau = -9 / np.log((v10 - v1) / (v10 * 1.05 - v1 + 1e-9) + 1e-9)
        vals = v1 + (v10 - v1) * (1 - np.exp(-(ks-1) / max(tau, 2)))
        vals = np.clip(vals, v1, v10)
        ax.plot(ks, vals, "o-", color=col, linewidth=2, label=m, markersize=5)
    ax.set_xlabel("K", fontsize=11)
    ax.set_ylabel("NDCG@K", fontsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(0.45, 0.87)
    ax.yaxis.grid(True, alpha=0.4); ax.xaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    ax.set_title("Figure 7 — NDCG@K Curves (K = 1..10)", fontsize=11,
                 fontweight="bold", color="#2E4057")
    fig.tight_layout()
    return fig


def make_pdf4():
    pdf_path = os.path.join(OUTPUT_DIR, "VTQ-Q4-2025-Comparison.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    s = STYLES
    story = []

    # ── P1: Cover + abstract ──
    story += [
        color_bar(), Spacer(1, 0.25*inch),
        Paragraph("VIRA Technical Quarterly", s["VQCoverTitle"]),
        Paragraph("Q4 2025 — Rigorous Comparative Study", s["VQCoverSub"]),
        Paragraph("Aether Analytics", s["VQCoverSub"]),
        Spacer(1, 0.15*inch),
        HRFlowable(width="100%", thickness=2, color=ACCENT),
        Spacer(1, 0.2*inch),
        Paragraph("Abstract", s["VQSecHead"]),
        Paragraph(
            "This report presents a rigorous Q4 2025 comparative evaluation of five "
            "document retrieval systems: AlphaNet, BetaSearch, GammaHybrid, VIRA-v1, "
            "and VIRA-v2. The study employs 500 test queries drawn from a held-out "
            "evaluation corpus, evaluating five metrics: Precision@4 (P@4), Recall@4 (R@4), "
            "Mean Reciprocal Rank (MRR), NDCG@10, and F1 score. Statistical significance "
            "testing was performed using two-sided Wilcoxon signed-rank tests at significance "
            "level α=0.05, comparing all methods against the VIRA-v2 reference system.",
            s["VQBody"]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "VIRA-v2 achieves the highest scores on P@4 (0.847), R@4 (0.871), and F1 (0.859). "
            "GammaHybrid records the best MRR (0.743) and BetaSearch the best NDCG@10 (0.812), "
            "suggesting different ranking behaviours across metrics. VIRA-v2 is statistically "
            "significantly better than AlphaNet (p=0.003) and BetaSearch (p=0.041), but the "
            "difference versus GammaHybrid (p=0.089) does not reach significance at α=0.05.",
            s["VQBody"]),
        PageBreak(),
    ]

    # ── P2: Statistical comparison table ──
    story += [Paragraph("Statistical Comparison", s["VQSecHead"]), Spacer(1, 0.1*inch)]
    stat = [["Method", "P@4", "R@4", "MRR", "NDCG@10", "F1", "p-value\nvs VIRA-v2"]]
    for m in METHODS:
        v = Q4_DATA[m]
        pval = "—" if v[5] is None else f"{v[5]:.3f}"
        stat.append([m, f"{v[0]:.3f}", f"{v[1]:.3f}", f"{v[2]:.3f}",
                      f"{v[3]:.3f}", f"{v[4]:.3f}", pval])
    story += [
        make_table(stat,
                   [1.2*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.85*inch, 0.75*inch, 0.95*inch],
                   left_align_cols=[0]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "Table 5. Full statistical comparison. P@4 values match Q2 DocBench-M results. "
            "p-values computed by two-sided Wilcoxon signed-rank test (n=500 queries).",
            s["VQCaption"]),
        Spacer(1, 0.15*inch),
        Paragraph(
            "GammaHybrid achieves the best MRR (0.743) despite not leading in other metrics, "
            "suggesting it ranks the top relevant result higher. VIRA-v2 is statistically "
            "significantly better than AlphaNet (p=0.003) and BetaSearch (p=0.041) at "
            "α=0.05, but not significantly better than GammaHybrid (p=0.089) or "
            "VIRA-v1 (p=0.112).", s["VQBody"]),
        PageBreak(),
    ]

    # ── P3: Radar chart ──
    story += [
        Paragraph("Multi-Metric Radar Analysis", s["VQSecHead"]),
        fig_to_rl(_radar_chart(), width=5.5*inch, height=5.0*inch),
        Paragraph(
            "Figure 6. Radar chart across all five evaluation metrics for five retrieval methods. "
            "BetaSearch shows a distinctive spike in NDCG@10 relative to its other metrics.",
            s["VQCaption"]),
        PageBreak(),
    ]

    # ── P4: NDCG@K curves ──
    story += [
        Paragraph("NDCG@K Curves", s["VQSecHead"]),
        fig_to_rl(_ndcg_curves(), width=6.5*inch),
        Paragraph(
            "Figure 7. NDCG@K curves showing ranking quality as K varies from 1 to 10. "
            "BetaSearch consistently surpasses other methods by K=7, consistent with its "
            "high NDCG@10 score.",
            s["VQCaption"]),
        PageBreak(),
    ]

    # ── P5: Conclusions ──
    story += [Paragraph("Conclusions", s["VQSecHead"])]
    concls = [
        "VIRA-v2 leads on precision (P@4=0.847), recall (R@4=0.871), and F1 (0.859), "
        "confirming hierarchical indexing as the most effective retrieval strategy for "
        "visually complex documents.",
        "GammaHybrid achieves the best MRR (0.743) despite not leading in other metrics, "
        "indicating it ranks the single most relevant result higher than VIRA-v2.",
        "VIRA-v2 is statistically significantly better than AlphaNet (p=0.003) and "
        "BetaSearch (p=0.041) but not GammaHybrid (p=0.089) at significance level α=0.05.",
        "BetaSearch’s NDCG@10 (0.812) exceeds VIRA-v2’s (0.809), suggesting "
        "BetaSearch may rank a broader set of relevant results slightly earlier in the "
        "ranking, even if overall precision is lower.",
        "The VIRA-v1 to VIRA-v2 upgrade (+4.6 pp P@4, +5.9 pp R@4) is statistically "
        "non-significant (p=0.112), but the consistent directional improvement across all "
        "metrics supports continued development of the VLM-enriched indexing approach.",
    ]
    for i, txt in enumerate(concls, 1):
        story.append(Paragraph(f"{i}. {txt}", s["VQFinding"]))

    doc.build(story)
    return pdf_path


# ════════════════════════════════════════════════════════════════════════════════
# PDF 5 — VTQ-2025-AnnualSummary
# ════════════════════════════════════════════════════════════════════════════════

TREND_P4 = [0.71,  0.847, 0.861, 0.874]
TREND_R4 = [0.73,  0.854, 0.868, 0.881]
QTR       = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]


def _trend_chart():
    x = np.arange(len(QTR))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#F4F6F9"); ax.set_facecolor("#F4F6F9")
    ax.plot(x, TREND_P4, "o-", color="#2E4057", linewidth=2.5,
            markersize=8, markerfacecolor="#048A81", label="P@4")
    ax.plot(x, TREND_R4, "s--", color="#E07A5F", linewidth=2.5,
            markersize=8, markerfacecolor="#E07A5F", label="R@4")
    for xi, p, r in zip(x, TREND_P4, TREND_R4):
        ax.text(xi, p+0.008, f"{p:.3f}", ha="center", fontsize=9,
                color="#2E4057", fontweight="bold")
        ax.text(xi, r-0.018, f"{r:.3f}", ha="center", fontsize=9,
                color="#E07A5F", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(QTR, fontsize=10)
    ax.set_ylim(0.65, 0.92)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Figure 8 — Quarterly P@4 and R@4 Trend (2025)", fontsize=11,
                 fontweight="bold", color="#2E4057")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def make_pdf5():
    pdf_path = os.path.join(OUTPUT_DIR, "VTQ-2025-AnnualSummary.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    s = STYLES
    story = []

    # ── P1: Cover + year-in-review ──
    story += [
        color_bar(), Spacer(1, 0.25*inch),
        Paragraph("VIRA Technical Quarterly", s["VQCoverTitle"]),
        Paragraph("2025 Annual Summary", s["VQCoverSub"]),
        Paragraph("Aether Analytics", s["VQCoverSub"]),
        Spacer(1, 0.15*inch),
        HRFlowable(width="100%", thickness=2, color=ACCENT),
        Spacer(1, 0.2*inch),
        Paragraph("Year in Review", s["VQSecHead"]),
        Paragraph(
            "2025 marked the first full operational year of VIRA, Aether Analytics’ "
            "Visual Intelligence Retrieval Architecture. The year began with the public "
            "introduction of VIRA’s architecture in Q1, establishing a baseline "
            "Precision@4 of 0.71 on internal evaluations. By Q2 the first cross-system "
            "benchmark demonstrated VIRA-v2’s superiority with P@4=0.847, and by "
            "Q3 2025 VIRA-v2 was deployed to production, serving live document retrieval "
            "workloads across enterprise clients.", s["VQBody"]),
        Spacer(1, 0.05*inch),
        Paragraph(
            "Across the year, the system indexed a total of 1.2 million documents, "
            "with quarter-over-quarter compounding growth driven by new enterprise "
            "deployments and the expansion of supported document formats. Indexing "
            "throughput improved from 12 pages/minute in Q1 to over 35 pages/minute "
            "in Q4 following storage-tier optimisations to the Tantivy BM25 index.",
            s["VQBody"]),
        Spacer(1, 0.05*inch),
        Paragraph(
            "The systematic Q3 error analysis programme identified Layout Confusion "
            "as the dominant failure mode at 31% of errors, and proposed mitigations "
            "are already showing results: the overall error rate declined from 42.0% "
            "in Q1 to 27.7% in Q4, a 34% year-over-year reduction. The 2026 roadmap "
            "targets Multimodal Fusion (Q1), Streaming Index (Q2), Video Support (Q3), "
            "and Cross-Document Synthesis (Q4) to further extend VIRA’s capabilities.",
            s["VQBody"]),
        PageBreak(),
    ]

    # ── P2: Trend chart ──
    story += [
        Paragraph("Quarterly Performance Trend", s["VQSecHead"]),
        fig_to_rl(_trend_chart(), width=6.5*inch),
        Paragraph(
            "Figure 8. Quarterly P@4 (blue circles) and R@4 (orange squares) across 2025. "
            "Both metrics show consistent quarter-over-quarter improvement.",
            s["VQCaption"]),
        PageBreak(),
    ]

    # ── P3: Top 5 Annual Findings ──
    story += [Paragraph("Top 5 Annual Findings", s["VQSecHead"]), Spacer(1, 0.1*inch)]
    findings = [
        ("Hierarchical indexing delivered the largest single P@4 gain.",
         "The introduction of hindex navigation in VIRA-v1/v2 contributed a +8.4 pp "
         "improvement from Q1 (P@4=0.71) to Q2 (P@4=0.847), the largest single-quarter "
         "accuracy jump of the year."),
        ("Layout Confusion remains the dominant error class at 31%.",
         "Despite targeted mitigations, Layout Confusion (387 cases in Q3 analysis) "
         "persists as the leading failure mode, underscoring the need for multi-scale "
         "document processing in the 2026 roadmap."),
        ("VIRA-v2 is 7.1% faster than VIRA-v1 despite higher accuracy.",
         "VIRA-v2’s latency of 3.9 s represents a 7.1% improvement over VIRA-v1’s "
         "4.2 s, achieved through index pruning optimisations that eliminated redundant "
         "embedding lookups in the hierarchical index."),
        ("BetaSearch’s NDCG@10 (0.812) suggests it ranks relevant results earlier.",
         "Despite trailing VIRA-v2 in P@4 and R@4, BetaSearch leads on NDCG@10 (0.812 "
         "vs VIRA-v2’s 0.809), indicating that its BM25 backbone ranks a broader "
         "set of relevant results higher in the retrieval list."),
        ("GammaHybrid’s high recall (0.834) makes it safest when missing results is costly.",
         "GammaHybrid’s R@4 of 0.834 on DocBench-M leads all systems, making it the "
         "recommended choice in risk-averse applications (e.g. legal or medical retrieval) "
         "where missing a relevant document has high cost."),
    ]
    for i, (title, body) in enumerate(findings, 1):
        story += [
            Paragraph(f"<b>Finding {i}: {title}</b>", s["VQFinding"]),
            Paragraph(body, s["VQBody"]),
            Spacer(1, 0.08*inch),
        ]
    story.append(PageBreak())

    # ── P4: Cross-quarter comparison table ──
    story += [Paragraph("Cross-Quarter Comparison", s["VQSecHead"]), Spacer(1, 0.1*inch)]
    cq = [
        ["Metric",            "Q1",    "Q2",    "Q3",    "Q4",    "Change"],
        ["P@4 (best)",        "0.710", "0.847", "0.861", "0.874", "+23.1%"],
        ["R@4 (best)",        "0.730", "0.871", "0.868", "0.881", "+20.7%"],
        ["Error Rate",        "42.0%", "31.0%", "28.4%", "27.7%", "−34.0%"],
        ["Avg Latency (best)","3.4 s", "1.2 s", "1.1 s", "1.0 s", "−70.6%"],
        ["Docs Indexed",      "8,432", "24,891","67,340","142,008","+1584%"],
    ]
    story += [
        make_table(cq, [1.5*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.85*inch],
                   left_align_cols=[0]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "Table 6. Cross-quarter comparison of key metrics across all four quarters of 2025. "
            "Error rate and latency show improvement (reduction); P@4, R@4, and docs indexed show growth.",
            s["VQCaption"]),
        PageBreak(),
    ]

    # ── P5: 2026 Roadmap ──
    story += [Paragraph("2026 Technology Roadmap", s["VQSecHead"]), Spacer(1, 0.1*inch)]
    road = [
        ["Quarter", "Feature", "Description", "Expected P@4 Impact"],
        ["Q1 2026", "Multimodal Fusion",
         "Joint text+image embedding training on paired visual-document corpus",
         "+3–5 pp"],
        ["Q2 2026", "Streaming Index",
         "Real-time document ingestion pipeline with sub-second index updates",
         "neutral"],
        ["Q3 2026", "Video Support",
         "Frame-level retrieval for video documents with temporal indexing",
         "new capability"],
        ["Q4 2026", "Cross-Document Synthesis",
         "Multi-hop reasoning across document corpus for complex queries",
         "+5–8 pp"],
    ]
    story += [
        make_table(road, [0.85*inch, 1.35*inch, 2.75*inch, 1.25*inch],
                   left_align_cols=[0, 1, 2]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "Table 7. 2026 technology roadmap with expected impact on P@4 retrieval precision.",
            s["VQCaption"]),
        Spacer(1, 0.15*inch),
        Paragraph(
            "The 2026 roadmap prioritises capabilities that extend VIRA beyond static "
            "text-and-image documents: Video Support in Q3 opens an entirely new modality, "
            "and Cross-Document Synthesis in Q4 targets the long-tail of complex multi-hop "
            "queries that no single-document retrieval system can currently answer. "
            "Together these features are projected to add 8–13 pp to VIRA’s "
            "precision while maintaining sub-5-second query latency.", s["VQBody"]),
    ]

    doc.build(story)
    return pdf_path


# ════════════════════════════════════════════════════════════════════════════════
# QA Dataset
# ════════════════════════════════════════════════════════════════════════════════

QA_ITEMS = [
    # ── PDF 1 ──
    {"id": "q001", "question": "What embedding model does the VIRA Embedding Engine use?",
     "answer": "gme-Qwen2-VL-7B",
     "relevant_docs": ["VTQ-Q1-2025-Architecture.pdf"], "relevant_pages": [3],
     "primary_page": 3, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q002", "question": "What is the maximum number of tokens the Embedding Engine can process?",
     "answer": "2048 tokens",
     "relevant_docs": ["VTQ-Q1-2025-Architecture.pdf"], "relevant_pages": [3],
     "primary_page": 3, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q003", "question": "What is the default top-k value used by the VIRA Query Processor?",
     "answer": "4",
     "relevant_docs": ["VTQ-Q1-2025-Architecture.pdf"], "relevant_pages": [3],
     "primary_page": 3, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q004",
     "question": "What are the four main components shown in the VIRA system architecture diagram?",
     "answer": "Document Ingestion, Embedding Engine, Hierarchical Index, and Query Processor",
     "relevant_docs": ["VTQ-Q1-2025-Architecture.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "figure", "question_type": "visual_only",
     "difficulty": "medium"},
    {"id": "q005",
     "question": "What was VIRA's average query latency and P@4 score in Q1 2025?",
     "answer": "Average latency of 3.4 seconds and P@4 of 0.71",
     "relevant_docs": ["VTQ-Q1-2025-Architecture.pdf"], "relevant_pages": [3, 5],
     "primary_page": 5, "answer_location": "cross_page", "question_type": "cross_page",
     "difficulty": "medium"},
    # ── PDF 2 ──
    {"id": "q006", "question": "What is VIRA-v2's Precision@4 on DocBench-M?",
     "answer": "0.847",
     "relevant_docs": ["VTQ-Q2-2025-Benchmarks.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q007", "question": "Which method has the lowest average latency in the Q2 benchmark?",
     "answer": "AlphaNet with 1.2 seconds",
     "relevant_docs": ["VTQ-Q2-2025-Benchmarks.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q008", "question": "Which method achieves the highest Recall@4 on DocBench-M?",
     "answer": "VIRA-v2 with R@4 of 0.871",
     "relevant_docs": ["VTQ-Q2-2025-Benchmarks.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "exact_lookup",
     "difficulty": "easy"},
    {"id": "q009",
     "question": "How well does VIRA-v2 perform in terms of speed compared to accuracy?",
     "answer": "VIRA-v2 achieves the highest accuracy (P@4=0.847 on DocBench-M) with a competitive latency of 3.9 seconds, second only to AlphaNet in speed",
     "relevant_docs": ["VTQ-Q2-2025-Benchmarks.pdf"], "relevant_pages": [2, 4],
     "primary_page": 2, "answer_location": "table", "question_type": "semantic_paraphrase",
     "difficulty": "medium"},
    {"id": "q010",
     "question": "What training dataset was used to pretrain the AlphaNet model?",
     "answer": "The document does not mention AlphaNet's training dataset",
     "relevant_docs": [], "relevant_pages": [],
     "primary_page": None, "answer_location": "none", "question_type": "out_of_scope",
     "difficulty": "easy"},
    # ── PDF 3 ──
    {"id": "q011", "question": "What is the most frequently occurring error type?",
     "answer": "Layout Confusion with 387 occurrences (31.0% of total errors)",
     "relevant_docs": ["VTQ-Q3-2025-ErrorAnalysis.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q012", "question": "How many error types are classified as Critical severity?",
     "answer": "One: Hallucination",
     "relevant_docs": ["VTQ-Q3-2025-ErrorAnalysis.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "exact_lookup",
     "difficulty": "easy"},
    {"id": "q013",
     "question": "What mitigation strategy is recommended for Context Overflow errors?",
     "answer": "chunk size reduction",
     "relevant_docs": ["VTQ-Q3-2025-ErrorAnalysis.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q014", "question": "How many total error cases were analyzed in Q3?",
     "answer": "1,247 error cases",
     "relevant_docs": ["VTQ-Q3-2025-ErrorAnalysis.pdf"], "relevant_pages": [1],
     "primary_page": 1, "answer_location": "text", "question_type": "exact_lookup",
     "difficulty": "easy"},
    {"id": "q015",
     "question": "What proportion of errors are related to the document ingestion stage?",
     "answer": "46% — Layout Confusion (31%) and OCR Failure (15%) both affect Document Ingestion",
     "relevant_docs": ["VTQ-Q3-2025-ErrorAnalysis.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "cross_page",
     "difficulty": "hard"},
    # ── PDF 4 ──
    {"id": "q016",
     "question": "Is VIRA-v2 statistically significantly better than AlphaNet at the 0.01 significance level?",
     "answer": "Yes, with p=0.003 which is below the 0.01 threshold",
     "relevant_docs": ["VTQ-Q4-2025-Comparison.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "medium"},
    {"id": "q017", "question": "Which method achieves the best NDCG@10 score?",
     "answer": "BetaSearch with 0.812",
     "relevant_docs": ["VTQ-Q4-2025-Comparison.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q018", "question": "Which method achieves the best MRR score and what is it?",
     "answer": "GammaHybrid with MRR of 0.743",
     "relevant_docs": ["VTQ-Q4-2025-Comparison.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q019",
     "question": "Is VIRA-v2 significantly better than GammaHybrid at the α=0.05 level?",
     "answer": "No, the p-value is 0.089 which exceeds α=0.05, so the difference is not statistically significant",
     "relevant_docs": ["VTQ-Q4-2025-Comparison.pdf"], "relevant_pages": [2],
     "primary_page": 2, "answer_location": "table", "question_type": "exact_lookup",
     "difficulty": "medium"},
    {"id": "q020",
     "question": "What does the radar chart reveal about BetaSearch's performance profile?",
     "answer": "BetaSearch shows a distinctive spike in NDCG@10 and R@4 relative to its other metrics",
     "relevant_docs": ["VTQ-Q4-2025-Comparison.pdf"], "relevant_pages": [3],
     "primary_page": 3, "answer_location": "figure", "question_type": "visual_only",
     "difficulty": "hard"},
    # ── PDF 5 ──
    {"id": "q021",
     "question": "By what percentage did the best P@4 score improve from Q1 to Q4 2025?",
     "answer": "23.1%",
     "relevant_docs": ["VTQ-2025-AnnualSummary.pdf"], "relevant_pages": [4],
     "primary_page": 4, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q022", "question": "When was VIRA-v2 deployed to production?",
     "answer": "Q3 2025",
     "relevant_docs": ["VTQ-2025-AnnualSummary.pdf"], "relevant_pages": [1],
     "primary_page": 1, "answer_location": "text", "question_type": "exact_lookup",
     "difficulty": "easy"},
    {"id": "q023",
     "question": "What feature is planned for Q1 2026 in the roadmap?",
     "answer": "Multimodal Fusion — joint text and image embedding training",
     "relevant_docs": ["VTQ-2025-AnnualSummary.pdf"], "relevant_pages": [5],
     "primary_page": 5, "answer_location": "table", "question_type": "table_cell",
     "difficulty": "easy"},
    {"id": "q024",
     "question": "How did the error rate change from Q1 to Q4 2025?",
     "answer": "The error rate decreased from 42.0% in Q1 to 27.7% in Q4, a 34% reduction",
     "relevant_docs": ["VTQ-2025-AnnualSummary.pdf"], "relevant_pages": [1, 4],
     "primary_page": 4, "answer_location": "cross_page", "question_type": "cross_page",
     "difficulty": "medium"},
    # ── Cross-document ──
    {"id": "q025",
     "question": "Which method from the Q2 benchmarks achieved the best NDCG@10 in the Q4 comparative study?",
     "answer": "BetaSearch with NDCG@10 of 0.812",
     "relevant_docs": ["VTQ-Q2-2025-Benchmarks.pdf", "VTQ-Q4-2025-Comparison.pdf"],
     "relevant_pages": [2, 2], "primary_page": 2,
     "answer_location": "cross_doc", "question_type": "cross_doc", "difficulty": "medium"},
    {"id": "q026",
     "question": "What is the relationship between the most common error type found in Q3 and the annual P@4 improvement?",
     "answer": "Layout Confusion (31% of Q3 errors) was the dominant failure mode. Its mitigation through multi-scale processing contributed to the P@4 improvement from 0.71 in Q1 to 0.874 in Q4",
     "relevant_docs": ["VTQ-Q3-2025-ErrorAnalysis.pdf", "VTQ-2025-AnnualSummary.pdf"],
     "relevant_pages": [2, 4], "primary_page": 2,
     "answer_location": "cross_doc", "question_type": "cross_doc", "difficulty": "hard"},
    {"id": "q027",
     "question": "How does GammaHybrid's recall advantage compare to its ranking in the annual findings?",
     "answer": "GammaHybrid achieves the highest recall (R@4=0.834 on DocBench-M in Q2) and is specifically highlighted in Annual Finding 5 as the best choice when missing results is costly",
     "relevant_docs": ["VTQ-Q2-2025-Benchmarks.pdf", "VTQ-2025-AnnualSummary.pdf"],
     "relevant_pages": [2, 3], "primary_page": 2,
     "answer_location": "cross_doc", "question_type": "cross_doc", "difficulty": "hard"},
    {"id": "q028",
     "question": "What is the expected P@4 impact of the Q4 2026 feature compared to the total improvement achieved across all of 2025?",
     "answer": "The Q4 2026 Cross-Document Synthesis feature is expected to add 5–8 pp, compared to the 16.4 pp improvement (0.71→0.874) achieved across all of 2025",
     "relevant_docs": ["VTQ-Q4-2025-Comparison.pdf", "VTQ-2025-AnnualSummary.pdf"],
     "relevant_pages": [5, 4], "primary_page": 5,
     "answer_location": "cross_doc", "question_type": "cross_doc", "difficulty": "hard"},
]


def write_qa_dataset():
    qa_path = os.path.join(OUTPUT_DIR, "qa_dataset.jsonl")
    with open(qa_path, "w", encoding="utf-8") as f:
        for item in QA_ITEMS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return qa_path


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("Generating VIRA benchmark PDFs...")

    pdf1 = make_pdf1()
    print(f"  [1/5] {os.path.basename(pdf1)}")

    pdf2 = make_pdf2()
    print(f"  [2/5] {os.path.basename(pdf2)}")

    pdf3 = make_pdf3()
    print(f"  [3/5] {os.path.basename(pdf3)}")

    pdf4 = make_pdf4()
    print(f"  [4/5] {os.path.basename(pdf4)}")

    pdf5 = make_pdf5()
    print(f"  [5/5] {os.path.basename(pdf5)}")

    qa_path = write_qa_dataset()
    print(f"\nQA dataset written: {os.path.basename(qa_path)}")

    # ── Summary ──
    import fitz  # PyMuPDF — verify page counts
    pdfs = [pdf1, pdf2, pdf3, pdf4, pdf5]
    print("\n" + "="*55)
    print("BENCHMARK GENERATION SUMMARY")
    print("="*55)
    total_pages = 0
    for p in pdfs:
        with fitz.open(p) as doc:
            n = doc.page_count
        total_pages += n
        status = "OK" if n == 5 else f"WARNING: {n} pages"
        print(f"  {os.path.basename(p)}: {n} pages [{status}]")
    print(f"\n  PDFs generated  : {len(pdfs)}")
    print(f"  Total pages     : {total_pages}")
    print(f"  QA items        : {len(QA_ITEMS)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*55)


if __name__ == "__main__":
    main()
