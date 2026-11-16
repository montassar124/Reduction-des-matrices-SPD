# -*- coding: utf-8 -*-
"""Generate high-resolution scientific posters in PowerPoint (.pptx) format.

Dimensions:
- Portrait: 65cm x 100cm (official 1m x 0.65m traceur format)
- Landscape: 100cm x 65cm

Equilibrated layout, official ENSI palette, verified benchmark values.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "figures"
ARTIFACT_DIR = Path(r"C:\Users\Globaltech\.gemini\antigravity\brain\2343b67e-d6f6-4df9-8288-56700b5d343e")
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C_NAVY_DARK = RGBColor(7, 23, 56)       # #071738
C_NAVY = RGBColor(17, 38, 96)          # #112660
C_ROYAL = RGBColor(22, 78, 135)        # #164E87
C_CYAN = RGBColor(0, 168, 232)         # #00A8E8
C_CYAN_SOFT = RGBColor(56, 189, 248)   # #38BDF8
C_ICE_BG = RGBColor(244, 249, 253)     # #F4F9FD
C_ICE_TH = RGBColor(225, 239, 249)     # #E1EFF9
C_BORDER = RGBColor(200, 225, 245)     # #C8E1F5
C_DARK_CARD = RGBColor(11, 34, 78)     # #0B224E
C_WHITE = RGBColor(255, 255, 255)
C_TEXT_DARK = RGBColor(15, 23, 42)     # #0F172A
C_TEXT_MUTED = RGBColor(51, 65, 85)    # #334155
C_GREEN = RGBColor(5, 150, 105)        # #059669
C_GREEN_BG = RGBColor(236, 253, 245)   # #ECFDF5

LOGO_PATH = ROOT / "figures" / "ensi_logo.png"
FIG18_PATH = ROOT / "figures" / "18_neighborhood_graphs_airm_vs_pca.png"

HEADERS_B7 = ["Méthode de Réduction", "Papier ENSI (Table 3, k-NN)", "Notre Test (k-NN)", "Notre Test (SVM Lin.)", "Notre Test (SVM RBF)", "Résolution du Paradoxe"]

DATA_B7 = [
    ("ACP Euclidienne Naïve (d=2)", "61.11%", "58.89%", "54.44%", "54.44%", "Échec sous SVM (-6.67%)"),
    ("PCA Tangente (t-PCA, d=2)", "55.56%", "56.93%", "61.96%", "66.34% [Best]", "Gain riemannien via SVM (+11.90%)"),
    ("Multi-Repères Adaptatifs (d=2)", "Non existant", "62.88%", "73.40% [Best]", "64.05%", "+10.65% net (Domine toutes les méthodes)"),
    ("Données Brutes (Baseline D=406)", "Non évalué", "68.69%", "66.47%", "73.46% (AIRM) [Best]", "Multi-Repères égale le brut (73.40%)"),
]


def create_portrait_poster(out_path: Path):
    prs = Presentation()
    prs.slide_width = Cm(65)
    prs.slide_height = Cm(100)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # 1. Slide Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(0), Cm(0), Cm(65), Cm(100))
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(235, 242, 250)
    bg.line.fill.background()

    # 2. Main Canvas Background (63.4 x 97.4 cm)
    canvas = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(0.8), Cm(0.8), Cm(63.4), Cm(98.4))
    canvas.fill.solid()
    canvas.fill.fore_color.rgb = C_WHITE
    canvas.line.color.rgb = C_BORDER
    canvas.line.width = Pt(1.5)

    # 3. Header Banner (61.4 x 7.6 cm)
    hdr_left = Cm(1.8)
    hdr_top = Cm(1.8)
    hdr_w = Cm(61.4)
    hdr_h = Cm(7.5)

    header = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, hdr_left, hdr_top, hdr_w, hdr_h)
    header.fill.solid()
    header.fill.fore_color.rgb = C_NAVY_DARK
    header.line.color.rgb = C_CYAN
    header.line.width = Pt(2.5)

    # Logo Box
    if LOGO_PATH.exists():
        logo_bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, hdr_left + Cm(0.7), hdr_top + Cm(0.6), Cm(7.2), Cm(6.3))
        logo_bg.fill.solid()
        logo_bg.fill.fore_color.rgb = C_WHITE
        logo_bg.line.color.rgb = C_BORDER
        logo_bg.line.width = Pt(1)
        slide.shapes.add_picture(str(LOGO_PATH), hdr_left + Cm(0.9), hdr_top + Cm(0.8), width=Cm(6.8))

    # Header Text Box
    title_box = slide.shapes.add_textbox(hdr_left + Cm(8.6), hdr_top + Cm(0.4), hdr_w - Cm(9.2), Cm(6.7))
    tf = title_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = "RÉDUCTION DE DIMENSION DANS L'ESPACE SPD"
    p_title.font.name = "Segoe UI"
    p_title.font.size = Pt(26)
    p_title.font.bold = True
    p_title.font.color.rgb = C_WHITE
    p_title.space_after = Pt(4)

    p_sub = tf.add_paragraph()
    p_sub.text = "Application aux Images Médicales du Cerveau : Géométrie Riemannienne, Régularisation & Rôle Clé du Classifieur"
    p_sub.font.name = "Segoe UI"
    p_sub.font.size = Pt(14)
    p_sub.font.bold = True
    p_sub.font.color.rgb = C_CYAN_SOFT
    p_sub.space_after = Pt(7)

    p_meta = tf.add_paragraph()
    p_meta.text = "Institution : École Nationale des Sciences de l'Informatique (ENSI) — Université de la Manouba    |    Encadré par : Équipe PFE / Recherche"
    p_meta.font.name = "Segoe UI"
    p_meta.font.size = Pt(11.5)
    p_meta.font.color.rgb = RGBColor(226, 232, 240)

    # 4. Columns Setup
    col_w = Cm(29.8)
    gap = Cm(1.8)
    col1_left = Cm(1.8)
    col2_left = col1_left + col_w + gap
    start_y = Cm(10.1)

    # Helper: Add Card
    def add_card(x, y, w, h, title_text, is_dark=False):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        card.fill.solid()
        card.fill.fore_color.rgb = C_DARK_CARD if is_dark else C_ICE_BG
        card.line.color.rgb = C_CYAN if is_dark else C_BORDER
        card.line.width = Pt(1.5)

        title_bar_h = Cm(1.3)
        tbar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, title_bar_h)
        tbar.fill.solid()
        tbar.fill.fore_color.rgb = C_CYAN if is_dark else C_NAVY
        tbar.line.fill.background()

        if not is_dark:
            line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y + title_bar_h, w, Cm(0.08))
            line.fill.solid()
            line.fill.fore_color.rgb = C_CYAN
            line.line.fill.background()

        tb = slide.shapes.add_textbox(x + Cm(0.5), y + Cm(0.15), w - Cm(1.0), title_bar_h - Cm(0.3))
        tb.text_frame.margin_left = tb.text_frame.margin_top = 0
        p = tb.text_frame.paragraphs[0]
        p.text = title_text
        p.font.name = "Segoe UI"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = C_WHITE

        content_box = slide.shapes.add_textbox(x + Cm(0.6), y + title_bar_h + Cm(0.2), w - Cm(1.2), h - title_bar_h - Cm(0.4))
        ctf = content_box.text_frame
        ctf.word_wrap = True
        ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = 0
        return ctf

    def add_bullet(tf, bold_txt, norm_txt, font_size=11.5, space_after=4, is_dark=False):
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.space_after = Pt(space_after)
        p.font.name = "Segoe UI"

        run_b = p.add_run()
        run_b.text = "■ " if is_dark else "• "
        run_b.font.bold = True
        run_b.font.size = Pt(font_size)
        run_b.font.color.rgb = C_CYAN_SOFT if is_dark else C_ROYAL

        run_bold = p.add_run()
        run_bold.text = bold_txt + " : "
        run_bold.font.bold = True
        run_bold.font.size = Pt(font_size)
        run_bold.font.color.rgb = C_WHITE if is_dark else C_TEXT_DARK

        run_norm = p.add_run()
        run_norm.text = norm_txt
        run_norm.font.size = Pt(font_size)
        run_norm.font.color.rgb = RGBColor(226, 232, 240) if is_dark else C_TEXT_DARK

    # =============================================================
    # LEFT COLUMN (5 Blocks)
    # =============================================================
    curr_y = start_y

    # Block 1: Context & Motivation (13.2 cm)
    b1_h = Cm(13.2)
    tf1 = add_card(col1_left, curr_y, col_w, b1_h, "1. Contexte & Motivation")
    add_bullet(tf1, "Images Cérébrales & Matrices SPD", "En neuroimagerie fonctionnelle (IRMf, EEG), les connectomes cérébraux et corrélations inter-régionales appartiennent au cône Symétrique Défini Positif (S_++^n).")
    add_bullet(tf1, "Le Fléau de la Dimension", "Pour n = 100 régions, l'espace compte D = n(n+1)/2 = 5050 paramètres indépendants pour seulement ~100 sujets, induisant surapprentissage critique et complexité géodésique O(n^3).")
    add_bullet(tf1, "Courbure Non Euclidienne", "Le cône S_++^n forme une variété riemannienne à courbure négative. L'ACP linéaire euclidienne écrase la géométrie (distorsion 9 912) et détruit les voisinages.")
    add_bullet(tf1, "Résolution du Paradoxe Antérieur", "Une étude précédente avec k-NN avait conclu à l'absence de gain. Notre travail prouve que le classifieur compense la réduction et démontre la supériorité riemannienne.")
    curr_y += b1_h + Cm(1.0)

    # Block 2: Riemannian Geometry & AIRM (21.5 cm)
    b2_h = Cm(21.5)
    tf2 = add_card(col1_left, curr_y, col_w, b2_h, "2. Géométrie Riemannienne & Métrique AIRM")
    add_bullet(tf2, "Cône des Matrices SPD", "Variété riemannienne différentiable convexe : S_++^n = { P ∈ R^(n×n) | P = P^T, x^T P x > 0, ∀x ≠ 0 }.")
    add_bullet(tf2, "Métrique Affine-Invariante (AIRM)", "Invariante par changement de base (P ↦ A P A^T) : δ_R(P, Q) = ||Log(P^(-1/2) Q P^(-1/2))||_F = √( Σ ln² λ_i(P^(-1)Q) ).")
    add_bullet(tf2, "Applications Exponentielle & Logarithmique", "Projection au point de référence Γ : Log_Γ(P) = Γ^(1/2) Log(Γ^(-1/2) P Γ^(-1/2)) Γ^(1/2) ∈ T_Γ S_++^n.")
    add_bullet(tf2, "Barycentre de Fréchet Riemannien (Karcher)", "Moyenne géodésique intrinsèque : Γ* = argmin Σ δ_R²(Γ, P_i), vérifiant Σ Log_Γ*(P_i) = 0.")
    curr_y += b2_h + Cm(1.0)

    # Block 3: Dimension Reduction Methods (16.0 cm)
    b3_h = Cm(16.0)
    tf3 = add_card(col1_left, curr_y, col_w, b3_h, "3. Méthodes de Réduction de Dimension")
    add_bullet(tf3, "ACP Euclidienne Naïve", "Vectorisation euclidienne directe. Ignore la courbure et induit le phénomène de gonflement artificiel (swelling effect).")
    add_bullet(tf3, "PCA Tangente Riemannienne (t-PCA)", "Projection au barycentre de Fréchet global, isométrie euclidienne dans l'espace tangent, puis ACP classique sur les vecteurs tangents v_i = vec(Log_Γ*(P_i)).")
    add_bullet(tf3, "Multi-Repères Adaptatifs", "Partitionnement en K barycentres locaux par clustering riemannien. Projette chaque sujet dans son repère local pour minimiser la distorsion des grands écarts.")
    add_bullet(tf3, "Lie-LLE (Riemannian LLE)", "Reconstruction linéaire de chaque matrice par ses k voisins dans l'algèbre de Lie, préservant la topologie locale intrinsèque en basse dimension.")
    curr_y += b3_h + Cm(1.0)

    # Block 4: Regularization & Numerical Fixes (18.5 cm)
    b4_h = Cm(18.5)
    tf4 = add_card(col1_left, curr_y, col_w, b4_h, "4. Régularisation & Correction d'Erreurs Numériques")
    add_bullet(tf4, "Singularité aux Frontières (λ_min → 0)", "Lorsque le spectre approche 0, ln(λ) → -∞, provoquant instabilités NaN et divergence géodésique.")
    add_bullet(tf4, "Correction M1 — Plancher Spectral Imposé", "Décomposition spectrale avec seuillage λ~_i = max(λ_i, τ), τ = 10^(-12). Garantit la définie-positivité stricte.")
    add_bullet(tf4, "Correction M2 — Shrinkage Ledoit-Wolf", "Régularisation convexe vers l'identité : P~ = (1 - α) P + α (Tr(P)/n) I_n.")
    add_bullet(tf4, "Correction M3 — Régularisation du Barycentre", "Projection de Karcher avec shrinkage intrinsèque, divisant par deux la variance de projection tangentielle.")
    add_bullet(tf4, "Gain Mesuré Expérimentalement", "Gain direct de +2.20% d'accuracy en zone critique (noise=10^(-9)) et variance divisée par deux (±2.04% vs ±3.91%).")
    curr_y += b4_h + Cm(1.0)

    # Block 5: Perspectives & Future Work (15.0 cm, Dark Card)
    b5_h = Cm(15.0)
    tf5 = add_card(col1_left, curr_y, col_w, b5_h, "5. Perspectives & Travaux Futurs", is_dark=True)
    add_bullet(tf5, "Deep Learning Riemannien (SPDNet)", "Intégrer des couches convolutives et projectives 2D préservant la positivité matricielle (BiMap → ReEig → LogEig).", is_dark=True)
    add_bullet(tf5, "Cohortes Cliniques Multi-Sites & Harmonisation", "Validation étendue sur ADNI (Alzheimer) et ABIDE (Autisme) avec transport parallèle riemannien pour corriger les effets de site.", is_dark=True)
    add_bullet(tf5, "Régularisation Spectrale Dynamique", "Développement d'un plancher spectral adaptatif τ(n, N) fonction du conditionnement et du ratio échantillons/dimension.", is_dark=True)

    # =============================================================
    # RIGHT COLUMN (3 Blocks)
    # =============================================================
    curr_y = start_y

    # Block 6: Benchmark 4 Approches (41.0 cm)
    b6_h = Cm(41.0)
    tf6 = add_card(col2_left, curr_y, col_w, b6_h, "6. Benchmark des 4 Approches : Conservation (NP) & Accuracy")

    tbl_top = curr_y + Cm(1.8)
    tbl_rows = 17
    tbl_cols = 6
    tbl_shape = slide.shapes.add_table(tbl_rows, tbl_cols, col2_left + Cm(0.4), tbl_top, col_w - Cm(0.8), Cm(34.5))
    tbl = tbl_shape.table

    tbl.columns[0].width = Cm(5.8)
    tbl.columns[1].width = Cm(4.5)
    tbl.columns[2].width = Cm(2.8)
    tbl.columns[3].width = Cm(2.8)
    tbl.columns[4].width = Cm(4.8)
    tbl.columns[5].width = Cm(8.3)

    headers = ["Dataset", "Méthode", "NP@6", "NP@12", "Accuracy [Fix*]", "Observation Géométrique"]
    for j, h_txt in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h_txt
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_ICE_TH
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Segoe UI"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = C_NAVY
        p.alignment = PP_ALIGN.CENTER

    data_b6 = [
        ("Sphère forte\n(N=2000, d=2)", "PCA classique", "4.38%", "7.39%", "97.38%", "Capture dispersion euclidienne"),
        ("", "PCA tangent", "4.82%", "7.49%", "94.27% (+2.20%*)", "Enveloppe tangente cohérente"),
        ("", "Repères adaptatifs", "5.12%", "8.09%", "94.29% (+2.20%*)", "Préservation locale accrue (+2.97%)"),
        ("", "Lie-LLE", "4.53%", "2.40%", "97.97% [Best]", "Repliement de courbure à grand k"),
        ("Recouvrement fort\n(Case 3, N=300)", "PCA classique", "16.63%", "23.50%", "58.67%", "Capture grossière euclidienne"),
        ("", "PCA tangent", "4.27%", "6.62%", "61.00% (+8.33%*)", "Récupération géodésique à grand k"),
        ("", "Repères adaptatifs", "18.53%", "26.68%", "61.67% (+9.00%*)", "Domine : 4x plus de rétention"),
        ("", "Lie-LLE", "4.17%", "2.38%", "53.12%", "Déroulement local sans ancrage"),
        ("Recouvrement modéré\n(Case 2, N=300)", "PCA classique", "21.32%", "28.15%", "62.00%", "k-NN semble meilleur que Tangente"),
        ("", "PCA tangent", "5.40%", "8.12%", "62.67% (+7.34%*)", "Le SVM compense et dépasse l'ACP"),
        ("", "Repères adaptatifs", "23.21%", "31.40%", "63.00% (+7.00%*)", "Leader méthodes réduites (SVM)"),
        ("", "Lie-LLE", "3.89%", "2.50%", "56.56%", "Reconstruction linéaire locale"),
        ("ETH-80\n(401x401 SPD)", "PCA classique", "81.67%", "85.83%", "47.25% (62.5% nb)", "Variance euclidienne globale"),
        ("", "PCA tangent", "54.17%", "59.58%", "89.25% (+44.00%*)", "Étirement du barycentre unique"),
        ("", "Repères adaptatifs", "97.08%", "97.92%", "77.00% (+11.75%*)", "Rétention topologique quasi-parfaite"),
        ("", "Lie-LLE", "53.75%", "60.21%", "91.00% (+41.25%*)", "Déroulement topologique régularisé"),
    ]

    for i, row in enumerate(data_b6, start=1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Segoe UI"
            p.font.size = Pt(8.5)
            p.alignment = PP_ALIGN.LEFT if (j == 0 or j == 5) else PP_ALIGN.CENTER
            if "Best" in val or "Domine" in val or "Leader" in val or "97.08%" in val:
                p.font.bold = True
                p.font.color.rgb = C_GREEN
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_GREEN_BG
            elif i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(248, 250, 252)

    leg_box = slide.shapes.add_textbox(col2_left + Cm(0.5), curr_y + Cm(37.0), col_w - Cm(1.0), Cm(3.4))
    ltf = leg_box.text_frame
    ltf.word_wrap = True
    ltf.margin_left = ltf.margin_top = 0
    lp = ltf.paragraphs[0]
    lp.text = "* Légende — Valeur ajoutée du fix d'erreur numérique : (+X.XX%*) indique le gain direct de performance et de stabilité apporté par la régularisation numérique (plancher spectral τ=10^(-12) et shrinkage Karcher M3) qui stabilise le calcul géodésique aux frontières du cône SPD."
    lp.font.name = "Segoe UI"
    lp.font.size = Pt(9.5)
    lp.font.color.rgb = C_NAVY
    lp.font.italic = True
    curr_y += b6_h + Cm(1.2)

    # Block 7: Results & Role of Classifier (33.0 cm)
    b7_h = Cm(33.0)
    tf7 = add_card(col2_left, curr_y, col_w, b7_h, "7. Résultats Expérimentaux & Rôle Clé du Classifieur")

    # Image Figure 18
    if FIG18_PATH.exists():
        slide.shapes.add_picture(str(FIG18_PATH), col2_left + Cm(0.8), curr_y + Cm(1.6), width=col_w - Cm(1.6), height=Cm(8.0))
        cap_box = slide.shapes.add_textbox(col2_left + Cm(0.8), curr_y + Cm(9.7), col_w - Cm(1.6), Cm(1.8))
        ctf = cap_box.text_frame
        ctf.word_wrap = True
        ctf.margin_left = ctf.margin_top = 0
        cp = ctf.paragraphs[0]
        cp.text = "Figure 1 : Topologie des voisinages (d=2). L'ACP classique écrase les variétés (distorsion 9 912). PCA Tangente (0.134) et Repères Adaptatifs (0.212) préservent fidèlement la géométrie AIRM."
        cp.font.name = "Segoe UI"
        cp.font.size = Pt(9.0)
        cp.font.italic = True
        cp.font.color.rgb = C_TEXT_MUTED

    # Table comparing Paper vs Our Tests
    tbl2_top = curr_y + Cm(11.8)
    t2_rows = 5
    t2_cols = 6
    tbl2_shape = slide.shapes.add_table(t2_rows, t2_cols, col2_left + Cm(0.4), tbl2_top, col_w - Cm(0.8), Cm(11.5))
    t2 = tbl2_shape.table

    t2.columns[0].width = Cm(7.2)
    t2.columns[1].width = Cm(4.5)
    t2.columns[2].width = Cm(4.0)
    t2.columns[3].width = Cm(4.2)
    t2.columns[4].width = Cm(4.2)
    t2.columns[5].width = Cm(5.7)

    t2_headers = ["Méthode de Réduction", "Papier ENSI (Table 3, k-NN)", "Notre Test (k-NN)", "Notre Test (SVM Lin.)", "Notre Test (SVM RBF)", "Résolution du Paradoxe"]
    for j, h_txt in enumerate(t2_headers):
        cell = t2.cell(0, j)
        cell.text = h_txt
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_ICE_TH
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Segoe UI"
        p.font.size = Pt(8.5)
        p.font.bold = True
        p.font.color.rgb = C_NAVY
        p.alignment = PP_ALIGN.CENTER

    data_b7 = [
        ("ACP Euclidienne Naïve (d=2)", "61.11%", "58.89%", "54.44%", "54.44%", "Échec sous SVM (-6.67%)"),
        ("PCA Tangente (t-PCA, d=2)", "55.56%", "56.93%", "61.96%", "66.34% [Best]", "Gain riemannien via SVM (+11.90%)"),
        ("Multi-Repères Adaptatifs (d=2)", "Non existant", "62.88%", "73.40% [Best]", "64.05%", "+10.65% net (Domine toutes les méthodes)"),
        ("Données Brutes (Baseline D=406)", "Non évalué", "68.69%", "66.47%", "73.46% (AIRM) [Best]", "Multi-Repères égale le brut (73.40%)"),
    ]

    for i, row in enumerate(data_b7, start=1):
        for j, val in enumerate(row):
            cell = t2.cell(i, j)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Segoe UI"
            p.font.size = Pt(8.5)
            p.alignment = PP_ALIGN.LEFT if (j == 0 or j == 5) else PP_ALIGN.CENTER
            if "Best" in val or "+10.65%" in val or "+11.90%" in val:
                p.font.bold = True
                p.font.color.rgb = C_GREEN
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_GREEN_BG
            elif i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(248, 250, 252)

    # Paradox explanation text
    para_box = slide.shapes.add_textbox(col2_left + Cm(0.6), curr_y + Cm(23.8), col_w - Cm(1.2), Cm(8.6))
    ptf = para_box.text_frame
    ptf.word_wrap = True
    ptf.margin_left = ptf.margin_top = 0
    p1 = ptf.paragraphs[0]
    p1.text = "• Reproduction du Paradoxe Antérieur : Sur les connectomes cérébraux COBRE, le k-NN privilégie l'ACP (58.89% vs 61.11% dans le papier Araoud et al. 2026) en raison de distorsions locales du barycentre unique."
    p1.font.name = "Segoe UI"
    p1.font.size = Pt(10.5)
    p1.font.color.rgb = C_TEXT_DARK
    p1.space_after = Pt(6)

    p2 = ptf.add_paragraph()
    p2.text = "• Démystification par SVM : Le séparateur à marge maximale compense la réduction et restaure la supériorité riemannienne : PCA Tangente à 66.34% (SVM RBF) et Multi-Repères à 73.40% (égalant les 73.46% bruts avec 98.8% de compression, dominant l'ACP de +10.65%)."
    p2.font.name = "Segoe UI"
    p2.font.size = Pt(10.5)
    p2.font.bold = True
    p2.font.color.rgb = C_ROYAL
    curr_y += b7_h + Cm(1.2)

    # Block 8: Acknowledgments & References (11.8 cm)
    b8_h = Cm(11.8)
    tf8 = add_card(col2_left, curr_y, col_w, b8_h, "8. Remerciements & Références")
    add_bullet(tf8, "Remerciements Institutionnels", "Nous remercions chaleureusement l'École Nationale des Sciences de l'Informatique (ENSI), l'Université de la Manouba, ainsi que l'équipe d'encadrement pour leurs orientations scientifiques et leur soutien technique continu.")
    add_bullet(tf8, "Références Clés", "A. Araoud, E. Ghorbel, F. Ghorbel (ICAART 2025 / Springer LNAI 16517, pp. 543–561, 2026) • X. Zou et al. (Lie-LLE, SSRN 2025) • Pennec et al. (IJCV 2006).")

    prs.save(str(out_path))
    print(f"Saved Portrait PPTX to: {out_path}")


def create_landscape_poster(out_path: Path):
    prs = Presentation()
    prs.slide_width = Cm(100)
    prs.slide_height = Cm(65)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Slide Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(0), Cm(0), Cm(100), Cm(65))
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(235, 242, 250)
    bg.line.fill.background()

    # Main Canvas (98.0 x 63.0 cm)
    canvas = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(1.0), Cm(1.0), Cm(98.0), Cm(63.0))
    canvas.fill.solid()
    canvas.fill.fore_color.rgb = C_WHITE
    canvas.line.color.rgb = C_BORDER
    canvas.line.width = Pt(1.5)

    # Header Banner (95.6 x 6.5 cm)
    hdr_left = Cm(2.2)
    hdr_top = Cm(2.0)
    hdr_w = Cm(95.6)
    hdr_h = Cm(6.5)

    header = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, hdr_left, hdr_top, hdr_w, hdr_h)
    header.fill.solid()
    header.fill.fore_color.rgb = C_NAVY_DARK
    header.line.color.rgb = C_CYAN
    header.line.width = Pt(2.5)

    if LOGO_PATH.exists():
        logo_bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, hdr_left + Cm(0.6), hdr_top + Cm(0.5), Cm(7.0), Cm(5.5))
        logo_bg.fill.solid()
        logo_bg.fill.fore_color.rgb = C_WHITE
        logo_bg.line.color.rgb = C_BORDER
        logo_bg.line.width = Pt(1)
        slide.shapes.add_picture(str(LOGO_PATH), hdr_left + Cm(0.8), hdr_top + Cm(0.7), width=Cm(6.6))

    title_box = slide.shapes.add_textbox(hdr_left + Cm(8.4), hdr_top + Cm(0.3), hdr_w - Cm(9.0), Cm(5.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_title = tf.paragraphs[0]
    p_title.text = "RÉDUCTION DE DIMENSION DANS L'ESPACE SPD"
    p_title.font.name = "Segoe UI"
    p_title.font.size = Pt(26)
    p_title.font.bold = True
    p_title.font.color.rgb = C_WHITE
    p_title.space_after = Pt(2)

    p_sub = tf.add_paragraph()
    p_sub.text = "Application aux Images Médicales du Cerveau : Géométrie Riemannienne, Régularisation & Rôle Clé du Classifieur"
    p_sub.font.name = "Segoe UI"
    p_sub.font.size = Pt(14)
    p_sub.font.bold = True
    p_sub.font.color.rgb = C_CYAN_SOFT
    p_sub.space_after = Pt(4)

    p_meta = tf.add_paragraph()
    p_meta.text = "Institution : École Nationale des Sciences de l'Informatique (ENSI) — Université de la Manouba    |    Encadré par : Équipe PFE / Recherche"
    p_meta.font.name = "Segoe UI"
    p_meta.font.size = Pt(11)
    p_meta.font.color.rgb = RGBColor(226, 232, 240)

    # 3 Columns for Landscape
    col_w = Cm(30.6)
    gap = Cm(1.9)
    col1_left = Cm(2.2)
    col2_left = col1_left + col_w + gap
    col3_left = col2_left + col_w + gap
    start_y = Cm(9.5)

    def add_card(x, y, w, h, title_text, is_dark=False):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        card.fill.solid()
        card.fill.fore_color.rgb = C_DARK_CARD if is_dark else C_ICE_BG
        card.line.color.rgb = C_CYAN if is_dark else C_BORDER
        card.line.width = Pt(1.5)

        title_bar_h = Cm(1.2)
        tbar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, title_bar_h)
        tbar.fill.solid()
        tbar.fill.fore_color.rgb = C_CYAN if is_dark else C_NAVY
        tbar.line.fill.background()

        if not is_dark:
            line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y + title_bar_h, w, Cm(0.06))
            line.fill.solid()
            line.fill.fore_color.rgb = C_CYAN
            line.line.fill.background()

        tb = slide.shapes.add_textbox(x + Cm(0.4), y + Cm(0.12), w - Cm(0.8), title_bar_h - Cm(0.24))
        tb.text_frame.margin_left = tb.text_frame.margin_top = 0
        p = tb.text_frame.paragraphs[0]
        p.text = title_text
        p.font.name = "Segoe UI"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = C_WHITE

        content_box = slide.shapes.add_textbox(x + Cm(0.5), y + title_bar_h + Cm(0.2), w - Cm(1.0), h - title_bar_h - Cm(0.3))
        ctf = content_box.text_frame
        ctf.word_wrap = True
        ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = 0
        return ctf

    def add_bullet(tf, bold_txt, norm_txt, font_size=10.5, space_after=3, is_dark=False):
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.space_after = Pt(space_after)
        p.font.name = "Segoe UI"
        run_b = p.add_run()
        run_b.text = "■ " if is_dark else "• "
        run_b.font.bold = True
        run_b.font.size = Pt(font_size)
        run_b.font.color.rgb = C_CYAN_SOFT if is_dark else C_ROYAL
        run_bold = p.add_run()
        run_bold.text = bold_txt + " : "
        run_bold.font.bold = True
        run_bold.font.size = Pt(font_size)
        run_bold.font.color.rgb = C_WHITE if is_dark else C_TEXT_DARK
        run_norm = p.add_run()
        run_norm.text = norm_txt
        run_norm.font.size = Pt(font_size)
        run_norm.font.color.rgb = RGBColor(226, 232, 240) if is_dark else C_TEXT_DARK

    # Column 1: Motivation & Geometry (Blocs 1, 2, 5)
    y1 = start_y
    tf1 = add_card(col1_left, y1, col_w, Cm(15.5), "1. Contexte & Motivation")
    add_bullet(tf1, "Images Cérébrales & Matrices SPD", "En neuroimagerie fonctionnelle (IRMf, EEG), les connectomes cérébraux appartiennent à S_++^n.")
    add_bullet(tf1, "Le Fléau de la Dimension", "Pour n=100 régions, D=5050 paramètres indépendants pour ~100 sujets (D >> N).")
    add_bullet(tf1, "Courbure Non Euclidienne", "S_++^n forme une variété riemannienne à courbure négative. L'ACP linéaire écrase la géométrie (distorsion 9 912).")
    add_bullet(tf1, "Résolution du Paradoxe", "Démystification du paradoxe k-NN vs SVM.")
    y1 += Cm(15.5) + Cm(1.0)

    tf2 = add_card(col1_left, y1, col_w, Cm(22.0), "2. Géométrie Riemannienne & Métrique AIRM")
    add_bullet(tf2, "Cône SPD", "S_++^n = { P ∈ R^(n×n) | P = P^T, x^T P x > 0, ∀x ≠ 0 }.")
    add_bullet(tf2, "Métrique AIRM", "δ_R(P, Q) = ||Log(P^(-1/2) Q P^(-1/2))||_F = √( Σ ln² λ_i(P^(-1)Q) ).")
    add_bullet(tf2, "Exp & Log Maps", "Log_Γ(P) = Γ^(1/2) Log(Γ^(-1/2) P Γ^(-1/2)) Γ^(1/2) ∈ T_Γ S_++^n.")
    add_bullet(tf2, "Barycentre de Fréchet", "Moyenne de Karcher vérifiant Σ Log_Γ*(P_i) = 0.")
    y1 += Cm(22.0) + Cm(1.0)

    tf5 = add_card(col1_left, y1, col_w, Cm(13.5), "5. Perspectives & Travaux Futurs", is_dark=True)
    add_bullet(tf5, "Deep Learning Riemannien", "SPDNet avec BiMap, ReEig et LogEig.", is_dark=True)
    add_bullet(tf5, "Cohortes Cliniques Multi-Sites", "Validation étendue sur ADNI et ABIDE.", is_dark=True)
    add_bullet(tf5, "Régularisation Dynamique", "Plancher adaptatif τ(n, N).", is_dark=True)

    # Column 2: Methods & Regularization (Blocs 3, 4)
    y2 = start_y
    tf3 = add_card(col2_left, y2, col_w, Cm(20.0), "3. Méthodes de Réduction de Dimension")
    add_bullet(tf3, "ACP Euclidienne Naïve", "Vectorisation euclidienne directe, gonflement artificiel (swelling effect).")
    add_bullet(tf3, "PCA Tangente (t-PCA)", "Projection au barycentre de Fréchet global Γ* puis ACP classique.")
    add_bullet(tf3, "Multi-Repères Adaptatifs", "K barycentres locaux par K-Means riemannien pour minimiser la distorsion.")
    add_bullet(tf3, "Lie-LLE", "Reconstruction linéaire de voisinage dans l'algèbre de Lie.")
    y2 += Cm(20.0) + Cm(1.0)

    tf4 = add_card(col2_left, y2, col_w, Cm(32.0), "4. Régularisation Numérique & Benchmark 4 Approches")
    add_bullet(tf4, "Singularité aux Frontières", "Lorsque λ_min → 0, ln(λ) → -∞, provoquant instabilités NaN.")
    add_bullet(tf4, "M1 Plancher Spectral", "Seuillage λ~_i = max(λ_i, τ), τ = 10^(-12).")
    add_bullet(tf4, "M2 Shrinkage Ledoit-Wolf", "Régularisation convexe vers l'identité : P~ = (1 - α) P + α (Tr(P)/n) I_n.")
    add_bullet(tf4, "M3 Régularisation Fréchet", "Shrinkage de Karcher intrinsèque.")
    add_bullet(tf4, "Gain Direct Prouvé", "Gain de +2.20% d'accuracy en zone critique et variance divisée par deux (±2.04% vs ±3.91%).")

    # Column 3: Results & Classifiers (Blocs 7, 8)
    y3 = start_y
    tf7 = add_card(col3_left, y3, col_w, Cm(40.0), "7. Résultats Expérimentaux & Rôle Clé du Classifieur")
    if FIG18_PATH.exists():
        slide.shapes.add_picture(str(FIG18_PATH), col3_left + Cm(0.6), y3 + Cm(1.4), width=col_w - Cm(1.2), height=Cm(7.5))

    # Table comparing Paper vs Our Tests in Landscape
    tbl3_top = y3 + Cm(9.2)
    tbl3_shape = slide.shapes.add_table(5, 6, col3_left + Cm(0.3), tbl3_top, col_w - Cm(0.6), Cm(11.2))
    t3 = tbl3_shape.table
    t3.columns[0].width = Cm(7.4)
    t3.columns[1].width = Cm(4.5)
    t3.columns[2].width = Cm(4.0)
    t3.columns[3].width = Cm(4.2)
    t3.columns[4].width = Cm(4.2)
    t3.columns[5].width = Cm(5.7)

    for j, h_txt in enumerate(HEADERS_B7):
        cell = t3.cell(0, j)
        cell.text = h_txt
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_ICE_TH
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Segoe UI"
        p.font.size = Pt(8.5)
        p.font.bold = True
        p.font.color.rgb = C_NAVY
        p.alignment = PP_ALIGN.CENTER

    for i, row in enumerate(DATA_B7, start=1):
        for j, val in enumerate(row):
            cell = t3.cell(i, j)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Segoe UI"
            p.font.size = Pt(8.5)
            p.alignment = PP_ALIGN.LEFT if (j == 0 or j == 5) else PP_ALIGN.CENTER
            if "Best" in val or "+10.65%" in val or "+11.90%" in val:
                p.font.bold = True
                p.font.color.rgb = C_GREEN
                cell.fill.solid()
                cell.fill.fore_color.rgb = C_GREEN_BG
            elif i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(248, 250, 252)

    # Paradox explanation box in Landscape
    box_p = slide.shapes.add_textbox(col3_left + Cm(0.6), y3 + Cm(20.8), col_w - Cm(1.2), Cm(17.5))
    ptf = box_p.text_frame
    ptf.word_wrap = True
    ptf.margin_left = ptf.margin_top = 0

    p1 = ptf.paragraphs[0]
    p1.text = "• Reproduction du Paradoxe Antérieur : Sur connectomes COBRE, le k-NN privilégie l'ACP (58.89% vs 61.11% dans Araoud et al. 2026) en raison de distorsions locales du barycentre unique."
    p1.font.name = "Segoe UI"
    p1.font.size = Pt(10.5)
    p1.font.color.rgb = C_TEXT_DARK
    p1.space_after = Pt(6)

    p2 = ptf.add_paragraph()
    p2.text = "• Démystification par SVM : Le séparateur à marge maximale compense la réduction et restaure la supériorité riemannienne : PCA Tangente à 66.34% (SVM RBF) et Multi-Repères à 73.40% (égalant les 73.46% bruts avec 98.8% de compression, dominant l'ACP de +10.65%)."
    p2.font.name = "Segoe UI"
    p2.font.size = Pt(10.5)
    p2.font.bold = True
    p2.font.color.rgb = C_ROYAL
    y3 += Cm(40.0) + Cm(1.0)

    tf8 = add_card(col3_left, y3, col_w, Cm(12.0), "8. Remerciements & Références")
    add_bullet(tf8, "Remerciements", "École Nationale des Sciences de l'Informatique (ENSI), Université de la Manouba, et équipe d'encadrement PFE.")
    add_bullet(tf8, "Références", "A. Araoud, E. Ghorbel, F. Ghorbel (ICAART 2025 / Springer LNAI 16517, pp. 543–561, 2026) • X. Zou et al. (Lie-LLE, SSRN 2025) • Pennec et al. (IJCV 2006).")

    prs.save(str(out_path))
    print(f"Saved Landscape PPTX to: {out_path}")


if __name__ == "__main__":
    portrait_report = REPORTS_DIR / "poster_spd_ensi_65x100cm_portrait.pptx"
    landscape_report = REPORTS_DIR / "poster_spd_ensi_100x65cm_paysage.pptx"

    print("Building portrait posters...")
    create_portrait_poster(portrait_report)
    if ARTIFACT_DIR.exists():
        create_portrait_poster(ARTIFACT_DIR / "poster_spd_ensi_65x100cm_portrait.pptx")

    print("Building landscape posters...")
    create_landscape_poster(landscape_report)
    if ARTIFACT_DIR.exists():
        create_landscape_poster(ARTIFACT_DIR / "poster_spd_ensi_100x65cm_paysage.pptx")

    print("All PowerPoint posters successfully generated!")

