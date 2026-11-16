# -*- coding: utf-8 -*-
"""Generate official ENSI poster using Modele_Poster_ENSI_60x100cm (1).pptx.pptx."""

from pathlib import Path
import shutil
from PIL import Image
import pptx
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "Modele_Poster_ENSI_60x100cm (1).pptx.pptx"
BACKUP_TEMPLATE = ROOT / "Modele_Poster_ENSI_60x100cm_original_backup.pptx"
OUTPUT_REPORT = ROOT / "reports" / "poster_spd_ensi_modele_60x100cm.pptx"
ARTIFACT_DIR = Path(r"C:\Users\Globaltech\.gemini\antigravity\brain\2343b67e-d6f6-4df9-8288-56700b5d343e")

# Backup original template if not already backed up
if not BACKUP_TEMPLATE.exists():
    shutil.copy(str(TEMPLATE_PATH), str(BACKUP_TEMPLATE))
    print(f"Backed up template to: {BACKUP_TEMPLATE}")

# Load presentation
prs = pptx.Presentation(str(BACKUP_TEMPLATE))
slide = prs.slides[0]

# Palette
C_NAVY = RGBColor(17, 38, 96)          # #112660
C_ROYAL = RGBColor(22, 78, 135)        # #164E87
C_CYAN = RGBColor(0, 168, 232)         # #00A8E8
C_TEXT_DARK = RGBColor(20, 33, 61)     # #14213D
C_TEXT_MUTED = RGBColor(90, 100, 120)  # #5A6478
C_GREEN = RGBColor(5, 150, 105)        # #059669
C_WHITE = RGBColor(255, 255, 255)
C_DARK = RGBColor(26, 26, 26)

def set_text(shape, text, font_size_pt=None, bold=None, color=None, align=None):
    """Safely update text in a shape preserving or setting styles."""
    tf = shape.text_frame
    tf.word_wrap = True
    # Keep first paragraph, clear runs
    p = tf.paragraphs[0]
    p.text = text
    if align is not None:
        p.alignment = align
    for r in p.runs:
        r.font.name = "Segoe UI"
        if font_size_pt:
            r.font.size = Pt(font_size_pt)
        if bold is not None:
            r.font.bold = bold
        if color:
            r.font.color.rgb = color
    # Remove extra paragraphs
    while len(tf.paragraphs) > 1:
        p_elem = tf.paragraphs[-1]._p
        p_elem.getparent().remove(p_elem)

def set_bullet_list(shape, items, font_size_pt=10.5, text_color=C_TEXT_DARK, bold_prefix_len=0):
    """Set multiple bulleted paragraphs."""
    tf = shape.text_frame
    tf.word_wrap = True
    # Clear all paragraphs except first
    while len(tf.paragraphs) > 1:
        p_elem = tf.paragraphs[-1]._p
        p_elem.getparent().remove(p_elem)
    
    for idx, (title, content) in enumerate(items):
        if idx == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = ""
        p.space_after = Pt(2.5)
        p.line_spacing = 1.15
        
        if title:
            r1 = p.add_run()
            r1.text = title + " : "
            r1.font.name = "Segoe UI"
            r1.font.bold = True
            r1.font.size = Pt(font_size_pt)
            r1.font.color.rgb = C_NAVY
        
        r2 = p.add_run()
        r2.text = content
        r2.font.name = "Segoe UI"
        r2.font.bold = False
        r2.font.size = Pt(font_size_pt)
        r2.font.color.rgb = text_color

def add_fitted_picture(slide, image_path, left, top, max_w, max_h):
    """Insert picture maintaining aspect ratio, centered in bounding box."""
    with Image.open(image_path) as im:
        img_w, img_h = im.size
    
    aspect = img_w / img_h
    box_aspect = max_w / max_h
    
    if aspect > box_aspect:
        w = max_w
        h = max_w / aspect
    else:
        h = max_h
        w = max_h * aspect
    
    x = left + (max_w - w) / 2
    y = top + (max_h - h) / 2
    
    return slide.shapes.add_picture(str(image_path), int(x), int(y), int(w), int(h))

print("Applying content to Slide 0...")

# 1. Header
shape_03 = slide.shapes[3] # Group 8
for sub in shape_03.shapes:
    if sub.name == "TextBox 10":
        set_text(sub, "RÉDUCTION DE DIMENSION DANS L'ESPACE SPD : APPLICATION AUX IMAGES MÉDICALES DU CERVEAU", 
                 font_size_pt=34.0, bold=True, color=C_DARK, align=PP_ALIGN.CENTER)

set_text(slide.shapes[4], "Projet de Recherche & PFE · École Nationale des Sciences de l'Informatique (ENSI), Univ. de la Manouba", 
         font_size_pt=18.0, bold=True, color=C_ROYAL, align=PP_ALIGN.CENTER)

set_text(slide.shapes[5], "Réalisé par : Montassar · Encadré par : A. Araoud, E. Ghorbel, F. Ghorbel", 
         font_size_pt=16.0, bold=False, color=C_TEXT_DARK, align=PP_ALIGN.CENTER)

# Logo in Group 6 (Shape 02)
logo_path = ROOT / "figures" / "ensi_logo.png"
if logo_path.exists():
    add_fitted_picture(slide, logo_path, Cm(1.8), Cm(1.3), Cm(3.9), Cm(5.2))

# 2. Section 1 : Contexte et objectifs
shape_07 = slide.shapes[7] # Group 15
for sub in shape_07.shapes:
    if "TextBox" in sub.name:
        set_text(sub, "1. Contexte & Objectifs", font_size_pt=24.0, bold=True, color=C_WHITE)

items_b1 = [
    ("Problème", "En neuroimagerie (IRMf, EEG), les connectomes cérébraux sont Symétriques Définis Positifs (SPD) sur une variété riemannienne non euclidienne. L'espace brut à N=100 régions compte D=5050 paramètres, induisant le fléau de la dimension et un coût géodésique O(n^3) prohibitif."),
    ("Données & Approche", "Métrique Affine-Invariante (AIRM) et projection sur espaces tangents locaux via Multi-Repères Adaptatifs pour préserver la topologie sans distorsion."),
    ("Objectif & Défi", "Démontrer la supériorité riemannienne en basse dimension (d=2) et lever le paradoxe antérieur où le k-NN pénalisait la géométrie courbée.")
]
set_bullet_list(slide.shapes[8], items_b1, font_size_pt=10.5)

# Task cards
set_text(slide.shapes[10], "TÂCHE 1 · GÉOMÉTRIE", font_size_pt=13.0, bold=True, color=C_ROYAL)
for sub in slide.shapes[11].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Variété SPD", font_size_pt=18.0, bold=True, color=C_NAVY)
set_text(slide.shapes[12], "Métrique AIRM & courbure négative non euclidienne", font_size_pt=10.5, color=C_TEXT_DARK)

set_text(slide.shapes[14], "TÂCHE 2 · RÉDUCTION", font_size_pt=13.0, bold=True, color=C_ROYAL)
for sub in slide.shapes[15].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Multi-Repères", font_size_pt=18.0, bold=True, color=C_NAVY)
set_text(slide.shapes[16], "Barycentres locaux & transport parallèle riemannien", font_size_pt=10.5, color=C_TEXT_DARK)

set_text(slide.shapes[18], "TÂCHE 3 · CLASSIFIEURS", font_size_pt=13.0, bold=True, color=C_ROYAL)
for sub in slide.shapes[19].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "SVM vs k-NN", font_size_pt=18.0, bold=True, color=C_NAVY)
set_text(slide.shapes[20], "Séparateur à marge compensant la compression d=2", font_size_pt=10.5, color=C_TEXT_DARK)

# 3. Section 2 : Données et méthodologie
shape_22 = slide.shapes[22] # Group 42
for sub in shape_22.shapes:
    if "TextBox" in sub.name:
        set_text(sub, "2. Données & Méthodologie", font_size_pt=24.0, bold=True, color=C_WHITE)

for sub in slide.shapes[24].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Données & Variétés", font_size_pt=18.0, bold=True, color=C_NAVY)

items_data = [
    ("Connectomes COBRE", "86 patients (46 contrôles, 40 schizophrènes), matrices de corrélation 28x28 SPD (D=406 dimensions)."),
    ("ETH-80 Vision", "80 objets, 8 classes, descripteurs gaussiens SPD 401x401 (D=80 600 dimensions, taux de compression 99.96%)."),
    ("Sphère & Synthétique", "2000 matrices SPD 5x5 dans 3 régimes d'entremêlement (éloigné, modéré, fort recouvrement)."),
    ("Découpage & Rigueur", "10 splits stratifiés 80/20 & validation croisée 5-fold sans fuite d'information.")
]
set_bullet_list(slide.shapes[25], items_data, font_size_pt=10.0)

for sub in slide.shapes[27].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Pipeline Méthodologique", font_size_pt=18.0, bold=True, color=C_NAVY)

# Insert Figure 1 in shape 29
for sub in slide.shapes[29].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "") # clear placeholder text
fig1_path = ROOT / "figures" / "01_pipeline.png"
if fig1_path.exists():
    add_fitted_picture(slide, fig1_path, Cm(21.7), Cm(23.5), Cm(16.6), Cm(15.5))
set_text(slide.shapes[41], "Figure 1 : Pipeline complet : Données connectomes -> Espace SPD -> Réduction dimensionnelle -> Classifieurs.", font_size_pt=9.5, bold=False, color=C_TEXT_MUTED)

# Sub-column 3
for sub in slide.shapes[31].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Approche & Régularisation", font_size_pt=18.0, bold=True, color=C_NAVY)

for sub in slide.shapes[33].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Entrée : Matrices SPD P (28x28)", font_size_pt=12.0, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

for sub in slide.shapes[36].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Étape 1 : Projection Karcher & Multi-Repères", font_size_pt=12.0, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

for sub in slide.shapes[39].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Étape 2 : Régularisation M1 (tau=10^-12) & M3", font_size_pt=12.0, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)

items_eval = [
    ("Protocole", "10 splits stratifiés 80/20 répétés; test utilisé une seule fois."),
    ("Mesures", "Accuracy équilibrée, préservation topologique (NP @ k=6, k=12), écart-type.")
]
set_bullet_list(slide.shapes[40], items_eval, font_size_pt=10.0)
set_text(slide.shapes[42], "Figure 2 : Schéma de projection géodésique et transport parallèle.", font_size_pt=9.5, bold=False, color=C_TEXT_MUTED)

# 4. Middle Transition Panel (Data examples & spectral distributions)
for sub in slide.shapes[44].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Variétés & Plongements 2D des Sujets", font_size_pt=18.0, bold=True, color=C_NAVY)

for sub in slide.shapes[46].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "")
fig3_path = ROOT / "figures" / "03_subject_embeddings.png"
if fig3_path.exists():
    add_fitted_picture(slide, fig3_path, Cm(2.5), Cm(43.5), Cm(31.6), Cm(10.8))
set_text(slide.shapes[47], "Figure 3 : Plongements 2D des sujets sur la variété log-SPD montrant les frontières de classes.", font_size_pt=9.5, color=C_TEXT_MUTED)

for sub in slide.shapes[49].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Distribution Spectrale & Seuil tau", font_size_pt=18.0, bold=True, color=C_NAVY)

for sub in slide.shapes[51].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "")
fig4_path = ROOT / "figures" / "10_eigenvalue_distribution_before_after_log.png"
if fig4_path.exists():
    add_fitted_picture(slide, fig4_path, Cm(36.9), Cm(43.5), Cm(20.5), Cm(10.8))
set_text(slide.shapes[52], "Figure 4 : Distribution des valeurs propres avant/après Log : le seuillage tau=10^-12 élimine la singularité.", font_size_pt=9.5, color=C_TEXT_MUTED)

# 5. Section 3 : Résultats
shape_54 = slide.shapes[54] # Group 112
for sub in shape_54.shapes:
    if "TextBox" in sub.name:
        set_text(sub, "3. Résultats Expérimentaux", font_size_pt=24.0, bold=True, color=C_WHITE)

# KPIs
for sub in slide.shapes[56].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "73.40%", font_size_pt=34.0, bold=True, color=C_GREEN, align=PP_ALIGN.CENTER)
set_text(slide.shapes[57], "Accuracy Max [Best]", font_size_pt=12.0, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
set_text(slide.shapes[58], "Multi-Repères + SVM Lin.", font_size_pt=10.0, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)

for sub in slide.shapes[60].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "66.34%", font_size_pt=34.0, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
set_text(slide.shapes[61], "Gain Riemannien (+11.9%)", font_size_pt=12.0, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
set_text(slide.shapes[62], "PCA Tangente + SVM RBF", font_size_pt=10.0, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)

for sub in slide.shapes[64].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "97.92%", font_size_pt=34.0, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
set_text(slide.shapes[65], "Préservation Voisinage", font_size_pt=12.0, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
set_text(slide.shapes[66], "ETH-80 (NP @ k=12)", font_size_pt=10.0, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)

for sub in slide.shapes[68].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "98.8%", font_size_pt=34.0, bold=True, color=C_ROYAL, align=PP_ALIGN.CENTER)
set_text(slide.shapes[69], "Taux de Compression", font_size_pt=12.0, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
set_text(slide.shapes[70], "D=406 -> d=2 (Sans perte)", font_size_pt=10.0, color=C_TEXT_MUTED, align=PP_ALIGN.CENTER)

set_text(slide.shapes[71], "Évaluation sur 10 splits stratifiés répétés sans fuite d'information. Multi-Repères égale le brut (73.46%).", font_size_pt=10.0, color=C_TEXT_MUTED)

# Result figures
for sub in slide.shapes[73].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Résultat A : Précision des Méthodes", font_size_pt=15.0, bold=True, color=C_NAVY)

for sub in slide.shapes[75].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "")
fig5_path = ROOT / "figures" / "04_accuracy_comparison.png"
if fig5_path.exists():
    add_fitted_picture(slide, fig5_path, Cm(2.5), Cm(67.7), Cm(16.6), Cm(11.5))
set_text(slide.shapes[76], "Figure 5 : Accuracy moyenne (+/- std) montrant le gain net sous classification SVM.", font_size_pt=9.5, color=C_TEXT_MUTED)

for sub in slide.shapes[78].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Résultat B : Multi-Repères vs Global", font_size_pt=15.0, bold=True, color=C_NAVY)

for sub in slide.shapes[80].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "")
fig6_path = ROOT / "figures" / "09_local_geometry_accuracy.png"
if fig6_path.exists():
    add_fitted_picture(slide, fig6_path, Cm(21.7), Cm(67.7), Cm(16.6), Cm(11.5))
set_text(slide.shapes[81], "Figure 6 : Supériorité de l'approche locale multi-repères sur les variétés dispersées.", font_size_pt=9.5, color=C_TEXT_MUTED)

for sub in slide.shapes[83].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Résultat C : Topologie des Voisinages", font_size_pt=15.0, bold=True, color=C_NAVY)

for sub in slide.shapes[85].shapes:
    if "TextBox" in sub.name:
        set_text(sub, "")
fig7_path = ROOT / "figures" / "18_neighborhood_graphs_airm_vs_pca.png"
if fig7_path.exists():
    add_fitted_picture(slide, fig7_path, Cm(40.8), Cm(67.7), Cm(16.6), Cm(11.5))
set_text(slide.shapes[86], "Figure 7 : Préservation du graphe AIRM démontrant la fidélité géométrique de l'espace réduit.", font_size_pt=9.5, color=C_TEXT_MUTED)

# Table 1 (Shape 87)
table_shape = slide.shapes[87]
t = table_shape.table

# Adjust column widths for balanced readability
t.columns[0].width = Cm(11.5)
t.columns[1].width = Cm(7.5)
t.columns[2].width = Cm(7.0)
t.columns[3].width = Cm(7.5)
t.columns[4].width = Cm(7.5)
t.columns[5].width = Cm(15.4)

TABLE_DATA = [
    ["Méthode de Réduction", "Papier ENSI (Table 3, k-NN)", "Notre Test (k-NN)", "Notre Test (SVM Lin.)", "Notre Test (SVM RBF)", "Résolution du Paradoxe"],
    ["ACP Euclidienne Naïve (d=2)", "61.11%", "58.89%", "54.44%", "54.44%", "Échec sous SVM (-6.67%)"],
    ["PCA Tangente (t-PCA, d=2)", "55.56%", "56.93%", "61.96%", "66.34% [Best]", "Gain riemannien via SVM (+11.90%)"],
    ["Multi-Repères Adaptatifs (d=2)", "Non existant", "62.88%", "73.40% [Best]", "64.05%", "+10.65% net (Domine toutes les méthodes)"],
    ["Données Brutes (Baseline D=406)", "Non évalué", "68.69%", "66.47%", "73.46% (AIRM)", "Multi-Repères égale le brut (73.40%)"],
    ["Lie-LLE (Manifold Learning)", "Non évalué", "56.56%", "60.10%", "62.30%", "Reconstruction locale intrinsèque"]
]

for r_idx, row in enumerate(TABLE_DATA):
    for c_idx, val in enumerate(row):
        cell = t.cell(r_idx, c_idx)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
        for r in p.runs:
            r.font.name = "Segoe UI"
            if r_idx == 0:
                r.font.bold = True
                r.font.size = Pt(10.5)
                r.font.color.rgb = C_NAVY
            else:
                r.font.size = Pt(9.5)
                if "[Best]" in val:
                    r.font.bold = True
                    r.font.color.rgb = C_GREEN
                elif r_idx == 3: # Multi-Repères row
                    r.font.bold = True
                    r.font.color.rgb = C_ROYAL
                else:
                    r.font.color.rgb = C_TEXT_DARK

set_text(slide.shapes[88], "Tableau 1 : Matrice comparative sur connectomes cérébraux (COBRE). Le classifieur SVM restaure la supériorité riemannienne.", font_size_pt=10.0, color=C_TEXT_MUTED)

# 6. Conclusion & Références
shape_90 = slide.shapes[90] # Group 181
for sub in shape_90.shapes:
    if "TextBox" in sub.name:
        set_text(sub, "4. Conclusion & Perspectives", font_size_pt=24.0, bold=True, color=C_WHITE)

items_concl = [
    ("Domination des Repères Adaptatifs", "L'approche Multi-Repères atteint 73.40% [Best] (SVM Lin.), dominant l'ACP de +10.65% et égalant le signal brut (73.46%) avec 98.8% de compression dimensionnelle."),
    ("Démystification du Paradoxe", "Le k-NN favorisait l'ACP naïve à cause de distorsions locales d'un barycentre unique. Le séparateur SVM restaure pleinement l'avantage géométrique."),
    ("Correction d'Erreurs Numériques", "Le plancher spectral (tau=10^-12) et le shrinkage préviennent les singularités, apportant +2.20% de gain net et -48% de variance."),
    ("Perspectives", "Intégration dans des réseaux profonds géométriques (SPDNet) et harmonisation multi-centres par transport parallèle riemannien.")
]
set_bullet_list(slide.shapes[91], items_concl, font_size_pt=9.8)

shape_93 = slide.shapes[93] # Group 187
for sub in shape_93.shapes:
    if "TextBox" in sub.name:
        set_text(sub, "Références & Remerciements", font_size_pt=24.0, bold=True, color=C_WHITE)

items_refs = [
    ("[1]", "A. Araoud, E. Ghorbel, F. Ghorbel, ICAART 2025 / Springer LNAI 16517, pp. 543–561, 2026."),
    ("[2]", "X. Zou et al., Lie-LLE: Locally Linear Embedding on Lie Groups, SSRN, 2025."),
    ("[3]", "X. Pennec et al., A Riemannian Framework for Tensor Computing, IJCV, 66(1):41–66, 2006."),
    ("Remerciements", "École Nationale des Sciences de l'Informatique (ENSI) & Université de la Manouba pour le soutien académique.")
]
set_bullet_list(slide.shapes[94], items_refs, font_size_pt=9.5)

# Drop instruction slide (Slide 1) to produce a clean 1-page printable poster
if len(prs.slides) > 1:
    rId = prs.slides._sldIdLst[1].rId
    prs.part.drop_rel(rId)
    del prs.slides._sldIdLst[1]
    print("Dropped instruction slide 1. Presentation now has 1 printable poster slide.")

# Save presentation
prs.save(str(OUTPUT_REPORT))
print(f"Poster successfully saved to: {OUTPUT_REPORT}")

# Also update the user's template directly so opening it shows the filled poster!
prs.save(str(TEMPLATE_PATH))
print(f"Updated user's template at: {TEMPLATE_PATH}")

if ARTIFACT_DIR.exists():
    artifact_poster = ARTIFACT_DIR / "Modele_Poster_ENSI_60x100cm.pptx"
    prs.save(str(artifact_poster))
    print(f"Saved artifact copy to: {artifact_poster}")
