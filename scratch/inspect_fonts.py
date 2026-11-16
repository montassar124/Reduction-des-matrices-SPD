import pptx

prs = pptx.Presentation(r"Modele_Poster_ENSI_60x100cm (1).pptx.pptx")
slide = prs.slides[0]

def inspect_text_styling(shape, name_prefix=""):
    if shape.has_text_frame:
        for p_idx, p in enumerate(shape.text_frame.paragraphs):
            for r_idx, r in enumerate(p.runs):
                font_name = r.font.name
                font_size = r.font.size.pt if r.font.size else "default"
                bold = r.font.bold
                color = r.font.color.rgb if (r.font.color and hasattr(r.font.color, 'rgb')) else "default"
                print(f"{name_prefix} P{p_idx} R{r_idx} [{font_name}, {font_size}pt, bold={bold}, color={color}]: '{r.text}'")
    elif shape.has_table:
        for r_idx, row in enumerate(shape.table.rows):
            for c_idx, cell in enumerate(row.cells):
                for p_idx, p in enumerate(cell.text_frame.paragraphs):
                    for run_idx, r in enumerate(p.runs):
                        font_size = r.font.size.pt if r.font.size else "default"
                        print(f"{name_prefix} Table cell({r_idx},{c_idx}) [{r.font.name}, {font_size}pt]: '{r.text}'")

for i, s in enumerate(slide.shapes):
    if s.has_text_frame:
        inspect_text_styling(s, f"Shape {i:02d} ({s.name}):")
    elif hasattr(s, "shapes"):
        for sub in s.shapes:
            if sub.has_text_frame:
                inspect_text_styling(sub, f"Shape {i:02d} -> {sub.name}:")
