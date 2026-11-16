import pptx

prs = pptx.Presentation(r"Modele_Poster_ENSI_60x100cm (1).pptx.pptx")
slide = prs.slides[0]

def inspect_shape(shape, prefix=""):
    text_content = ""
    if shape.has_text_frame:
        paras = [p.text for p in shape.text_frame.paragraphs if p.text.strip()]
        text_content = " | ".join(paras)
    elif shape.has_table:
        rows = []
        for row in shape.table.rows:
            rows.append(" [" + " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells) + "]")
        text_content = "TABLE: " + " ; ".join(rows)
    
    print(f"{prefix}[{shape.name}] ({shape.shape_type}) pos=({shape.left/914400*2.54:.1f}, {shape.top/914400*2.54:.1f}) size=({shape.width/914400*2.54:.1f}x{shape.height/914400*2.54:.1f})cm")
    if text_content:
        print(f"{prefix}    TXT: {text_content[:200]}")
    
    if hasattr(shape, "shapes"): # Group
        for sub in shape.shapes:
            inspect_shape(sub, prefix + "  ")

for i, s in enumerate(slide.shapes):
    inspect_shape(s, f"{i:02d}: ")
