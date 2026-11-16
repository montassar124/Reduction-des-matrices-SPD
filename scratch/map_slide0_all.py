import pptx

prs = pptx.Presentation(r"Modele_Poster_ENSI_60x100cm (1).pptx.pptx")
slide = prs.slides[0]

for idx, shape in enumerate(slide.shapes):
    text = ""
    if shape.has_text_frame:
        text = " // ".join(p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip())
    elif shape.has_table:
        text = f"TABLE {len(shape.table.rows)}x{len(shape.table.columns)}"
    elif hasattr(shape, "shapes"):
        subtexts = []
        for s in shape.shapes:
            if s.has_text_frame:
                subtexts.append(s.name + ": " + " // ".join(p.text.strip() for p in s.text_frame.paragraphs if p.text.strip()))
        text = "GROUP: " + " | ".join(subtexts)
    
    pos = f"pos=({shape.left/914400*2.54:.1f}, {shape.top/914400*2.54:.1f})cm"
    size = f"size=({shape.width/914400*2.54:.1f}x{shape.height/914400*2.54:.1f})cm"
    print(f"Shape {idx:02d} | {shape.name} ({shape.shape_type}) | {pos} | {size} | {text}")
