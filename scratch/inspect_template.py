import pptx

prs = pptx.Presentation(r"Modele_Poster_ENSI_60x100cm (1).pptx.pptx")
print(f"Slide width: {prs.slide_width / 914400 * 2.54:.2f} cm")
print(f"Slide height: {prs.slide_height / 914400 * 2.54:.2f} cm")
print(f"Number of slides: {len(prs.slides)}")

for idx, slide in enumerate(prs.slides):
    print(f"\nSlide {idx}: {len(slide.shapes)} shapes")
    for s_idx, shape in enumerate(slide.shapes):
        text = shape.text[:80].replace("\n", " ") if shape.has_text_frame else "<no text>"
        print(f"  Shape {s_idx:02d}: name='{shape.name}', type={shape.shape_type}, pos=({shape.left/914400*2.54:.1f}, {shape.top/914400*2.54:.1f})cm, size=({shape.width/914400*2.54:.1f}x{shape.height/914400*2.54:.1f})cm | text: {text}")
