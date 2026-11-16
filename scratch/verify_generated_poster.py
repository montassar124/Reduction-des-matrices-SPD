import pptx

prs = pptx.Presentation(r"reports/poster_spd_ensi_modele_60x100cm.pptx")
slide = prs.slides[0]

print(f"Total shapes: {len(slide.shapes)}")
images = [s for s in slide.shapes if s.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.PICTURE]
print(f"Total pictures: {len(images)}")
for i, img in enumerate(images):
    print(f"  Pic {i}: pos=({img.left/914400*2.54:.1f}, {img.top/914400*2.54:.1f})cm, size=({img.width/914400*2.54:.1f}x{img.height/914400*2.54:.1f})cm")

table_shape = [s for s in slide.shapes if s.has_table][0]
t = table_shape.table
print("\nTable rows:")
for r_idx, row in enumerate(t.rows):
    vals = [cell.text.replace("\n", " ") for cell in row.cells]
    print(f"  R{r_idx}: {vals}")
