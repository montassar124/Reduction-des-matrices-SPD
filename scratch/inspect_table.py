import pptx

prs = pptx.Presentation(r"Modele_Poster_ENSI_60x100cm (1).pptx.pptx")
slide = prs.slides[0]
table_shape = None
for s in slide.shapes:
    if s.has_table:
        table_shape = s
        break

if table_shape:
    t = table_shape.table
    print(f"Table dimensions: {len(t.rows)} rows x {len(t.columns)} cols")
    for r_idx, row in enumerate(t.rows):
        row_txt = [cell.text.replace("\n", " ") for cell in row.cells]
        print(f"Row {r_idx}: {row_txt}")
