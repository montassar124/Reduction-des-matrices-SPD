import pptx

prs = pptx.Presentation(r"Modele_Poster_ENSI_60x100cm (1).pptx.pptx")
slide = prs.slides[0]
for s in slide.shapes:
    if s.has_table:
        t = s.table
        print(f"Table cols: {len(t.columns)}")
        for i, c in enumerate(t.columns):
            print(f"  Col {i}: width={c.width/914400*2.54:.2f}cm")
