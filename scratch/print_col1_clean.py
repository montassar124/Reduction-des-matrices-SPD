with open('reports/poster_spd_ensi.html', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(535, min(len(lines), 625)):
    # strip data:image
    l = lines[i]
    if 'data:image' in l:
        print(f"Line {i+1}: <DATA IMAGE TRUNCATED>")
    else:
        print(f"Line {i+1}: {l}", end="")
