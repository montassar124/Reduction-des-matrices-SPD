with open('reports/poster_spd_ensi.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if '<p' in l:
        print(f"Line {i+1}: {l.strip()[:100]}")
