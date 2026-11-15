with open('reports/poster_spd_ensi.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Total lines:", len(lines))
for i in range(514, 825):
    l = lines[i].strip()
    if l.startswith('<!--') or 'class="bar-header"' in l or 'class="pill-header"' in l:
        print(f"Line {i+1}: {l}")
