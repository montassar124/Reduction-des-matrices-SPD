with open('reports/poster_spd_ensi.html', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(514, min(len(lines), 850)):
    print(lines[i], end="")
