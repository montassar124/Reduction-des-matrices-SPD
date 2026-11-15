with open('reports/poster_spd_ensi.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(596, 747):
    line = lines[i]
    if line.strip().startswith('data:image'):
        print(f"{i+1}: [IMAGE BASE64 TRUNCATED]")
    else:
        print(f"{i+1}: {line[:120]}", end='')
