with open('reports/poster_spd_ensi.html', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if any(tag in line for tag in ['<header', '<div class="card', '<h1>', '<h2>', '<table', '<h3', 'Bloc']):
            print(f"{idx+1}: {line.strip()[:100]}")
