with open('reports/poster_spd_ensi.html', 'r', encoding='utf-8') as f:
    text = f.read()

css = text[text.find('<style>'):text.find('</style>')]
lines = css.split('\n')
for i, l in enumerate(lines):
    if any(k in l for k in ['p {', 'p,', 'line-height', 'margin', 'gap:', 'padding']):
        print(f"{i+1}: {l.strip()}")
