with open('reports/poster_spd_ensi.html', 'r', encoding='utf-8') as f:
    text = f.read()

assert '73.40% [Best]' in text, "73.40% [Best] missing"
assert '+10.65% net (Domine toutes les m' in text, "Resolution text missing"
assert '63.33%' not in text, "Old 63.33% found in text"
assert 'margin: 1.5px 0 !important;' in text, "MathJax margin fix missing"
assert 'line-height: 1.24;' in text, "Line-height fix missing"

print("All assertions passed successfully! reports/poster_spd_ensi.html is clean, verified, and correctly configured.")
