with open('scripts/build_all_posters.py', 'r', encoding='utf-8') as f:
    text = f.read()

assert '73.40% [Best]' in text, "73.40% [Best] missing in build_all_posters.py"
assert '+10.65% net (Domine toutes les méthodes)' in text, "Resolution text missing in build_all_posters.py"
assert '63.33%' not in text, "Old 63.33% found in build_all_posters.py"

print("All assertions passed for build_all_posters.py as well!")
