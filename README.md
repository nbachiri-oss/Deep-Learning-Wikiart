# WikiArt

Première étape du projet : récupérer un sous-ensemble équilibré de styles depuis `huggan/wikiart` avec `make_wikiart_subset.py`.

## Récupération

```bash
python -m venv .venv-data
.venv-data/bin/python -m pip install -r requirements.txt
.venv-data/bin/python make_wikiart_subset.py --list-classes --label style
.venv-data/bin/python make_wikiart_subset.py \
  --label style \
  --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
  --per-class 300 --size 128 --out data/wikiart_style5 --dry-run
HF_HUB_DOWNLOAD_TIMEOUT=1800 HF_XET_HIGH_PERFORMANCE=1 \
  .venv-data/bin/python make_wikiart_subset.py \
  --label style \
  --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
  --per-class 300 --size 128 --out data/wikiart_style5
```

Le script épingle la révision du jeu de données et la sélection aléatoire utilise `seed=0`. Le plan de 300 images par style sélectionne 9 shards, environ 4,2 Go à transférer. Le résultat contient les dossiers de classes, `metadata.csv` et `subset_info.json`.

Les images restent dans `data/`, ignoré par Git. La fiche Hugging Face indique une licence inconnue; ne pas publier les images.
