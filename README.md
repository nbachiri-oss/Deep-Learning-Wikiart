# WikiArt

Projet de classification de styles picturaux sur un sous-ensemble reproductible de `huggan/wikiart`.

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

## Inspection et découpage

Après le téléchargement, inspecter les images et écrire le manifeste des découpages :

```bash
.venv-data/bin/python prepare_wikiart_splits.py \
  --data data/wikiart_style5 \
  --output-dir manifests \
  --seed 42
```

Le contrôle vérifie les chemins, étiquettes, fichiers JPEG, dimensions, modes couleur, fichiers corrompus et doublons exacts. Le découpage est stratifié par style en 70/15/15, soit 210 images d'entraînement, 45 de validation et 45 de test par style. Le manifeste utilise des chemins relatifs; le rapport d'inspection conserve les contrôles et les comptes.

Dans cet échantillon, 1 402 images sur 1 500 portent l'artiste `Unknown Artist` (93,5 %). Le découpage ne peut donc pas être stratifié par artiste; sept artistes nommés apparaissent dans plusieurs partitions. Il faudra garder cette limite à l'esprit lors de l'interprétation des résultats de test.
