# WikiArt

Projet de classification de styles picturaux sur un sous-ensemble reproductible de `huggan/wikiart`.

Le code et les explications sont regroupés dans **[wikiart.ipynb](wikiart.ipynb)** : consignes, récupération des données, inspection, découpage stratifié, MLP PyTorch, régularisation, ResNet-18 préentraîné et analyse des résultats. Les anciens scripts Python et `to_do.ipynb` sont remplacés par ce notebook.

## Installation et ouverture

Depuis la racine du dépôt :

```bash
python3 -m venv .venv-data
.venv-data/bin/python -m pip install -r requirements.txt
.venv-data/bin/python -m ipykernel install --user --name wikiart --display-name "Python (WikiArt)"
.venv-data/bin/jupyter lab wikiart.ipynb
```

Sélectionner le noyau **Python (WikiArt)**, puis exécuter les cellules dans l'ordre. Le notebook peut aussi être ouvert dans VS Code en sélectionnant l'interpréteur `.venv-data/bin/python`.

## Reproduction

La configuration se trouve au début du notebook :

- `DOWNLOAD_DATA = False` réutilise les images locales. Passer à `True` pour récupérer ou reprendre le téléchargement du sous-ensemble.
- `RUN_TRAINING = False` affiche les résultats MLP déjà enregistrés. Passer à `True` pour relancer les 40 époques et remplacer les artefacts de `MLP_CONFIG.output_dir`.
- `RUN_REGULARIZATION = False` relit les cinq essais enregistrés du MLP. Passer à `True` pour les relancer (référence, dropout, weight decay, combinaison et arrêt anticipé).
- `RUN_RESNET_HEAD = False` relit l’essai ResNet-18. Passer à `True` pour recalculer les caractéristiques ImageNet et entraîner la couche finale.
- Les autres paramètres, notamment les seeds, le batch, le taux d'apprentissage et les chemins, sont définis dans la même cellule. Après une modification, relancer le notebook depuis le début.

Sur un nouveau clone, les images sont absentes : commencer avec `DOWNLOAD_DATA = True`. Les fichiers de résultats CSV et JSON permettent de consulter l'expérience existante sans réentraîner. Pour sauvegarder une nouvelle expérience séparément, changer `output_dir` dans `MLP_CONFIG`.

Le sous-ensemble contient 300 images pour chacun des styles Impressionism, Baroque, Cubism, Ukiyo_e et Pop_Art. La révision du jeu est épinglée et la sélection utilise `seed=0`. Le téléchargement prévu concerne neuf shards, environ 4,2 Go. Les images, `metadata.csv` et `subset_info.json` restent dans `data/`, ignoré par Git. La fiche du jeu indique une licence inconnue ; ne pas publier les images.

L'inspection et le découpage sont exécutés dans le notebook. Les fichiers dans `manifests/` conservent le rapport de qualité et la partition stratifiée 70/15/15 avec `seed=42` : 210 images d'entraînement, 45 de validation et 45 de test par style. Les résultats du MLP sont dans `outputs/mlp_baseline/` ; les checkpoints restent ignorés par Git.

## État des expériences

Le MLP utilise des images RGB 64 × 64 normalisées dans [-1, 1] et une architecture dense 12 288 → 256 → 128 → 5 avec ReLU et `CrossEntropyLoss`. La première expérience de 40 époques sur MPS atteint 39,6 % d'exactitude de validation à l'époque 28, contre 98,6 % à l'entraînement. Le notebook affiche les courbes et l'écart de généralisation à partir des mesures enregistrées.

Le découpage présente une limite : 1 402 images sur 1 500 portent l'artiste `Unknown Artist` (93,5 %) et sept artistes nommés apparaissent dans plusieurs partitions. Les images de test sont inspectées pour la qualité des fichiers mais ne sont pas utilisées par le modèle à ce stade.

Les étapes 5 et 6 ajoutent cinq essais contrôlés du MLP et un ResNet-18 dont seule la tête est entraînée. Le corps préentraîné est gardé en mode évaluation et son empreinte est vérifiée avant et après l’expérience. Les caractéristiques de train et validation sont calculées une fois, sans charger les images de test. Le fine-tuning de `layer4` (étape 7) et la comparaison finale avec une régression logistique (étape 8) restent à réaliser. Le suivi des étapes et les consignes originales sont conservés dans le notebook. Les explications accompagnent chaque étape ; les fonctions de téléchargement et de contrôle sont repliées pour faciliter la lecture.

Les résultats des nouveaux essais sont dans `outputs/mlp_regularization/<essai>/` et `outputs/resnet18_head/` : historique CSV, résumé JSON et matrice de confusion de validation. Les checkpoints sont ignorés par Git. Les métriques d’entraînement de ces essais sont évaluées après chaque époque avec le dropout désactivé ; les mesures pendant les mises à jour sont conservées dans les colonnes `online_train_*`. Le meilleur checkpoint maximise l’exactitude de validation, puis minimise la perte en cas d’égalité. Les essais restent limités à une seed.

## Résultats des étapes 5 et 6

Les nouveaux essais utilisent la seed 42 et le même manifeste. Les métriques de train sont mesurées en mode évaluation au checkpoint retenu.

| Essai | Meilleure époque | Train | Validation |
|---|---:|---:|---:|
| MLP, référence relancée | 40 | 98,4 % | 39,6 % |
| MLP, dropout 0,3 | 27 | 91,0 % | 42,2 % |
| MLP, weight decay 0,0001 | 8 | 86,7 % | 38,2 % |
| MLP, dropout + weight decay | 3 | 49,3 % | 39,6 % |
| MLP, arrêt anticipé (patience 8) | 3 | 65,0 % | 37,8 % |
| ResNet-18, tête seule | 16 | 87,9 % | 75,6 % |

L’arrêt anticipé termine le MLP à l’époque 11 et le ResNet à l’époque 26. Le dropout est le candidat MLP retenu avec la validation, mais son gain reste modeste (six images sur 225) et doit être confirmé sur d’autres seeds. La tête du ResNet n’entraîne que 2 565 paramètres ; le corps ImageNet et ses statistiques de BatchNorm sont inchangés. Le score ResNet ne sépare pas les effets de l’architecture, du préentraînement et de la préparation des images. Ces résultats ne sont pas des scores de test et ne mesurent pas la latence CPU.
