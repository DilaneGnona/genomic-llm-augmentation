# experiment_snp

Quickstart

1. Créer un environnement Python et installer les dépendances
   - pip install -r requirements/base.txt
   - pip install -r requirements/ml.txt (optionnel)
   - pip install -r requirements/dl.txt (optionnel)

2. Configurer le projet
   - Copier .env.example vers .env et remplir les clés
   - Éditer config.yaml pour définir DATASET et chemins

3. Vérifier les données prétraitées
   - python scripts/verify_preprocessed_format.py pepper
   - python scripts/verify_preprocessed_format.py ipk_out_raw

4. Lancer un entraînement
   - python scripts/unified_modeling_pipeline.py --dataset pepper
