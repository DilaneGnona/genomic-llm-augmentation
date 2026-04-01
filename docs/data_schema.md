# Schéma des données prétraitées

## X.csv (caractéristiques génétiques)
- Colonnes:
  - Sample_ID (string)
  - SNPs (une colonne par VAR_ID issu de variant_manifest.csv)
- Types:
  - Sample_ID: texte
  - SNPs: numériques
    - Origine en encodage additif 0,1,2 (nombre d'allèles alternatifs)
    - Selon le dataset, les valeurs peuvent être standardisées (centrées/réduites) après imputation (ex.: pepper)
    - Dans ipk_out_raw, X.csv est généralement en 0/1/2 après imputation au mode
- Format: CSV, séparateur virgule, encodage UTF-8

## y.csv (phénotypes)
- Colonnes typiques:
  - Sample_ID (string)
  - YR_LS (float, continue)
  - YR_precision (float, optionnelle)
  - Yield_BV (float, continue, cible pour pepper)
- Notes:
  - Les colonnes cibles par dataset sont définies dans config.yaml (section TARGET_COLUMNS)
  - Les valeurs non numériques sont coercisées et les lignes NaN sont retirées pour la cible
- Format: CSV, séparateur virgule, encodage UTF-8

## pca_covariates.csv (structure de population)
- Colonnes:
  - Sample_ID (string)
  - PC1..PCN (float, N défini par PCA_COMPONENTS dans config.yaml, typiquement 5)
- Format: CSV, séparateur virgule, encodage UTF-8

## variant_manifest.csv (métadonnées variants)
- Colonnes:
  - VAR_ID (string)
  - CHR (string)
  - POS (int)
  - REF (string)
  - ALT (string)
- Format: CSV, séparateur virgule, encodage UTF-8

## sample_map.csv (statuts échantillons)
- Colonnes:
  - Sample_ID (string)
  - Status (string: kept/removed)
  - Failure_Reason (string, optionnelle selon pipeline)
- Format: CSV, séparateur virgule, encodage UTF-8

## Contraintes d’alignement
- Les fichiers X.csv, y.csv et pca_covariates.csv doivent partager le même ensemble de Sample_ID.
- Les Sample_ID sont toujours traités en chaînes de caractères pour éviter les collisions numériques.

## Validation
- Utiliser scripts/verify_preprocessed_format.py pour vérifier:
  - Présence des fichiers requis
  - Numéricité des colonnes de features
  - Alignement des Sample_ID
  - Existence d’une colonne cible continue adaptée au dataset
