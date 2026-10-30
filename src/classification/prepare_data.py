from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, UnidentifiedImageError

from src.common.config import (
    CLASSIFICATION_RAW_DIR,
    CLASSIFICATION_ARTIFACTS_DIR,
    ALLOWED_EXTENSIONS
)


def scan_classification_dataset():
    """
    Analyse les images du dataset de classification.

    Cette fonction :
    - identifie les classes ;
    - vérifie les extensions ;
    - détecte les images illisibles ;
    - construit un tableau des images valides.
    """

    records = []

    invalid_files = []

    # Vérifier que le dossier du dataset existe.
    if not CLASSIFICATION_RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {CLASSIFICATION_RAW_DIR}"
        )

    # Créer le dossier des résultats.
    CLASSIFICATION_ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Récupérer les dossiers correspondant aux classes.
    class_directories = sorted(
        path
        for path in CLASSIFICATION_RAW_DIR.iterdir()
        if path.is_dir()
    )

    if not class_directories:
        raise ValueError(
            "Aucun dossier de classe trouvé dans le dataset."
        )

    print("Classes détectées :")

    for class_directory in class_directories:
        print(f"- {class_directory.name}")

        # Parcourir les fichiers et sous-dossiers de la classe.
        for image_path in class_directory.rglob("*"):

            # Ignorer les dossiers.
            if not image_path.is_file():
                continue

            # Vérifier l'extension.
            if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:

                invalid_files.append({
                    "path": str(image_path),
                    "reason": "Extension non autorisée"
                })

                continue

            try:

                # Vérifier que le fichier contient une image lisible.
                with Image.open(image_path) as image:
                    image.verify()

                # Rouvrir l'image pour récupérer ses dimensions.
                with Image.open(image_path) as image:
                    width, height = image.size

                    image_format = image.format

                    image.convert("RGB").load()

                # Écarter les images dont les dimensions sont invalides.
                if width <= 0 or height <= 0:
                    raise ValueError(
                        "Dimensions d'image invalides."
                    )

                # Enregistrer les informations de l'image.
                records.append({
                    "path": str(image_path.resolve()),
                    "filename": image_path.name,
                    "label": class_directory.name,
                    "extension": image_path.suffix.lower(),
                    "format": image_format,
                    "width": width,
                    "height": height
                })

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ) as error:

                invalid_files.append({
                    "path": str(image_path),
                    "reason": str(error)
                })

    df = pd.DataFrame(records)

    # Enregistrer les fichiers invalides ou refusés.
    invalid_df = pd.DataFrame(
        invalid_files,
        columns=["path", "reason"]
    )

    invalid_df.to_csv(
        CLASSIFICATION_ARTIFACTS_DIR / "invalid_files.csv",
        index=False,
        encoding="utf-8"
    )

    if df.empty:
        raise ValueError(
            "Aucune image valide trouvée dans le dataset."
        )

    df.to_csv(
        CLASSIFICATION_ARTIFACTS_DIR / "image_inventory.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nAnalyse terminée.")
    print(f"Images valides : {len(df)}")
    print(f"Fichiers refusés ou illisibles : {len(invalid_df)}")

    print("\nNombre d'images par classe :")
    print(df["label"].value_counts().sort_index())

    print("\nRépartition des extensions :")
    print(df["extension"].value_counts())

    return df


if __name__ == "__main__":

    df = scan_classification_dataset()