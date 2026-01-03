# -*- coding: utf-8 -*-
"""
Preprocessing des données ESI
Prepare les datasets pour l'entrainement
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path='data/custom_training_data.csv'):
    """
    Charge le dataset

    Args:
        data_path: Chemin du fichier CSV

    Returns:
        DataFrame pandas
    """
    df = pd.read_csv(data_path)
    print(f"✅ {len(df)} exemples chargés")
    return df

def preprocess_data(df):
    """
    Nettoie et prepare les données

    Args:
        df: DataFrame brut

    Returns:
        DataFrame nettoyé
    """
    # Supprimer les valeurs manquantes
    df = df.dropna()

    # Vérifier la distribution des classes
    print("\n📊 Distribution des classes ESI:")
    for esi in range(1, 6):
        count = len(df[df['esi_label'] == esi])
        pct = (count / len(df)) * 100
        print(f"  ESI-{esi}: {count:3d} ({pct:5.1f}%)")

    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Separe en train/validation

    Args:
        df: DataFrame
        test_size: Proportion validation
        random_state: Seed

    Returns:
        train_df, val_df
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['esi_label']
    )

    print(f"\n✂️ Split:")
    print(f"  Train: {len(train_df)} exemples")
    print(f"  Val:   {len(val_df)} exemples")

    return train_df, val_df

if __name__ == "__main__":
    # Test preprocessing
    print("🔧 Test du preprocessing...\n")

    df = load_data()
    df = preprocess_data(df)
    train_df, val_df = split_data(df)

    print("\n✅ Preprocessing OK!")
