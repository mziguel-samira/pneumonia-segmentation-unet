# Détection et segmentation de la pneumonie sur radiographies thoraciques
**Deep Learning · U-Net · ResNet-18 · Transfer Learning · PyTorch · Imagerie médicale · DICOM**

Segmentation d'opacités pulmonaires sur le dataset **RSNA Pneumonia Detection** avec une architecture **U-Net**.
Le projet compare un U-Net léger entraîné *from scratch* (**SimpleUNet**) à un **U-Net à encodeur ResNet-18**
(initialisation aléatoire vs pré-entraînement ImageNet), et évalue l'apport du transfert d'apprentissage.

> Projet réalisé dans le cadre du module **Deep Learning** — Master IA & TE, Faculté des sciences appliquées de Nador (2025-2026),
> encadré par Pr. Jamal Boussouf.

**Mots-clés :** Python, PyTorch, Deep Learning, U-Net, ResNet-18, Transfer Learning, Imagerie médicale, DICOM

---

## Problématique

Le dataset RSNA fournit des **boîtes englobantes** (bounding boxes) et non des masques pixel par pixel, car l'annotation
fine est coûteuse et chronophage. L'idée est de **rasteriser ces boîtes en pseudo-masques binaires** et de s'en servir
comme cibles d'entraînement pour un modèle de segmentation.

## Dataset RSNA Pneumonia Detection

Dataset fourni par la **Radiological Society of North America (RSNA)** en collaboration avec le NIH et Kaggle.

- **26 684 radiographies thoraciques** au format DICOM (1024 × 1024 pixels)
- **6 012 examens positifs** (pneumonie confirmée par des radiologues)
- **9 555 bounding boxes** annotées manuellement
- **20 672 examens négatifs**

**Format du CSV** (`stage_2_train_labels.csv`) :

| Colonne | Description |
|---|---|
| `patientId` | Identifiant unique du patient |
| `x`, `y` | Coin supérieur gauche de la boîte |
| `width`, `height` | Dimensions de la boîte |
| `Target` | 1 = pneumonie présente, 0 = sain |

**Pourquoi c'est difficile :** annotations en boîtes (pas en masques), opacités diffuses et variables, fort déséquilibre (6012 positifs vs 20672 négatifs).

**Notre approche :** rasterisation des boîtes → pseudo-masques binaires → fusion par union logique par patient.

## Données

| | Valeur |
|---|---|
| Examens radiographiques (RSNA) | 26 684 |
| Examens positifs (Target = 1) | 6 012 |
| Boîtes annotées | 9 555 |
| Train / Val / Test (niveau patient, seed = 42) | 4 208 / 902 / 902 |

Seuls les examens **positifs** sont utilisés pour l'entraînement et l'évaluation, car les examens négatifs n'ont pas de lésion à segmenter (pas de masque cible possible).

**Prétraitement** (`pydicom`, `OpenCV`) :
1. Lecture du DICOM et application de la **VOI LUT**
2. Correction de la polarité (`MONOCHROME1` → inversion, `MONOCHROME2` → conservation)
3. Normalisation dans [0, 1]
4. Redimensionnement en **256 × 256**
5. Union logique des boîtes d'un patient → pseudo-masque binaire

> Les données ne sont pas incluses dans ce dépôt (licence RSNA / Kaggle).
> 📥 Disponible ici : [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

## Architectures

| Modèle | Initialisation | Paramètres |
|---|---|---|
| SimpleUNet (encodeur 16→32→64, bottleneck 128) | Aléatoire (*from scratch*) | 483 153 |
| SimpleUNet | Fine-tuning depuis le checkpoint *from scratch* | 483 153 |
| U-Net ResNet-18 (`segmentation_models_pytorch`) | Aléatoire | 14 321 937 |
| U-Net ResNet-18 (`segmentation_models_pytorch`) | Poids **ImageNet** | 14 321 937 |

## Entraînement

- **Perte :** 0,5 × BCE + 0,5 × Soft Dice
- **Optimiseur :** AdamW (weight decay 5e-4), gradient clipping (norme max 1,0)
- **Learning rate :** 3e-4 (from scratch), 1e-4 (fine-tuning), `ReduceLROnPlateau` sur l'IoU de validation
- **Batch size :** 16, jusqu'à 10 epochs, **early stopping** (patience 4), sauvegarde du meilleur modèle selon l'IoU de validation
- **Métriques :** IoU (calculé sur les images positives) et Dice sur le jeu de test, seuil 0,5

## Résultats (jeu de test, 902 patients positifs)

| Modèle | Initialisation | Best Val IoU | **Test IoU** | **Test Dice** |
|---|---|---|---|---|
| SimpleUNet | From scratch | 0,4622 | 0,4420 | 0,5857 |
| SimpleUNet | Fine-tuning (checkpoint) | 0,4704 | 0,4551 | 0,5985 |
| ResNet-18 U-Net | From scratch | 0,4829 | 0,4751 | 0,6169 |
| ResNet-18 U-Net | Fine-tuning ImageNet | 0,4981 | **0,4787** | **0,6233** |

Le U-Net ResNet-18 pré-entraîné sur ImageNet obtient les meilleurs scores. Le SimpleUNet, avec environ **3 %** des paramètres,
atteint plus de **92 %** de son IoU, ce qui en fait une alternative très légère.

### Limites

- Les cibles sont des **pseudo-masques rectangulaires** issus de boîtes englobantes : l'IoU mesure donc l'accord avec une
  approximation grossière des lésions, pas avec une vérité terrain au pixel près.
- L'évaluation porte uniquement sur les **cas positifs** ; les cas négatifs (20 672 examens) ne sont pas encore pris en compte.
- Une seule partition train/val/test (seed 42), sans validation croisée.

### Perspectives

- Évaluer sur l'ensemble du dataset, y compris les cas négatifs
- Tester d'autres encodeurs ou des poids pré-entraînés sur données médicales
- Ajouter de la data augmentation et une analyse des erreurs par taille de lésion

## Reproduire

Le notebook a été exécuté sur **Kaggle** (GPU) avec le dataset de la compétition
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).

1. Ouvrir `segmentation.ipynb` sur Kaggle et ajouter la compétition comme source de données
   (le chemin attendu est `/kaggle/input/competitions/rsna-pneumonia-detection-challenge`).
2. Pour une exécution locale, installer les dépendances puis adapter la variable `DATA_DIR` de la première cellule :

```bash
git clone https://github.com/mziguel-samira/pneumonia-segmentation-unet.git
cd pneumonia-segmentation-unet
pip install -r requirements.txt
jupyter notebook segmentation.ipynb