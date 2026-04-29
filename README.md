# 🌿 Détection et Classification Automatique des Maladies des Plantes

> Projet d'Ingénierie de l'Image — 2025-2026  
> Faculté des Sciences de Tunis — Université Tunis El Manar

Pipeline complète de vision par ordinateur pour la détection des maladies de la tomate, combinant **Machine Learning classique** et **Deep Learning (ResNet18)**, avec une interface web interactive.

---

## 📊 Résultats

| Approche | Modèle | Accuracy |
|----------|--------|----------|
| ML | SVM (RBF) | 81,3 % |
| ML | Random Forest | 80,2 % |
| ML | KNN (k=5) | 78,0 % |
| **DL** | **ResNet18 (fine-tuning)** | **98,2 %** |

---

## 🏗️ Architecture du Projet

```
Projet_ING/
├── Computer-Vision/
│   ├── notebooks/
│   │   ├── plant_disease_pipeline.ipynb              # Pipeline ML complet
│   │   └── plant_disease_pipeline_deep_learning.ipynb # Pipeline DL (ResNet18)
│   ├── src/                    # Modules Python (features, preprocessing)
│   ├── backend/                # API FastAPI
│   ├── frontend/               # Interface React 18 + Vite
│   ├── results/                # Plots, modèles, features CSV
│   ├── data/                   # Dataset PlantVillage (non versionné)
│   ├── PROJET_EXPLICATION.md   # Documentation technique détaillée
│   └── requirements.txt        # Dépendances Python
└── rapport/
    └── rapport.tex             # Rapport LaTeX académique
```

---

## 🔬 Pipeline

```
Image brute ──► Prétraitement ──► Segmentation ──► Extraction Features ──► Classification
                 (4 étapes)      (HSV + CLAHE)      (12 descripteurs)     (ML ou DL)
```

### 1. Prétraitement
Redimensionnement 256×256 → Flou Gaussien (5×5) → Niveaux de gris → Conversion HSV

### 2. Segmentation (HSV + CLAHE)
- Normalisation d'illumination via CLAHE sur le canal L de l'espace LAB
- Seuillage HSV sur le canal Hue [30°–95°]
- Nettoyage morphologique (ouverture + fermeture)

### 3. Extraction de caractéristiques (12 descripteurs)
- **Couleur HSV** (6) : h/s/v_mean + h/s/v_std
- **Texture GLCM** (3) : contrast, energy, homogeneity
- **Forme** (3) : area, perimeter, circularity

### 4. Classification
- **ML classique** : SVM, Random Forest, KNN sur les 12 features
- **Deep Learning** : ResNet18 pré-entraîné (ImageNet), fine-tuning layer4 + fc

---

## 🚀 Installation

### Prérequis
- Python 3.10+
- Node.js 18+ (pour le frontend)
- GPU CUDA (optionnel, pour le DL)

### Setup

```bash
# Cloner le projet
cd Projet_ING/Computer-Vision

# Créer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Installer les dépendances Python
pip install -r requirements.txt

# Installer les dépendances frontend
cd frontend
npm install
cd ..
```

### Dataset
Télécharger le dataset [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) et placer les images dans :
```
Computer-Vision/data/PlantVillage/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── ...
└── Tomato_healthy/
```

---

## ▶️ Utilisation

### Notebooks (entraînement)
```bash
# Ouvrir Jupyter
jupyter notebook notebooks/
```
- `plant_disease_pipeline.ipynb` — pipeline ML complet
- `plant_disease_pipeline_deep_learning.ipynb` — fine-tuning ResNet18

### Interface Web
```powershell
# Option 1 : Script automatique
.\run_servers.ps1

# Option 2 : Lancement manuel
# Terminal 1 — Backend
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Ouvrir http://localhost:5173 dans le navigateur.


## 🛠️ Technologies

| Outil | Usage |
|-------|-------|
| Python 3.10 / OpenCV 4.x | Traitement d'images |
| scikit-learn / scikit-image | ML classique, GLCM |
| PyTorch / torchvision | ResNet18 fine-tuning |
| FastAPI | Backend API REST |
| React 18 + Vite | Frontend web |






