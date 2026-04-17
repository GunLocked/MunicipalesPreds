# Documentation — Pipeline de classification politique municipale

## Vue d'ensemble

```
generate_train.py          modele_tfidf.ipynb
      │                           │
      ▼                           ▼
corpus_train.json ──► TF-IDF ──► Régression logistique ──► Évaluation
(200 docs synthétiques)           (fitté sur train)         (51 docs réels)
                                                                  ▲
                                                          corpus.json (test)
```

---

## Partie 1 — Génération du corpus d'entraînement (`generate_train.py`)

### Pourquoi générer des données synthétiques ?

Le corpus réel ne contient que **51 documents**, ce qui est insuffisant pour entraîner un classifieur robuste. L'objectif est d'atteindre une répartition **80 % train / 20 % test**, soit 200 documents synthétiques pour 51 documents réels de test.

Le corpus réel est **intégralement réservé au test** : aucune de ses phrases n'est utilisée pour générer le train, ce qui évite tout overfitting ou fuite de données.

### Principe

Pour chaque combinaison **classe politique × thème**, un pool de phrases a été rédigé manuellement en s'inspirant du vocabulaire et des positions observés dans le corpus réel.

**5 classes × 6 thèmes × ~10 phrases = 300 phrases au total**

| Thème | Description |
|---|---|
| `securite` | Ordre public, police, délinquance |
| `ecologie` | Transition écologique, environnement |
| `economie` | Fiscalité, emploi, commerce |
| `mobilite` | Transports, voiture, vélo |
| `education` | Écoles, périscolaire, jeunesse |
| `sociale` | Logement, santé, solidarité |

### Marqueurs lexicaux distinctifs par classe

Chaque pool est conçu pour que le vocabulaire soit **politiquement discriminant** :

| Classe | Marqueurs clés |
|---|---|
| **EG** — Extrême gauche | *réquisition, municipalisation, coopérative, gratuité, émancipation, justice sociale, sans-abri, exilés* |
| **G** — Gauche | *transition énergétique, biodiversité, économie sociale, encadrement des loyers, médiation, circuits courts* |
| **C** — Centre | *attractivité, innovation, smart city, multimodal, partenariat public-privé, stabilité fiscale* |
| **D** — Droite | *ordre, gel des impôts, investisseurs privés, liberté de circulation, excellence, artisans* |
| **ED** — Extrême droite | *armement police, tolérance zéro, Français d'abord, préférence locale, communautarisme, laxisme* |

### Génération d'un document

```python
def gen_doc(label, doc_id):
    for theme in THEMES:
        pool = PHRASES[label][theme]     # pool des ~10 phrases du bon bord
        n = random.randint(3, 6)         # on tire entre 3 et 6 phrases
        selected = random.sample(pool, n)  # sans remise → pas de doublon dans un doc
        themes_text[theme] = ' '.join(selected)
```

Chaque document généré est donc une **combinaison aléatoire de phrases** issues du pool de sa classe. Avec `random.sample` (sans remise), un même document ne contient jamais deux fois la même phrase.

### Résultat

```
200 documents  ·  40 par classe  ·  6 thèmes renseignés chacun
```

**Limite connue :** avec 10 phrases par pool et 40 documents à générer, les mêmes phrases réapparaissent dans plusieurs documents (similarité cosine intra-classe ≈ 0.80). Le vocabulaire est cohérent mais peu varié. C'est la limite d'une approche synthétique sans vrai corpus d'entraînement.

---

## Partie 2 — Pipeline TF-IDF (`modele_tfidf.ipynb`)

### Étape 1 — Chargement et séparation train/test

```
corpus_train.json  →  X_train_raw, y_train  (200 docs)
corpus.json        →  X_test_raw,  y_test   (51 docs réels)
```

Pour chaque document, le texte est constitué de la **concaténation des 6 thèmes** :

```python
text = ' '.join(d['themes'].values())
```

Le `non_classe` est ignoré car il contient du bruit (noms propres, slogans, numéros de contact).

### Étape 2 — Vectorisation TF-IDF

**TF-IDF = Term Frequency × Inverse Document Frequency**

- **TF** : fréquence du mot dans le document
- **IDF** : log(nb total docs / nb docs contenant ce mot) → pénalise les mots trop communs

Un mot rare dans le corpus général mais fréquent dans les docs d'une classe obtient un score élevé → c'est exactement ce qu'on veut pour distinguer les bords politiques.

```python
vectorizer = TfidfVectorizer(
    max_features=2000,    # 2000 termes les plus informatifs
    min_df=2,             # ignore les hapax (mots dans 1 seul doc)
    ngram_range=(1, 2),   # unigrammes ET bigrammes ("police municipale" > "police")
    sublinear_tf=True,    # log(tf + 1) pour atténuer l'effet taille de doc
    stop_words=...,       # stopwords français + verbes conjugués + noms propres
)

X_train = vectorizer.fit_transform(X_train_raw)  # fit + transform sur train uniquement
X_test  = vectorizer.transform(X_test_raw)        # transform seulement (pas de fit)
```

> **Règle d'or :** le `fit` n'est jamais appliqué sur le test. Fitter le vectoriseur sur le test ferait fuiter de l'information (le modèle "verrait" le vocabulaire du test avant la prédiction).

**Résultat :** chaque document devient un vecteur de 2000 dimensions, où chaque dimension est le score TF-IDF d'un terme.

### Étape 3 — Régression logistique

```python
clf = LogisticRegression(
    C=1.0,                    # régularisation L2 (évite le surapprentissage)
    class_weight='balanced',  # compense le déséquilibre des classes (G: 8 docs vs C: 12)
    solver='lbfgs',           # adapté au multiclasse sur petit corpus
    multi_class='multinomial',
)

clf.fit(X_train, y_train)
```

Pour chaque document, le modèle calcule une **probabilité pour chacune des 5 classes** et prédit celle qui est maximale.

**Pourquoi la régression logistique sur TF-IDF ?**
- Les vecteurs TF-IDF sont creux (beaucoup de zéros) et haute dimension → la régression logistique est très efficace dans ce cas
- Les coefficients sont directement interprétables : un coefficient positif élevé pour un terme dans une classe signifie que ce terme est discriminant pour cette classe
- Elle fournit des probabilités calibrées, utiles pour les graphiques de distribution

### Étape 4 — Évaluation sur le corpus réel

```python
y_pred = clf.predict(X_test)   # prédiction sur les 51 docs réels
```

Les métriques sont calculées **uniquement** sur les 51 documents réels jamais vus pendant l'entraînement.

**Résultats obtenus (baseline) :**

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| Extrême gauche | 0.31 | 0.45 | 0.37 |
| Gauche | 0.33 | 0.25 | 0.29 |
| Centre | 0.40 | 0.17 | 0.24 |
| Droite | 0.50 | 0.20 | 0.29 |
| Extrême droite | 0.45 | 0.90 | 0.60 |
| **Accuracy globale** | | | **39 %** |

**Interprétation :**
- L'extrême droite est bien détectée (F1 = 0.60) car son lexique sécuritaire est très distinctif
- Centre et Droite sont mal discriminés (F1 ~ 0.25) car leurs vocabulaires se chevauchent
- 39 % est la **baseline honnête** — le modèle souffre d'un décalage lexical entre le train synthétique et les vrais tracts

### Résumé du pipeline complet

```
corpus.json (51 docs réels)
        │
        └──────────────────────────────────► TEST (jamais vu à l'entraînement)
                                                         │
generate_train.py                                        │
        │                                                │
        ▼                                                ▼
corpus_train.json (200 docs synthétiques)         Métriques finales
        │                                         accuracy = 39%
        ▼
TfidfVectorizer.fit_transform()
        │
        ▼
LogisticRegression.fit()
        │
        ▼
TfidfVectorizer.transform() ◄── corpus.json (test)
        │
        ▼
LogisticRegression.predict()
        │
        ▼
classification_report / confusion_matrix / probabilités
```

---

## Pistes d'amélioration

| Levier | Impact attendu | Complexité |
|---|---|---|
| Enrichir le corpus réel (plus de vrais tracts) | Fort | Collecte manuelle |
| Augmenter les pools de phrases synthétiques | Moyen | Rédaction |
| Tester SVM linéaire (`LinearSVC`) | Faible à moyen | Faible |
| Ajouter le `non_classe` avec filtrage des bruits | Moyen | Faible |
| Passer à un modèle de langue (CamemBERT) | Fort | Élevée |
