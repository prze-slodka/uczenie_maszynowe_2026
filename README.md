# Machine Learning Projects

Repozytorium zawiera dwa projekty związane z klasyfikacją tekstu przy użyciu różnych metod uczenia maszynowego:

1. **Multinomial Naive Bayes** – własna implementacja klasyfikatora Bayesa.
2. **Neural Network (Linear + Conv1D)** – model sieci neuronowej zbudowany w PyTorch.

Oba projekty wykorzystują zbiór danych **20 Newsgroups** do klasyfikacji dokumentów tekstowych.

---

## Project Structure

```text
.
├── naive_bayes.py      # Demonstracja klasyfikacji tekstu metodą Naive Bayes
├── skrypt.py           # Własna implementacja Multinomial Naive Bayes
├── linear_conv1d.py    # Sieć neuronowa Linear + Conv1D w PyTorch
├── .gitignore
└── .sklearn_data/      # Pobrane dane (ignorowane przez Git)
```

---

# 1. Multinomial Naive Bayes

## Overview

Projekt przedstawia własną implementację algorytmu **Multinomial Naive Bayes** bez użycia gotowych modeli ze scikit-learn.

Model został zaimplementowany od podstaw przy użyciu biblioteki NumPy.

## Features

* własna implementacja klasyfikatora Naive Bayes,
* wygładzanie Laplace'a (Laplace Smoothing),
* obliczanie prawdopodobieństw w przestrzeni logarytmicznej,
* klasyfikacja tekstów z wykorzystaniem reprezentacji Bag of Words,
* ewaluacja dokładności modelu.

## Dataset

Wykorzystano wybrane kategorie ze zbioru danych 20 Newsgroups:

* `comp.graphics`
* `sci.space`
* `rec.sport.hockey`

Teksty są przekształcane do postaci liczności słów za pomocą `CountVectorizer`.

## Workflow

1. Pobranie danych.
2. Ograniczenie liczby dokumentów.
3. Wektoryzacja tekstów.
4. Trenowanie modelu Naive Bayes.
5. Predykcja na zbiorze testowym.
6. Obliczenie accuracy.
7. Test na przykładowych zdaniach.

## Technologies

* Python
* NumPy
* scikit-learn

---

# 2. Neural Network: Linear + Conv1D

## Overview

Projekt przedstawia prostą sieć neuronową do klasyfikacji tekstu zbudowaną w PyTorch.

Model wykorzystuje połączenie warstw:

* Linear
* ReLU
* Conv1D
* Linear (output layer)

## Architecture

```text
Input (TF-IDF, 5000 features)
        │
        ▼
Linear(5000 → 128)
        │
      ReLU
        │
        ▼
Conv1D(1 → 32)
        │
      ReLU
        │
        ▼
Flatten
        │
        ▼
Linear(4096 → 20)
        │
        ▼
Output Classes
```

## Dataset

Wykorzystano pełny zbiór **20 Newsgroups**.

Teksty są konwertowane do reprezentacji numerycznej przy pomocy:

```python
TfidfVectorizer(max_features=5000)
```

## Training

Parametry treningu:

* Optimizer: Adam
* Learning Rate: 0.001
* Loss Function: CrossEntropyLoss
* Epochs: 50

## Workflow

1. Pobranie danych.
2. Konwersja tekstów do TF-IDF.
3. Podział na train/test.
4. Konwersja do tensorów PyTorch.
5. Trenowanie modelu.
6. Monitorowanie wartości funkcji straty.

## Technologies

* Python
* PyTorch
* scikit-learn

---

# Installation

```bash
git clone <repository-url>
cd <repository-name>

pip install numpy torch scikit-learn
```

---

# Running

### Naive Bayes

```bash
python naive_bayes.py
```

### Neural Network

```bash
python linear_conv1d.py
```

---

# Educational Purpose

Projekty zostały stworzone w celach edukacyjnych, aby porównać klasyczne podejście probabilistyczne (Naive Bayes) z podejściem opartym na sieciach neuronowych do klasyfikacji tekstu.

## Contributors

* **prze-slodka** – Project Owner
* **zbielec (https://github.com/zbielec)** – Co-author

