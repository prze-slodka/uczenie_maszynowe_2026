# Importy potrzebnych bibliotek
import random       # Do losowego wyboru próbek
import shutil       # Do usuwania katalogów (w razie błędu cache)
from pathlib import Path  # Do obsługi ścieżek plików


# Stałe określające maksymalną liczbę dokumentów do pobrania
# (ograniczenie rozmiaru danych dla szybszego działania)
MAX_TRAIN_DOCS = 300
MAX_TEST_DOCS = 100


# Funkcja pomocnicza: ogranicza rozmiar zbioru danych do max_docs dokumentów
# Używa losowego wyboru z stałym seed'em dla powtarzalności
def _sample_dataset(data, target, max_docs: int, seed: int = 42):
    if max_docs <= 0 or len(data) <= max_docs:
        return data, target

    # Losowy wybór indeksów
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    indices = indices[:max_docs]

    # Wybierz próbki na podstawie wylosowanych indeksów
    sampled_data = [data[i] for i in indices]
    sampled_target = [target[i] for i in indices]
    return sampled_data, sampled_target


# Główna funkcja demonstracyjna: klasyfikacja tekstów algorytmem Naive Bayes
# Używa zbioru danych 20 Newsgroups (grupy dyskusyjne z lat 90.)
def run_20newsgroups_demo() -> None:
    # Sprawdzenie i importowanie wymaganych bibliotek do ML
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.naive_bayes import MultinomialNB
    except ImportError:
        print("Brak zaleznosci. Zainstaluj: pip install scikit-learn")
        return

    # Wybrane kategorie z 20 Newsgroups do klasyfikacji
    categories = [
        "comp.graphics",      # Grafika komputerowa
        "sci.space",          # Kosmonautyka
        "rec.sport.hockey",   # Hokej
    ]

    # Konfiguracja pobierania danych z datasetu
    data_home = Path(".sklearn_data")
    fetch_args = {
        "categories": categories,
        "remove": ("headers", "footers", "quotes"),  # Usuń metadane
        "data_home": str(data_home),
    }

    # Pobieranie danych z obsługą błędów
    try:
        train = fetch_20newsgroups(subset="train", **fetch_args)
        test = fetch_20newsgroups(subset="test", **fetch_args)
    except Exception as exc:
        print(f"Blad podczas ladowania 20 Newsgroups: {type(exc).__name__}: {exc}")
        print("Probuję naprawić cache i pobrać dane ponownie...")
        # Jeśli jest błąd z cache'em, spróbuj go usunąć
        cache_dir = data_home / "20news_home"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        try:
            train = fetch_20newsgroups(subset="train", **fetch_args)
            test = fetch_20newsgroups(subset="test", **fetch_args)
        except Exception as retry_exc:
            print(f"Ponowna proba nieudana: {type(retry_exc).__name__}: {retry_exc}")
            print("Sprobuj ponownie pozniej lub sprawdz polaczenie z internetem.")
            return

    # Ograniczenie rozmiaru zbiorów treningowego i testowego
    train_data, train_target = _sample_dataset(train.data, train.target, MAX_TRAIN_DOCS)
    test_data, test_target = _sample_dataset(test.data, test.target, MAX_TEST_DOCS)

    print(
        f"Rozmiar danych po ograniczeniu: train={len(train_data)}, test={len(test_data)}"
    )

    # Przygotowanie danych: konwersja tekstów na macierz liczności słów
    # CountVectorizer: liczy wystąpienia słów, usuwa stopwords, ogranicza do 3000 najczęstszych
    vectorizer = CountVectorizer(stop_words="english", min_df=2, max_features=3000)
    x_train = vectorizer.fit_transform(train_data)
    x_test = vectorizer.transform(test_data)

    # Trenowanie modelu Naive Bayes (algorytm probabilistyczny dla klasyfikacji)
    model = MultinomialNB(alpha=1.0)  # alpha=1.0 to wygładzanie Laplace'a
    model.fit(x_train, train_target)
    y_pred = model.predict(x_test)

    # Ewaluacja modelu: dokładność na zbiorze testowym
    acc = accuracy_score(test_target, y_pred)
    print(f"20 Newsgroups accuracy: {acc:.4f}")


    # Test na przykładowych zdaniach: sprawdzenie, do której kategorii je model przypisze
    samples = [
        "The shuttle launch was delayed because of weather conditions.",
        "The team won in overtime after a great defensive play.",
        "3D rendering and image processing require fast GPUs.",
    ]
    sample_matrix = vectorizer.transform(samples)
    sample_preds = model.predict(sample_matrix)
    print("\nPrzykladowe predykcje:")
    for text, pred_idx in zip(samples, sample_preds):
        print(f"- '{text}' -> {train.target_names[pred_idx]}")


# Punkt wejścia programu
def main() -> None:
    run_20newsgroups_demo()


if __name__ == "__main__":
    main()