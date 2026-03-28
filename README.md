# uczenie_maszynowe_2026
Przedmiot analiza danych i uczenie maszynowe - skrypty własne

Skrypt uruchamia klasyfikację Naive Bayes na zbiorze 20 Newsgroups.

## Uruchamianie

Najpierw zainstaluj zależność:

```bash
pip install scikit-learn
```

Następnie uruchom:

```bash
python naive_bayes.py
```

Skrypt domyślnie działa w trybie szybszym (ogranicza liczbę dokumentów), żeby nie czekać bardzo długo.

Limity są ustawione bezpośrednio w kodzie: `MAX_TRAIN_DOCS` i `MAX_TEST_DOCS` w pliku `naive_bayes.py`.

Skrypt pobierze dane 20 Newsgroups, wytrenuje klasyfikator `MultinomialNB`, wypisze `accuracy`, raport klasyfikacji oraz kilka przykładowych predykcji.
