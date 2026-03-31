import numpy as np


class MultinomialNaiveBayes:

    def __init__(self, alpha: float = 1.0):
        # alpha to wygładzanie Laplace'a, ktore zapobiega zerowym prawdopodobienstwom.
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.classes_ = None
        self.log_class_priors_ = None
        self.log_feature_probs_ = None

    def fit(self, x, y):
        y = np.asarray(y)
        classes, class_counts = np.unique(y, return_counts=True)

        n_classes = len(classes)
        n_features = x.shape[1]

        # log P(c): prior dla kazdej klasy.
        log_class_priors = np.log(class_counts / class_counts.sum())
        log_feature_probs = np.zeros((n_classes, n_features), dtype=np.float64)

        for class_index, class_label in enumerate(classes):
            # Bierzemy probki nalezace do danej klasy.
            x_class = x[y == class_label]
            # Zliczamy wystapienia kazdej cechy w tej klasie.
            feature_counts = np.asarray(x_class.sum(axis=0)).ravel().astype(np.float64)

            # Wygładzanie Laplace'a: dodajemy alpha do kazdej cechy.
            smoothed_feature_counts = feature_counts + self.alpha
            smoothed_total = smoothed_feature_counts.sum()

            # log P(w_i | c): znormalizowane prawdopodobienstwo cechy w klasie.
            log_feature_probs[class_index, :] = np.log(
                smoothed_feature_counts / smoothed_total
            )

        self.classes_ = classes
        self.log_class_priors_ = log_class_priors
        self.log_feature_probs_ = log_feature_probs
        return self

    def predict_log_proba(self, x):
        # Zwraca log-score dla klas: x * log P(w|c) + log P(c)
        if self.log_feature_probs_ is None or self.log_class_priors_ is None:
            raise ValueError("Model is not fitted yet")

        return x @ self.log_feature_probs_.T + self.log_class_priors_

    def predict(self, x):
        # Wybiera klase o najwiekszym log-score dla kazdej probki.
        log_proba = self.predict_log_proba(x)
        best_indices = np.argmax(log_proba, axis=1)
        return self.classes_[best_indices]
