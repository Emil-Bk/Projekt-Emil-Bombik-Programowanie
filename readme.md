# Titanic – Przewidywanie przeżycia pasażerów

**Baza Danych:** [Titanic – Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/c/titanic)  
**Zadanie:** Klasyfikacja binarna – czy pasażer przeżył katastrofę (`Survived`: 0/1)

---

## Opis danych

Zbiór zawiera dane 891 pasażerów Titanica. Celem jest przewidzenie, który pasażer przeżył.

| Kolumna | Opis |
|---------|------|
| `Pclass` | Klasa biletu (1/2/3) |
| `Sex` | Płeć |
| `Age` | Wiek |
| `SibSp` | Liczba rodzeństwa/małżonków na pokładzie |
| `Parch` | Liczba rodziców/dzieci na pokładzie |
| `Fare` | Cena biletu |
| `Embarked` | Port zaokrętowania |
| `Survived` | **Zmienna docelowa: 1 = przeżył, 0 = nie przeżył** |

---

## Cel analizy

Przewidywanie przeżycia pasażera (klasyfikacja binarna).  
Metryka oceny: **F1-score** (`classification_report`).

---

## Metodologia

1. Pobieranie danych (`!wget` wewnątrz notatnika)
2. Czyszczenie – uzupełnienie braków (`Age` → mediana, `Embarked` → moda), usunięcie zbędnych kolumn (`PassengerId`, `Name`, `Ticket`, `Cabin`)
3. Analiza korelacji – macierz korelacji, heatmapa
4. Stratyfikowany podział treningowy/testowy (80/20) z zachowaniem proporcji zmiennej docelowej
5. Pipeline – `ColumnTransformer` (`StandardScaler` dla cech numerycznych + `OneHotEncoder` dla cech kategorycznych)
6. Modele: Logistic Regression, Decision Tree, Random Forest (+ walidacja krzyżowa CV=5)
7. `GridSearchCV` – dobór hiperparametrów dla Random Forest
8. `RandomizedSearchCV` – rozszerzony, losowy dobór hiperparametrów dla Random Forest
9. Ocena na zbiorze testowym (`classification_report` oraz `f1_score`)
10. Porównanie modeli

---

## Wyniki

Poniższe zestawienie przedstawia wyniki metryki F1-score dla poszczególnych modeli na zbiorze testowym:

| Model | F1-score (test) |
|-------|----------------|
| Decision Tree | ~0.72 |
| Logistic Regression | ~0.76 |
| Random Forest (Baseline) | ~0.79 |
| Random Forest (Randomized Search) | ~0.79 |
| **Random Forest (Grid Search)** | **~0.80** |

**Najważniejsze wnioski:**
* **Najlepszy model:** Random Forest zoptymalizowany za pomocą **Grid Search** osiągnął najwyższy i najbardziej stabilny wynik F1-score na zbiorze testowym.
* **Overfitting:** Model drzewa decyzyjnego (`Decision Tree`) uległ silnemu przeuczeniu – osiągnął idealny wynik na zbiorze treningowym, ale znacznie słabszy na teście.
* **Prosty baseline:** Regresja logistyczna okazała się dobrym, bardzo szybkim i wysoce interpretowalnym modelem bazowym (F1 ~0.76).
