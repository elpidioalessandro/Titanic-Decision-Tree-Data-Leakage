# Titanic - A Decision Tree Classifier and a Data Leakage Case Study

A machine learning project predicting Titanic passenger survival with a decision tree. Built on the classic Titanic dataset, its real focus is methodological: how a hidden **data leakage** bug inflated the model's accuracy, how it was found, and how a proper scikit-learn pipeline fixed it, with a strong emphasis on honest interpretation of results.

The project was reviewed by a university instructor and refined through multiple feedback cycles, integrating more rigorous techniques at each iteration.

## The common thread

One principle runs through the whole project: the most accurate-looking number is not the most trustworthy one. A first version reached 80.3% accuracy, but that number was inflated by data leakage. The corrected model reports a lower, and honest, 77.1%. Methodological correctness matters more than a flattering metric.

## The problem

Predicting whether a passenger survived the Titanic disaster from their characteristics: sex, age, travel class, and port of embarkation. A binary classification problem on 891 passengers.

## Approach

* Exploratory analysis: missing-value inspection, distribution of `Age` (median vs mean, justifying median imputation), survival rates by sex (74% vs 19%) and class     (63% → 24%), supported by plots
* Preprocessing inside a `Pipeline` with `ColumnTransformer` (`SimpleImputer` + `OneHotEncoder`), fitted on the training folds only to prevent data leakage
* Model selection: decision tree depth chosen via 5-fold cross-validation (`GridSearchCV`), not a single fragile validation split
* Baseline: Logistic Regression, to contextualize the tree's gain
* Honest evaluation: accuracy, classification report and confusion matrix on a held-out test set

## The data leakage

The first version imputed missing values and encoded categories on the **entire dataset before splitting**, letting information from the test setleak into training. It scored 80.3% and looked correct.

Rebuilt with all preprocessing learned **only from the training data** inside a pipeline — making leakage impossible by design — the honestaccuracy dropped to 77.1%. The lower number is the real one.

## Results

| Metric | Value |
|---|---|
| Cross-validation accuracy (Decision Tree) | 81.6% |
| Baseline (Logistic Regression, CV) | 79.6% |
| **Test set accuracy** | **77.1%** |

The Decision Tree adds a real but modest margin over the linear baseline. The confusion matrix reveals the model's true limitation: it recognizes non-survivors well (93% recall) but misses nearly half of the survivors (51% recall), a bias toward the majority class that accuracy alone hides.

## Tech stack

Python · pandas · scikit-learn · matplotlib

## Structure

* `ElpidioAlessandroMorettiMLI0.ipynb` — full notebook with code, outputs, and interpretive commentary
* `titanic_sub.csv` — dataset

## How to read it

The notebook is meant to be read top to bottom: each section alternates commented code with text cells explaining the why behind the choices, not just the how.

Note: markdown explanations in the notebook are written in Italian. The code, outputs, and this README are self-explanatory for non-Italian readers.

## Dataset

The dataset (`titanic_sub.csv`) is an extract of the classic Titanic passenger dataset, provided as part of a machine learning course. It is used here for educational purposes only; all rights remain with their original authors.
