# Toxic Comment Classification

A Natural Language Processing project comparing traditional machine learning and deep learning approaches for toxic comment detection.

The project uses a labelled Wikipedia comment dataset and evaluates **Naive Bayes, Support Vector Machine (SVM), BiLSTM and Text CNN** models using F1-score and ROC-AUC.

![Model Performance Comparison](results/figures/model_comparison.png)

## Project Overview

Online platforms need reliable methods for identifying harmful or toxic language at scale. This project explores multiple NLP modelling approaches, ranging from TF-IDF-based classical machine learning to neural sequence and convolutional models.

The aim was not only to train classifiers, but to compare how different modelling strategies perform on the same text-classification problem.

## Dataset

The analysis uses a labelled Wikipedia toxic-comment dataset containing more than **159,000 comments**.

The raw `train.csv` file is intentionally excluded from this repository because of its size. See [`data/README.md`](data/README.md) for dataset setup instructions.

## Models Evaluated

- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Bidirectional Long Short-Term Memory network (BiLSTM)
- Text Convolutional Neural Network (Text CNN)

## Methodology

The notebook covers:

- loading and inspecting labelled comment data;
- text preprocessing and preparation;
- TF-IDF feature extraction for classical machine-learning models;
- tokenisation and sequence preparation for neural models;
- Naive Bayes and SVM training;
- BiLSTM and Text CNN development using TensorFlow/Keras;
- model evaluation using F1-score and ROC-AUC;
- comparative visualisation of model performance.

## Results

| Model | F1-score | ROC-AUC |
|---|---:|---:|
| Naive Bayes | 0.614 | 0.945 |
| SVM | 0.778 | **0.964** |
| BiLSTM | **0.783** | 0.961 |
| Text CNN | 0.721 | 0.942 |

### Key Findings

- **BiLSTM achieved the highest F1-score (0.783)**, providing the strongest balance between precision and recall in this comparison.
- **SVM achieved the highest ROC-AUC (0.964)** and an F1-score very close to BiLSTM.
- The SVM result shows that a well-designed traditional NLP pipeline can remain highly competitive with more computationally expensive deep-learning models.
- Naive Bayes produced a strong ROC-AUC but a substantially lower F1-score.
- Text CNN performed reasonably well but did not outperform SVM or BiLSTM in this experiment.

## Repository Structure

```text
toxic-comment-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   └── toxic_comment_classification.ipynb
├── data/
│   └── README.md
└── results/
    └── figures/
        └── model_comparison.png
```

## Installation

```bash
git clone https://github.com/Ape-programmer/toxic-comment-classification.git
cd toxic-comment-classification
python -m venv .venv
pip install -r requirements.txt
```

Place the dataset according to `data/README.md`, start Jupyter from the repository root, and open `notebooks/toxic_comment_classification.ipynb`.

## Technologies

**Python • Pandas • NumPy • Scikit-learn • TensorFlow • Keras • NLP • Matplotlib • Jupyter Notebook**

## Limitations

- Toxic-comment datasets can contain significant class imbalance.
- Performance is dependent on preprocessing choices, decision threshold and dataset distribution.
- F1-score and ROC-AUC capture different aspects of classifier performance and should be interpreted together.
- Results from an academic dataset may not directly generalise to all real-world moderation environments.

## Future Improvements

Potential extensions include threshold optimisation, class-weight or resampling experiments, transformer-based models such as DistilBERT or BERT, model explainability, error analysis across toxicity categories and deployment as a moderation API or web application.

## Author

**Abiola Peace Emmanuel**  
MSc Artificial Intelligence & Data Science  
GitHub: **Ape-programmer**
