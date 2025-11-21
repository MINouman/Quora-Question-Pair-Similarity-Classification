# Quora Question Pair Similarity Classification

A Natural Language Processing (NLP) project designed to identify duplicate questions on Quora. By detecting semantically equivalent questions, this system helps improve user experience by reducing redundancy and grouping similar queries together.

## 📌 Project Overview

The goal of this project is to classify pairs of questions as either "Duplicate" (semantically similar) or "Not Duplicate" based on their text content. The project progresses from traditional machine learning baselines to an advanced Deep Learning approach utilizing Transfer Learning with **Sentence Transformers** and **Artificial Neural Networks (ANN)**.

## 📂 Dataset

*   **Source:** Quora Question Pairs Dataset.
*   **Data Points:** ~404,000 question pairs.
*   **Target:** `is_duplicate` (Binary Classification: 0 = Non-Duplicate, 1 = Duplicate).
*   **Characteristics:** The dataset is imbalanced, with significantly more non-duplicate pairs than duplicate ones.

## 🛠 Tech Stack

*   **Language:** Python 3.12+
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **NLP Preprocessing:** SpaCy, Scikit-Learn (TF-IDF)
*   **Deep Learning:** TensorFlow, Keras
*   **Transfer Learning:** Hugging Face `sentence-transformers`
*   **Optimization:** Keras Tuner

## ⚙️ Methodology

### 1. Exploratory Data Analysis (EDA)
*   Analyzed class imbalance (Duplicates vs. Non-Duplicates).
*   Engineered features: Character length, word count differences.
*   Visualized distributions to understand the semantic density of questions.

### 2. Text Preprocessing
*   **Cleaning:** Lowercasing, removal of special characters.
*   **Lemmatization:** Used `spaCy` to reduce words to their root forms (e.g., "running" -> "run") to normalize the vocabulary.

### 3. Feature Engineering
Two distinct approaches were taken:
*   **Approach A (Statistical):** TF-IDF Vectorization (Top 10k features) + Cosine Similarity features.
*   **Approach B (Semantic):** Generating 384-dimensional dense vector embeddings using the pre-trained BERT model `all-MiniLM-L6-v2`.

### 4. Modeling Strategy
*   **Baselines:**
    *   Logistic Regression.
    *   Linear Support Vector Machine (SVM).
*   **Advanced Architecture:**
    *   **Input:** Absolute difference between Q1 and Q2 embeddings (`|u - v|`).
    *   **Hidden Layers:** Deep Dense layers with ReLU activation.
    *   **Regularization:** Batch Normalization and Dropout to prevent overfitting.
    *   **Output:** Sigmoid activation for binary probability.

### 5. Hyperparameter Tuning
Utilized **Keras Tuner** (`RandomSearch`) to optimize:
*   Number of neurons in hidden layers.
*   Dropout rates.
*   Learning rates for the Adam optimizer.

## 📊 Results

The Deep Learning model significantly outperformed the statistical baselines by capturing semantic context rather than just keyword overlap.

| Model | Feature Method | Accuracy |
| :--- | :--- | :--- |
| Logistic Regression | TF-IDF + Cosine Sim | ~76.0% |
| Linear SVM | TF-IDF + Cosine Sim | ~77.0% |
| **Sentence Transformer + ANN** | **BERT Embeddings** | **81.17%** |

*Note: The final ANN model achieved a robust F1-Score of 0.74 for the minority "Duplicate" class.*

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/quora-similarity.git
    cd quora-similarity
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras-tuner sentence-transformers spacy
    python -m spacy download en_core_web_sm
    ```

3.  **Run the Notebook:**
    Open `Question Pair Similarity Classification.ipynb` in Jupyter Lab or Google Colab (GPU recommended for the Sentence Transformer encoding step).

## 📈 Future Improvements

*   **Model Architecture:** Implement a true Siamese Network architecture where weights are shared during the embedding training phase rather than pre-computing embeddings.
*   **Data Augmentation:** Use back-translation or synonym replacement to balance the minority class.
*   **Ensembling:** Combine the predictions of the SVM and the ANN to potentially squeeze out higher accuracy.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
