# 🎬 Movie Recommendation System — Collaborative Filtering with TensorFlow

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow) ![Keras](https://img.shields.io/badge/Keras-Adam-red?logo=keras) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview

This project implements a **Collaborative Filtering Recommendation System** from scratch using TensorFlow and Keras. The model learns user preferences and movie characteristics from historical rating data to predict how a user would rate movies they haven't seen yet — and surfaces the most relevant recommendations.

Collaborative filtering is the backbone of real-world recommendation engines used by Netflix, Amazon, and Spotify. This project demonstrates a production-ready implementation of the algorithm, complete with custom cost function design, vectorized optimization, and personalized rating prediction.

**Problem Statement:** Given a sparse matrix of user-movie ratings, learn latent feature representations for both users and movies such that their dot product approximates the true rating — enabling accurate predictions for unseen user-movie pairs.

---

## 🧠 Key Features

- **Custom Cost Function** — Hand-implemented collaborative filtering loss with L2 regularization to prevent overfitting
- **Vectorized TensorFlow Implementation** — Optimized matrix operations using `tf.linalg.matmul` for fast, scalable computation
- **Personalized Recommendations** — New user ratings are injected into the system and the model adapts to generate tailored suggestions
- **Rating Normalization** — Per-movie mean normalization ensures unbiased predictions across sparsely and densely rated movies
- **Adam Optimizer with GradientTape** — Custom training loop using TensorFlow's automatic differentiation for fine-grained control
- **Real-World Movie Dataset** — Trained and evaluated on a dataset of 4,778 movies and 443 users

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.x, Keras |
| Data Manipulation | NumPy, Pandas |
| Environment | Jupyter Notebook |
| Optimizer | Adam (learning rate: 0.1) |

---

## 📊 Dataset

**Source:** MovieLens-style small dataset (loaded via custom utility functions)

**Key Matrices:**

| Matrix | Shape | Description |
|---|---|---|
| `Y` | (4778, 443) | Rating matrix — actual rating values (1–5) |
| `R` | (4778, 443) | Binary mask — 1 if the movie was rated, 0 otherwise |
| `X` | (4778, 10) | Movie feature matrix (latent factors) |
| `W` | (443, 10) | User preference matrix (latent factors) |
| `b` | (1, 443) | Per-user bias terms |

**Scale:** 4,778 movies × 443 users × 10 latent features

**Preprocessing:**
- Per-movie mean normalization of ratings (`normalizeRatings`) to handle movies with few ratings
- New user ratings appended as an additional column to `Y` and `R` before training
- Rating matrix remains sparse — only rated entries participate in cost computation

---

## ⚙️ Methodology

### 1. Data Loading & Exploration
Pre-computed parameter matrices (`X`, `W`, `b`) and the ratings data (`Y`, `R`) are loaded using custom utility functions. Matrix dimensions are verified to ensure consistency across all tensors.

### 2. Cost Function Design
Two versions of the collaborative filtering cost function are implemented:

- **Loop-based version** (`cofi_cost_func`) — iterates over all user-movie pairs where `R[i,j] = 1`, useful for conceptual clarity
- **Vectorized version** (`cofi_cost_func_v`) — applies the mask `R` via element-wise multiplication after computing the full prediction matrix, enabling GPU-friendly computation

Both versions include **L2 regularization** on `W` and `X` to prevent the model from overfitting sparse ratings.

### 3. Injecting New User Ratings
A custom rating profile is created by assigning scores (1–5) to 13 known movies (e.g., Shrek, Inception, Lord of the Rings). These ratings are prepended as a new column in `Y` and `R`, allowing the model to learn this user's preferences during training.

### 4. Normalization
Ratings are normalized by subtracting each movie's mean rating (computed only over rated entries). This ensures predictions for new users start from a sensible baseline rather than zero.

### 5. Parameter Initialization
`X`, `W`, and `b` are randomly initialized using `tf.random.normal` with a fixed seed for reproducibility. All parameters are defined as `tf.Variable` to enable gradient tracking.

### 6. Training Loop
A custom training loop runs for 200 iterations using `tf.GradientTape`:
- Forward pass computes the vectorized cost
- Gradients are computed with respect to `X`, `W`, and `b`
- Adam optimizer applies parameter updates

---

## 🤖 Model Details

**Algorithm:** Matrix Factorization via Collaborative Filtering

**Objective Function:**

$$J = \frac{1}{2} \sum_{(i,j): R_{i,j}=1} \left( \mathbf{w}^{(j)} \cdot \mathbf{x}^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac{\lambda}{2} \left( \sum_{j} \|\mathbf{W}\|^2 + \sum_{i} \|\mathbf{X}\|^2 \right)$$

**Key Design Decisions:**

| Parameter | Value | Rationale |
|---|---|---|
| Latent Features (`num_features`) | 100 | Balanced expressiveness vs. overfitting |
| Regularization (`lambda_`) | 1 | Prevents memorization of sparse ratings |
| Learning Rate | 0.1 | Fast convergence with Adam optimizer |
| Iterations | 200 | Sufficient for loss plateau |
| Optimizer | Adam | Adaptive learning rates per parameter |

---

## 📈 Results & Evaluation

### Training Loss Convergence

| Iteration | Training Loss |
|---|---|
| 0 | 2,321,191 |
| 40 | 51,864 |
| 80 | 13,631 |
| 120 | 5,808 |
| 160 | 3,435 |
| 180 | 2,902 |

The model converges rapidly — loss drops by over **98%** within the first 100 iterations.

### Prediction Accuracy on Rated Movies

The model closely reproduces the original ratings for the new user:

| Movie | Original Rating | Predicted Rating |
|---|---|---|
| Shrek (2001) | 5.0 | **4.90** |
| Incredibles, The (2004) | 5.0 | **4.90** |
| Lord of the Rings: Return of the King (2003) | 5.0 | **4.89** |
| Harry Potter and the Chamber of Secrets (2002) | 5.0 | **4.88** |
| Amelie (2001) | 2.0 | **2.13** |
| Inception (2010) | 3.0 | **3.00** |
| Nothing to Declare (2010) | 1.0 | **1.26** |

### Top Recommendations (Unseen Movies)
After training, the top recommendations for the new user include:
1. My Sassy Girl (2001) — Predicted: **4.49**
2. Martin Lawrence Live: Runteldat (2002) — Predicted: **4.48**
3. Memento (2000) — Predicted: **4.48**

---

## ▶️ How to Run

### Prerequisites

```bash
Python >= 3.10
TensorFlow >= 2.x
NumPy
Pandas
Jupyter Notebook
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/movie-recommender-collab-filtering.git
cd movie-recommender-collab-filtering

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook
```

Open `Untitled.ipynb` and run all cells sequentially. Ensure the `utils.py` file and dataset files are present in the same directory.

### Expected Project Structure

```
movie-recommender/
├── Untitled.ipynb          # Main notebook
├── utils.py                # Dataset loading & normalization utilities
├── data/
│   ├── small_movies_Y.csv  # Ratings matrix
│   ├── small_movies_R.csv  # Mask matrix
│   └── movies.csv          # Movie title list
└── requirements.txt
```

---

## 📌 Future Improvements

- **Neural Collaborative Filtering (NCF)** — Replace the dot-product with a deep neural network to capture non-linear user-movie interactions
- **Cold Start Handling** — Incorporate content-based features (genre, cast, director) for movies or users with no prior ratings
- **Implicit Feedback** — Extend to handle watch history, clicks, and dwell time rather than just explicit star ratings
- **Hyperparameter Tuning** — Grid search over `lambda_`, `num_features`, and learning rate for optimal performance
- **REST API Deployment** — Wrap the trained model in a FastAPI or Flask service for real-time recommendation serving
- **Scalability** — Migrate to a distributed training setup (e.g., TensorFlow Distributed) for full-scale MovieLens datasets (20M+ ratings)
- **A/B Testing Framework** — Evaluate recommendation quality using online metrics (CTR, watch time) alongside offline RMSE

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or file an issue.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Built using TensorFlow and collaborative filtering — because great recommendations shouldn't be accidental.*
