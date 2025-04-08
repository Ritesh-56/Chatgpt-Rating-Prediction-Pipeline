# 🧠 ChatGPT Rating Prediction Pipeline

This project presents a complete machine learning pipeline for predicting user ratings based on their written reviews of ChatGPT. The goal is to analyze the sentiment and content of reviews to classify them accurately into rating categories.

---

## 🚀 Project Overview

With the rise in usage of AI tools like ChatGPT, user reviews have become a valuable source of feedback. This project focuses on:

- **Preprocessing** user-written reviews.
- **Extracting features** using NLP techniques.
- **Training a classifier** to predict the rating given by the user.
- **Evaluating** the model's performance.

---

## 📂 Dataset

The dataset used in this project contains real user reviews of ChatGPT. Each entry includes:

- `review` - The written review text
- `rating` - The numeric rating given (used as the label)
- Additional metadata (`date`, `title`, etc.), which were dropped for simplicity.

---

## 🔧 Tech Stack & Libraries

- **Python**
- **Pandas** & **NumPy** – for data handling
- **Scikit-learn** – for model building and evaluation
- **NLP Tools**:
  - `TfidfVectorizer` – to convert text into numeric features
  - `PassiveAggressiveClassifier` – a fast linear classifier used for training

---

## 🧹 Pipeline Steps

1. **Load and Clean Data**: Remove unnecessary columns and handle missing values.
2. **Text Preprocessing**: Basic text cleaning and normalization.
3. **Vectorization**: Apply TF-IDF to convert text data into feature vectors.
4. **Train-Test Split**: Divide the data into training and testing sets.
5. **Model Training**: Train a Passive Aggressive Classifier on the training data.
6. **Evaluation**: Measure model accuracy on test data.

---

## 📈 Results

- The model was able to achieve reasonable accuracy on the test set, indicating that textual features from reviews are predictive of user ratings.
- Additional improvements can be made using advanced models such as logistic regression, SVMs, or transformers (e.g., BERT).

---

## 📌 Future Improvements

- Implement deep learning models (e.g., LSTM, BERT) for better performance.
- Add data visualizations and word cloud analysis.
- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Expand the dataset for broader evaluation.

---

## 🗂️ Project Structure

📁 ChatGPT-Rating-Prediction/ │ ├── 📄 Chatgpt_rating_prediction.ipynb # Main notebook with pipeline ├── 📄 chatgpt_reviews.csv # Dataset (not included here for privacy) ├── 📄 README.md # Project overview and instructions

yaml
Copy
Edit

---

## 🤝 Contributing

Contributions, ideas, and suggestions are welcome! Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Kaggle for hosting the dataset.
- Scikit-learn and the open-source community for machine learning tools.
