# Sentiment Analysis on X (Twitter) Data Using NLP and Machine Learning

## 📖 Overview
This project analyzes the sentiments expressed in tweets on X (formerly Twitter), classifying them into **positive**, **negative**, or **neutral** categories. The project utilizes **natural language processing (NLP)** techniques and a **Support Vector Machine (SVM)** model to gain insights into public sentiment trends.

---

## 💡 Features
- **Data Preprocessing**:
  - Tokenization, punctuation removal, and stopword filtering.
  - Handles mentions, special characters, and noisy text.
- **Sentiment Analysis**:
  - Sentiment classification using VADER for initial labeling.
  - SVM model trained on TF-IDF features for enhanced accuracy.
- **Visualization**:
  - Bar plots to display sentiment distribution.

---

## 🚀 Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - **Natural Language Toolkit (NLTK)**: For text processing and sentiment scoring.
  - **Scikit-learn**: For machine learning and feature extraction.
  - **Matplotlib**: For visualizing sentiment trends.
- **Tools**: Google Colab, Pandas, TF-IDF Vectorizer

---

## 📂 Project Structure



---

## 📝 Usage Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-x.git
   cd sentiment-analysis-x

 2. Install Dependencies: Ensure you have Python 3.x installed. Then, install the required libraries:

     ```bash
     pip install -r requirements.txt

Run the Notebook: Open sentiment_analysis.ipynb in Jupyter Notebook or Google Colab to

Preprocess tweets
Train the sentiment classifier
Evaluate results
Make Predictions: Use the saved svm_sentiment_model.pkl and tfidf_vectorizer.pkl to classify new tweets.

🎨 Visualizations
Bar plots generated in the project show sentiment distribution in the dataset:


✨ Results
The model achieved an accuracy of 90% on the test dataset.
Sentiment trends were successfully identified, revealing public opinions on diverse topics.

📂 Future Improvements
Incorporate deep learning models like LSTMs for improved sentiment prediction.
Add topic modeling to classify tweets by subject matter.

📜 License
This project is licensed under the MIT License.

🤝 Contributing
Feel free to contribute! Fork the repository, make your changes, and submit a pull request.

✨ Author - Tushar Shinde✨
