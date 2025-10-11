
# 🩺 Diabetic Detection in Pregnant Women

## 📌 Overview
This project aims to predict the likelihood of diabetes in pregnant women based on key medical parameters.  
It leverages **machine learning models** to analyze health-related features and provide accurate predictions that can assist in early diagnosis and preventive healthcare.

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib  
- **Model:** Decision Tree Classifier  
- **Environment:** Virtual Environment (venv)

---

## 🧩 Project Structure
```

Healthcare/
├── Diabetic_Pregnant_Women_Detection/
│   ├── data/                 # Dataset used for training/testing
│   ├── notebooks/            # Jupyter notebook for model training
│   ├── model.pkl             # Trained Decision Tree model
│   ├── main.py               # Script for model prediction
│   ├── README.md             # Project documentation
│   └── requirements.txt      # Dependencies

````

---

## 🚀 How to Setup and Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/almohsinkhan/DataSentience-AIML.git
cd src/Healthcare/Diabetic_Pregnant_Women_Detection
````

### 2️⃣ Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the model

```bash
python main.py
```

---

## 🧠 How It Works

1. The dataset is preprocessed (handling missing values, scaling, etc.).
2. The model (`DecisionTreeClassifier`) is trained on the dataset.
3. The trained model is saved as `model.pkl`.
4. The `main.py` script loads this model and takes user input for medical parameters to predict diabetes risk.
5. Finally, it prints whether the pregnant woman is **likely diabetic or not**.

---

## 📊 Model Performance

* **Algorithm:** Decision Tree Classifier
* **Accuracy:** ~77% on test data after hyperparameter tuning (max_depth, criterion)

---

## 🙌 Contributors

* **Mohsin Khan** 

---

## 💡 Future Improvements

* Implement ensemble models (Random Forest, XGBoost)
* Add GUI or web-based interface using Streamlit
* Deploy the model using Flask API

