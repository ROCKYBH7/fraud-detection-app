# 💳 Fraud Detection App

A Streamlit-based interactive dashboard to predict fraudulent transactions using a trained machine learning model. Visualizations include probability charts, feature importance, and sample transaction previews.

---

## 📌 Features

- Predict whether a transaction is **Fraudulent** or **Not Fraudulent**.
- View **fraud probability** via an interactive donut chart.
- Analyze **top 5 feature importances** for the model.
- Explore **sample transactions** in a table.
- Dark mode UI for better readability.

---

## 🛠 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/ROCKYBH7/fraud-detection-app.git
cd fraud-detection-app

```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
```


3. **Activate the virtual environment:**

- **Windows:**
  
```bash
venv\Scripts\activate
```

- **Linux / MacOS:**

```bash
source venv/bin/activate
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## 📂 Dataset

The dataset used in this project (~136MB) cannot be included in the repository due to GitHub's 100MB file size limit.

Dwnload it here:

🔗 [fraudTest.csv - Google Drive](https://drive.google.com/file/d/1Bi1FBGutUHsaYNDi_88HpjkDYIjiSD_S/view?usp=sharing)

After downloading, place the file inside the `Data/` folder:

## 📂 Project Structure

Fraud-Detection-Project/

│── saved_models/ # (optional folder for storing models)

│── .gitattributes # Git LFS tracking file

│── .gitignore # Ignored files list

│── README.md # Project documentation

│── fraud_app_final.py # Streamlit app

│── fraud_model.pkl # Pre-trained ML model

│── requirements.txt # Dependencies

│── train_model.py # Script to train model

│── Data/ # (create manually)

│     └── fraudTest.csv # Dataset (downloaded separately)


⚠️ Ensure the dataset remains in the Data/ folder with the same filename for the app to function properly.

## 🏃‍♂️ Usage

Run the Streamlit app:

```bash
streamlit run fraud_app_final.py
```

-Enter transaction details, user information, and merchant information using the sidebar.

- Click Predict Fraud to see the result.

- View the probability donut chart, top feature importances, and sample transactions.

## 🧰 Tools & Libraries

- Python 3.13

- Streamlit

- Pandas, NumPy

- Scikit-learn

- Plotly

- Git LFS (for large dataset)

## 👤 Author

**Balaji R H**  

- GitHub: [ROCKYBH7](https://github.com/ROCKYBH7)  
- LinkedIn: [Balaji R H](https://www.linkedin.com/in/balaji-r-h-a81107298)


## ⚠️ Notes

- Ensure fraudTest.csv is in the correct Data/ folder.

- The trained model (fraud_model.pkl) is included, so predictions will work out-of-the-box.

- Recommended Python version: 3.13.3

