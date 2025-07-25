# 🧠 Employee Salary Prediction App

This project is a **Machine Learning-based web application** that predicts whether an individual's income is `<=50K` or `>50K` using demographic and professional data.

It supports:  
✅ Single-entry prediction via a form  
📂 Batch prediction via CSV file upload  

Built with **Python**, **Streamlit**, and **scikit-learn**.

---

## 🚀 Demo  
Live Deployment: [Click here to access the app](https://employee-salary-prediction-yagq4bbmmov5ut8w7egc56.streamlit.app/)

---

## 📌 Features

- Trained on UCI Adult Income dataset.  
- Multiple ML models were tested:  
  - Logistic Regression  
  - K-Nearest Neighbors  
  - Random Forest Classifier  
  - Gradient Boosting Classifier ✅ *(Best performing model)*  
- Final model: **Gradient Boosting Classifier** with an accuracy of **85.77%**.  
- Batch and single prediction support.  
- User-friendly UI with Streamlit.

---

## 🧠 Tech Stack

| Category       | Tools/Frameworks Used               |
| -------------- | --------------------------------- |
| Language       | Python                            |
| ML Libraries   | scikit-learn, pandas, joblib      |
| Visualization  | Streamlit                        |
| Deployment     | Streamlit Cloud                  |

---

## Folder Structure

- employee-salary-prediction/
  - app2.py                # Streamlit App
  - best_model.pkl         # Trained ML model
  - requirements.txt       # Required Python libraries
  - sample_input.csv       # Sample batch input CSV file
  - README.md              # Project readme
  - notebooks/
    - model_building.ipynb # Colab notebook for training



---

## 📄 Sample Input File
To test batch prediction, use the file sample_input.csv provided in the repo.
It contains the correct structure and columns expected by the model.

## 📊 Model Performance

- **Accuracy:** 85.77%  
- **Best Model:** Gradient Boosting Classifier

---

## 📚 References

- UCI Adult Dataset  
- Streamlit Documentation  
- scikit-learn Documentation  

---

## 🤝 Contributing

Feel free to fork this repo and suggest improvements or open issues. PRs are welcome!

---

## 🧑‍💻 Author

**Vani Sree**  
[GitHub Profile](https://github.com/vanisree2204-sys)

---

## 📌 License

This project is licensed under the MIT License.

---

## 🎉 Thank You!

> **Thank you for checking out my Employee Salary Prediction project!**  
> I hope this app helps you explore how machine learning can make data-driven decisions easier and more accessible.  
>  
> Feel free to reach out if you have any questions, ideas, or want to collaborate.  
> Happy coding and best of luck with your projects! 🚀😊
