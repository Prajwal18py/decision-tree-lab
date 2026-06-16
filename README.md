# 🌳 Decision Tree Lab

An interactive Decision Tree visualization tool built with **Streamlit** and **scikit-learn**. Tune hyperparameters in real-time, visualize tree structure, run Grid Search CV, and export decision rules — all from a clean dark UI.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎛️ Hyperparameter Tuning | max_depth, criterion, min_samples_split, min_samples_leaf, ccp_alpha, splitter, max_features, max_leaf_nodes |
| 🌳 Tree Visualization | Interactive depth-controlled tree plot with feature importance bar chart |
| 📊 Dataset Explorer | Distribution plots, class balance, feature stats — Iris, Breast Cancer, Wine, Diabetes, Synthetic |
| 📈 Performance Metrics | Accuracy, CV score, confusion matrix, classification report (classification) · R², RMSE, MAE, residual plots (regression) |
| 🔍 Grid Search CV | Exhaustive param grid with heatmap visualization of results |
| 📋 Decision Rules | Export human-readable text rules as `.txt` |

---

## 🚀 Setup & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Prajwal18py/decision-tree-lab.git
cd decision-tree-lab

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📁 Project Structure

```
decision-tree-lab/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🖥️ Usage

1. **Sidebar → Task Type**: Switch between Classification and Regression
2. **Sidebar → Dataset**: Pick a built-in dataset or generate synthetic data
3. **Sidebar → Hyperparameters**: Tune all major Decision Tree parameters
4. **Train Model** button → results update across all tabs

### Tabs
- **Dataset** — EDA, distributions, class balance
- **Tree Viz** — Visual tree + feature importance
- **Performance** — Metrics, confusion matrix, CV scores
- **Grid Search** — Exhaustive hyperparameter search with heatmap
- **Rules** — Text export of decision rules

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/)

---

## 📬 Author

**Prajwal** — [github.com/Prajwal18py](https://github.com/Prajwal18py)