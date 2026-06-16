import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.datasets import (load_iris, load_breast_cancer, load_wine,
                               load_diabetes, make_classification, make_regression)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              mean_squared_error, r2_score, mean_absolute_error)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Tree Lab",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 50%, #0a1020 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1830 0%, #091220 100%);
    border-right: 1px solid #1e3a5f;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4fc3f7 !important;
    font-family: 'JetBrains Mono', monospace;
}

.main-header {
    background: linear-gradient(135deg, #0d2137 0%, #0a3d62 100%);
    border: 1px solid #1565c0;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(79,195,247,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.main-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4fc3f7, #81d4fa, #b3e5fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}

.main-header p {
    color: #78909c;
    font-size: 0.95rem;
    margin: 0.5rem 0 0 0;
    font-weight: 300;
}

.metric-card {
    background: linear-gradient(135deg, #0d2137 0%, #0a1a2e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}

.metric-card:hover {
    border-color: #4fc3f7;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #4fc3f7;
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    font-size: 0.75rem;
    color: #546e7a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #4fc3f7;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}

.tag {
    display: inline-block;
    background: rgba(79,195,247,0.1);
    border: 1px solid rgba(79,195,247,0.3);
    color: #4fc3f7;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.2rem;
}

.stTabs [data-baseweb="tab"] {
    color: #546e7a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

.stTabs [aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom: 2px solid #4fc3f7 !important;
}

.stSelectbox label, .stSlider label, .stRadio label {
    color: #90a4ae !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: #e3f2fd;
    border: 1px solid #1976d2;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding: 0.6rem 1.5rem;
    transition: all 0.3s;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0);
    border-color: #4fc3f7;
    box-shadow: 0 0 20px rgba(79,195,247,0.2);
    transform: translateY(-1px);
}

div[data-testid="stExpander"] {
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    background: rgba(13,33,55,0.5);
}

.highlight-box {
    background: rgba(21,101,192,0.1);
    border: 1px solid rgba(21,101,192,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

code {
    background: #0a1a2e !important;
    color: #4fc3f7 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌳 Decision Tree Lab</h1>
    <p>Interactive hyperparameter tuning · Real-time visualization · Full model diagnostics</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Task type
    st.markdown("### Task Type")
    task = st.radio("", ["Classification", "Regression"], label_visibility="collapsed")

    st.markdown("---")

    # Dataset
    st.markdown("### Dataset")
    if task == "Classification":
        dataset_name = st.selectbox("", ["Iris", "Breast Cancer", "Wine", "Synthetic"], label_visibility="collapsed")
    else:
        dataset_name = st.selectbox("", ["Diabetes", "Synthetic"], label_visibility="collapsed")

    st.markdown("---")

    # Hyperparameters
    st.markdown("### 🎛️ Hyperparameters")

    criterion_options = ["gini", "entropy", "log_loss"] if task == "Classification" else ["squared_error", "friedman_mse", "absolute_error", "poisson"]
    criterion = st.selectbox("Criterion", criterion_options)

    max_depth = st.slider("Max Depth", 1, 20, 3,
                          help="Maximum depth of the tree. None means fully grown.")
    use_none_depth = st.checkbox("Unlimited depth (None)", value=False)
    actual_depth = None if use_none_depth else max_depth

    min_samples_split = st.slider("Min Samples Split", 2, 50, 2,
                                   help="Min samples required to split a node.")
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 50, 1,
                                  help="Min samples required at a leaf node.")

    max_features_option = st.selectbox("Max Features", ["None (all)", "sqrt", "log2", "Custom"])
    if max_features_option == "Custom":
        max_features_val = st.slider("Custom Max Features", 1, 20, 5)
    elif max_features_option == "None (all)":
        max_features_val = None
    else:
        max_features_val = max_features_option

    max_leaf_nodes = st.slider("Max Leaf Nodes", 2, 100, 20,
                                help="Maximum number of leaf nodes.")
    use_none_leaves = st.checkbox("Unlimited leaf nodes", value=True)
    actual_leaf_nodes = None if use_none_leaves else max_leaf_nodes

    ccp_alpha = st.slider("CCP Alpha (Pruning)", 0.0, 0.1, 0.0, 0.001,
                           help="Cost-complexity pruning parameter. Higher = more pruning.")

    splitter = st.selectbox("Splitter", ["best", "random"])

    random_state = st.number_input("Random State", 0, 999, 42)

    st.markdown("---")

    # Train/test split
    st.markdown("### 📊 Data Split")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

    st.markdown("---")

    # Grid Search
    st.markdown("### 🔍 Grid Search")
    run_grid_search = st.checkbox("Enable Grid Search CV", value=False)
    cv_folds = st.slider("CV Folds", 2, 10, 5)

    st.markdown("---")
    run_btn = st.button("🚀 Train Model", use_container_width=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset(name, task):
    if task == "Classification":
        if name == "Iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target_names[data.target]
            return df, data.feature_names.tolist(), 'target', data.target_names.tolist()
        elif name == "Breast Cancer":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target_names[data.target]
            return df, data.feature_names.tolist(), 'target', data.target_names.tolist()
        elif name == "Wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target_names[data.target]
            return df, data.feature_names.tolist(), 'target', data.target_names.tolist()
        else:
            X, y = make_classification(n_samples=500, n_features=8, n_classes=3,
                                        n_informative=4, random_state=42)
            cols = [f"feature_{i}" for i in range(8)]
            df = pd.DataFrame(X, columns=cols)
            df['target'] = [f"class_{c}" for c in y]
            return df, cols, 'target', ["class_0", "class_1", "class_2"]
    else:
        if name == "Diabetes":
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df, data.feature_names.tolist(), 'target', None
        else:
            X, y = make_regression(n_samples=500, n_features=6, noise=20, random_state=42)
            cols = [f"feature_{i}" for i in range(6)]
            df = pd.DataFrame(X, columns=cols)
            df['target'] = y
            return df, cols, 'target', None

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
dataset_info = load_dataset(dataset_name, task)
df, feature_cols, target_col, class_names = dataset_info

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset", "🌳 Tree Viz", "📈 Performance", "🔍 Grid Search", "📋 Rules"
])

# ═══════════════════════════════════════════
# TAB 1 – DATASET EXPLORER
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Samples</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(feature_cols)}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    with col3:
        n_classes = len(class_names) if class_names else "—"
        st.markdown(f'<div class="metric-card"><div class="metric-value">{n_classes}</div><div class="metric-label">Classes</div></div>', unsafe_allow_html=True)
    with col4:
        missing = df.isnull().sum().sum()
        st.markdown(f'<div class="metric-card"><div class="metric-value">{missing}</div><div class="metric-label">Missing</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-header">Sample Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=300)

    with col_right:
        st.markdown('<div class="section-header">Feature Stats</div>', unsafe_allow_html=True)
        st.dataframe(df[feature_cols].describe().round(3), use_container_width=True, height=300)

    st.markdown('<div class="section-header">Feature Distributions</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, min(len(feature_cols), 4), figsize=(14, 3))
    fig.patch.set_facecolor('#0a0e1a')
    if len(feature_cols) == 1:
        axes = [axes]

    for i, (ax, feat) in enumerate(zip(axes, feature_cols[:4])):
        ax.set_facecolor('#0d1525')
        ax.hist(df[feat], bins=25, color='#1565c0', edgecolor='#4fc3f7', linewidth=0.5, alpha=0.85)
        ax.set_title(feat[:20], color='#90a4ae', fontsize=8, pad=5)
        ax.tick_params(colors='#546e7a', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a5f')

    plt.tight_layout(pad=1)
    st.pyplot(fig)
    plt.close()

    if task == "Classification":
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        class_dist = df[target_col].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        fig2.patch.set_facecolor('#0a0e1a')
        ax2.set_facecolor('#0d1525')
        colors = ['#1565c0', '#4fc3f7', '#0288d1', '#81d4fa', '#01579b']
        ax2.barh(class_dist.index, class_dist.values,
                 color=colors[:len(class_dist)], edgecolor='#4fc3f7', linewidth=0.5)
        ax2.set_xlabel('Count', color='#546e7a', fontsize=8)
        ax2.tick_params(colors='#90a4ae', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#1e3a5f')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
def train_model():
    le = LabelEncoder()
    X = df[feature_cols].values

    if task == "Classification":
        y = le.fit_transform(df[target_col])
    else:
        y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if task == "Classification":
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=actual_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features_val,
            max_leaf_nodes=actual_leaf_nodes,
            ccp_alpha=ccp_alpha,
            splitter=splitter,
            random_state=random_state
        )
    else:
        model = DecisionTreeRegressor(
            criterion=criterion,
            max_depth=actual_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features_val,
            max_leaf_nodes=actual_leaf_nodes,
            ccp_alpha=ccp_alpha,
            splitter=splitter,
            random_state=random_state
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Cross-val
    cv_scores = cross_val_score(model, X, y, cv=cv_folds,
                                 scoring='accuracy' if task == 'Classification' else 'r2')

    return {
        'model': model,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred,
        'le': le,
        'cv_scores': cv_scores,
        'class_names': class_names
    }

if run_btn:
    with st.spinner("Training Decision Tree..."):
        results = train_model()
        st.session_state.model = results['model']
        st.session_state.results = results
    st.success("✅ Model trained successfully!")

# ═══════════════════════════════════════════
# TAB 2 – TREE VISUALIZATION
# ═══════════════════════════════════════════
with tab2:
    if st.session_state.results is None:
        st.info("👈 Configure hyperparameters and click **Train Model** to visualize the tree.")
    else:
        results = st.session_state.results
        model = results['model']

        st.markdown('<div class="section-header">Decision Tree Structure</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 1])
        with col_right:
            fig_size = st.slider("Figure Width", 10, 40, 20)
            font_size = st.slider("Font Size", 6, 16, 9)
            max_depth_viz = st.slider("Viz Depth", 1, 10, min(4, model.get_depth() or 4))
            filled = st.checkbox("Filled nodes", True)
            rounded = st.checkbox("Rounded nodes", True)

        with col_left:
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.55))
            fig.patch.set_facecolor('#0a0e1a')
            ax.set_facecolor('#0a0e1a')

            display_class_names = None
            if task == "Classification" and class_names:
                display_class_names = class_names

            plot_tree(
                model,
                feature_names=feature_cols,
                class_names=display_class_names,
                filled=filled,
                rounded=rounded,
                max_depth=max_depth_viz,
                fontsize=font_size,
                ax=ax,
                impurity=True,
                proportion=False
            )

            for text in ax.texts:
                text.set_color('#e3f2fd')

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown('<div class="section-header">Tree Stats</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{model.get_depth()}</div><div class="metric-label">Actual Depth</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{model.get_n_leaves()}</div><div class="metric-label">Leaf Nodes</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{model.tree_.node_count}</div><div class="metric-label">Total Nodes</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(feature_cols)}</div><div class="metric-label">Features Used</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)

        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=True)

        fig3, ax3 = plt.subplots(figsize=(10, max(3, len(feature_cols) * 0.35)))
        fig3.patch.set_facecolor('#0a0e1a')
        ax3.set_facecolor('#0d1525')

        colors_bar = ['#4fc3f7' if v == fi_df['Importance'].max() else '#1565c0' for v in fi_df['Importance']]
        bars = ax3.barh(fi_df['Feature'], fi_df['Importance'], color=colors_bar, edgecolor='#0a2744', linewidth=0.5)

        for bar, val in zip(bars, fi_df['Importance']):
            ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', color='#90a4ae', fontsize=8)

        ax3.set_xlabel('Importance Score', color='#546e7a', fontsize=9)
        ax3.tick_params(colors='#90a4ae', labelsize=8)
        for spine in ax3.spines.values():
            spine.set_edgecolor('#1e3a5f')
        ax3.set_xlim(0, fi_df['Importance'].max() * 1.15)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ═══════════════════════════════════════════
# TAB 3 – PERFORMANCE
# ═══════════════════════════════════════════
with tab3:
    if st.session_state.results is None:
        st.info("👈 Train a model first to see performance metrics.")
    else:
        results = st.session_state.results
        model = results['model']
        y_test = results['y_test']
        y_pred = results['y_pred']
        cv_scores = results['cv_scores']

        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{acc:.3f}</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{cv_mean:.3f}</div><div class="metric-label">CV Mean</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{cv_std:.3f}</div><div class="metric-label">CV Std</div></div>', unsafe_allow_html=True)
            with c4:
                train_acc = accuracy_score(results['y_train'], model.predict(results['X_train']))
                overfit = train_acc - acc
                st.markdown(f'<div class="metric-card"><div class="metric-value">{overfit:.3f}</div><div class="metric-label">Overfit Gap</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_cm, col_cv = st.columns(2)

            with col_cm:
                st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                fig_cm.patch.set_facecolor('#0a0e1a')
                ax_cm.set_facecolor('#0d1525')

                cmap = plt.cm.Blues
                im = ax_cm.imshow(cm, interpolation='nearest', cmap=cmap)
                fig_cm.colorbar(im, ax=ax_cm)

                tick_marks = np.arange(len(class_names or np.unique(y_test)))
                labels = class_names or [str(i) for i in np.unique(y_test)]
                ax_cm.set_xticks(tick_marks)
                ax_cm.set_xticklabels(labels, rotation=30, ha='right', color='#90a4ae', fontsize=8)
                ax_cm.set_yticks(tick_marks)
                ax_cm.set_yticklabels(labels, color='#90a4ae', fontsize=8)

                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax_cm.text(j, i, str(cm[i, j]),
                                   ha='center', va='center', fontsize=11, fontweight='bold',
                                   color='white' if cm[i, j] > thresh else '#4fc3f7')

                ax_cm.set_ylabel('True', color='#546e7a', fontsize=9)
                ax_cm.set_xlabel('Predicted', color='#546e7a', fontsize=9)
                for spine in ax_cm.spines.values():
                    spine.set_edgecolor('#1e3a5f')
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()

            with col_cv:
                st.markdown('<div class="section-header">Cross-Validation Scores</div>', unsafe_allow_html=True)
                fig_cv, ax_cv = plt.subplots(figsize=(5, 4))
                fig_cv.patch.set_facecolor('#0a0e1a')
                ax_cv.set_facecolor('#0d1525')

                fold_nums = [f"Fold {i+1}" for i in range(len(cv_scores))]
                colors_cv = ['#4fc3f7' if s == cv_scores.max() else '#1565c0' for s in cv_scores]
                bars_cv = ax_cv.bar(fold_nums, cv_scores, color=colors_cv, edgecolor='#0a2744', linewidth=0.5)
                ax_cv.axhline(cv_mean, color='#ef5350', linestyle='--', linewidth=1.5, label=f'Mean: {cv_mean:.3f}')

                for bar, val in zip(bars_cv, cv_scores):
                    ax_cv.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                               f'{val:.3f}', ha='center', color='#90a4ae', fontsize=8)

                ax_cv.set_ylim(max(0, cv_scores.min() - 0.1), min(1.05, cv_scores.max() + 0.1))
                ax_cv.set_ylabel('Accuracy', color='#546e7a', fontsize=9)
                ax_cv.tick_params(colors='#90a4ae', labelsize=8)
                ax_cv.legend(fontsize=8, facecolor='#0d1525', labelcolor='#90a4ae', edgecolor='#1e3a5f')
                for spine in ax_cv.spines.values():
                    spine.set_edgecolor('#1e3a5f')
                plt.tight_layout()
                st.pyplot(fig_cv)
                plt.close()

            st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
            labels_for_report = class_names or None
            cr = classification_report(y_test, y_pred, target_names=labels_for_report, output_dict=True)
            cr_df = pd.DataFrame(cr).transpose().round(3)
            st.dataframe(cr_df, use_container_width=True)

        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{r2:.3f}</div><div class="metric-label">R² Score</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{rmse:.2f}</div><div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{mae:.2f}</div><div class="metric-label">MAE</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{cv_scores.mean():.3f}</div><div class="metric-label">CV R²</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_pred, col_res = st.columns(2)

            with col_pred:
                st.markdown('<div class="section-header">Predicted vs Actual</div>', unsafe_allow_html=True)
                fig_pv, ax_pv = plt.subplots(figsize=(5, 4))
                fig_pv.patch.set_facecolor('#0a0e1a')
                ax_pv.set_facecolor('#0d1525')
                ax_pv.scatter(y_test, y_pred, alpha=0.6, color='#1565c0', edgecolors='#4fc3f7', linewidth=0.5, s=30)
                lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                ax_pv.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.8)
                ax_pv.set_xlabel('Actual', color='#546e7a', fontsize=9)
                ax_pv.set_ylabel('Predicted', color='#546e7a', fontsize=9)
                ax_pv.tick_params(colors='#90a4ae', labelsize=7)
                for spine in ax_pv.spines.values():
                    spine.set_edgecolor('#1e3a5f')
                plt.tight_layout()
                st.pyplot(fig_pv)
                plt.close()

            with col_res:
                st.markdown('<div class="section-header">Residuals</div>', unsafe_allow_html=True)
                residuals = y_test - y_pred
                fig_res, ax_res = plt.subplots(figsize=(5, 4))
                fig_res.patch.set_facecolor('#0a0e1a')
                ax_res.set_facecolor('#0d1525')
                ax_res.scatter(y_pred, residuals, alpha=0.6, color='#1565c0', edgecolors='#4fc3f7', linewidth=0.5, s=30)
                ax_res.axhline(0, color='#ef5350', linestyle='--', linewidth=1.5)
                ax_res.set_xlabel('Predicted', color='#546e7a', fontsize=9)
                ax_res.set_ylabel('Residuals', color='#546e7a', fontsize=9)
                ax_res.tick_params(colors='#90a4ae', labelsize=7)
                for spine in ax_res.spines.values():
                    spine.set_edgecolor('#1e3a5f')
                plt.tight_layout()
                st.pyplot(fig_res)
                plt.close()

# ═══════════════════════════════════════════
# TAB 4 – GRID SEARCH
# ═══════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🔍 Hyperparameter Grid Search</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
        Configure the parameter grid below and run an exhaustive search to find the best hyperparameters 
        using cross-validation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_g1, col_g2, col_g3 = st.columns(3)

    with col_g1:
        gs_max_depths = st.multiselect("Max Depth values", [1,2,3,4,5,6,8,10,15,20], default=[2,3,5,8])
    with col_g2:
        gs_min_split = st.multiselect("Min Samples Split", [2,5,10,20,50], default=[2,5,10])
    with col_g3:
        gs_min_leaf = st.multiselect("Min Samples Leaf", [1,2,5,10,20], default=[1,2,5])

    gs_criterion = st.multiselect(
        "Criterion",
        ["gini", "entropy", "log_loss"] if task == "Classification" else ["squared_error", "friedman_mse", "absolute_error"],
        default=["gini", "entropy"] if task == "Classification" else ["squared_error", "friedman_mse"]
    )

    gs_btn = st.button("🔍 Run Grid Search", use_container_width=True)

    if gs_btn:
        if not gs_max_depths or not gs_min_split or not gs_min_leaf:
            st.warning("Select at least one value for each parameter.")
        else:
            param_grid = {
                'max_depth': gs_max_depths,
                'min_samples_split': gs_min_split,
                'min_samples_leaf': gs_min_leaf,
                'criterion': gs_criterion
            }

            le = LabelEncoder()
            X = df[feature_cols].values
            y = le.fit_transform(df[target_col]) if task == "Classification" else df[target_col].values

            estimator = DecisionTreeClassifier(random_state=random_state) if task == "Classification" else DecisionTreeRegressor(random_state=random_state)
            scoring = 'accuracy' if task == "Classification" else 'r2'

            total = 1
            for v in param_grid.values():
                total *= len(v)

            with st.spinner(f"Running {total} combinations × {cv_folds} folds = {total * cv_folds} fits..."):
                gs = GridSearchCV(estimator, param_grid, cv=cv_folds, scoring=scoring,
                                  n_jobs=-1, return_train_score=True)
                gs.fit(X, y)

            st.success(f"✅ Grid Search complete! Best score: **{gs.best_score_:.4f}**")

            st.markdown('<div class="section-header">Best Parameters</div>', unsafe_allow_html=True)
            best_params_df = pd.DataFrame([gs.best_params_])
            st.dataframe(best_params_df, use_container_width=True)

            st.markdown('<div class="section-header">Full Results</div>', unsafe_allow_html=True)
            results_df = pd.DataFrame(gs.cv_results_)
            cols_show = ['param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf',
                         'param_criterion', 'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']
            cols_show = [c for c in cols_show if c in results_df.columns]
            results_df_show = results_df[cols_show].sort_values('rank_test_score')
            results_df_show.columns = [c.replace('param_', '').replace('_', ' ').title() for c in results_df_show.columns]
            st.dataframe(results_df_show.head(20), use_container_width=True)

            # Heatmap: max_depth vs min_samples_split
            if len(gs_max_depths) > 1 and len(gs_min_split) > 1:
                st.markdown('<div class="section-header">Score Heatmap (Depth × Min Split)</div>', unsafe_allow_html=True)
                pivot = results_df.pivot_table(
                    values='mean_test_score',
                    index='param_max_depth',
                    columns='param_min_samples_split',
                    aggfunc='max'
                )
                fig_hm, ax_hm = plt.subplots(figsize=(8, 4))
                fig_hm.patch.set_facecolor('#0a0e1a')
                ax_hm.set_facecolor('#0d1525')
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Blues',
                            ax=ax_hm, linewidths=0.5, linecolor='#1e3a5f',
                            annot_kws={'size': 9, 'color': 'white'})
                ax_hm.set_xlabel('Min Samples Split', color='#546e7a', fontsize=9)
                ax_hm.set_ylabel('Max Depth', color='#546e7a', fontsize=9)
                ax_hm.tick_params(colors='#90a4ae', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig_hm)
                plt.close()

# ═══════════════════════════════════════════
# TAB 5 – TEXT RULES
# ═══════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">📋 Decision Rules (Text Format)</div>', unsafe_allow_html=True)

    if st.session_state.results is None:
        st.info("👈 Train a model to see the decision rules.")
    else:
        model = st.session_state.results['model']

        max_depth_rules = st.slider("Max depth to show in rules", 1, min(10, model.get_depth() or 10),
                                     min(4, model.get_depth() or 4))

        tree_rules = export_text(model, feature_names=feature_cols, max_depth=max_depth_rules,
                                  decimals=3, show_weights=True)

        st.code(tree_rules, language="text")

        st.download_button(
            "⬇️ Download Rules as .txt",
            data=tree_rules,
            file_name="decision_tree_rules.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown('<div class="section-header">Model Parameters Used</div>', unsafe_allow_html=True)
        params = st.session_state.results['model'].get_params()
        params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
        st.dataframe(params_df, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<br><br>
<div style="text-align:center; color: #263238; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
    Decision Tree Lab · Built with Streamlit + scikit-learn
</div>
""", unsafe_allow_html=True)