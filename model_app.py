import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

from xgboost import XGBClassifier

# Implementing OOPs and Creating a DibetesAnalyzer class for predicting continuous disease progression(regression) and Classifying High vs Low Disease progression (Classification).
class DibetesAnalyzer:
    def __init__(self, random_state=5):
        
        """Initialize regression models, pipelines, and Implementing hyperparameter tuning with GridSearchCV."""
        
        self.random_state = random_state
        self.feature_names = None
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=random_state)
        }

        # RobustScaler handles outliers in unscaled features, ensuring robust preprocessing.
        
        self.pipelines = {
            name: Pipeline([('scaler', RobustScaler()), ('model', model)])
            for name, model in self.models.items()
        }

        # Hyperparameter Tuning.

        self.param_grids = {
            'Random Forest': {
                'model__n_estimators': [50, 100],
                'model__max_depth': [10, 20],
                'model__min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 4]
            }
        }
        self.results = {}
        self.best_params = {}
        self.voting_pipeline = None

    def evaluate_model(self, pipeline, X_train, X_test, y_train, y_test):

        """This Function gives Evaluation scores for Each Regression Model"""

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred)**0.5  # RMSE Calculated quite manually for convenience
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'R2Score': r2}

    def analyze_split(self, X, y, test_size, split_name):
      
        """Splitting the dataset in train and test splits and Initializing the Pipelines with 3 regression models and an aggregating voting regressor(Which combines them all)"""
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=self.random_state)
        self.results[split_name] = {}
        for name, pipeline in self.pipelines.items():
            if name in self.param_grids:
                grid_search = GridSearchCV(pipeline, self.param_grids[name], cv=5, 
                                          scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                self.best_params[name] = {k.split('__')[1]: v for k, v in grid_search.best_params_.items()}
                self.results[split_name][name] = self.evaluate_model(grid_search.best_estimator_, 
                                                                    X_train, X_test, y_train, y_test)
            else:
                self.results[split_name][name] = self.evaluate_model(pipeline, X_train, X_test, y_train, y_test)

        # Voting Regressor combines individual models for enhanced performance.
        voting_reg = VotingRegressor([
            ('lr', self.models['Linear Regression']),
            ('rf', RandomForestRegressor(**self.best_params.get('Random Forest', 
                                                               {'n_estimators': 100, 'random_state': self.random_state}))),
            ('gb', GradientBoostingRegressor(**self.best_params.get('Gradient Boosting', 
                                                                   {'n_estimators': 100, 'learning_rate': 0.1, 
                                                                    'random_state': self.random_state})))
        ])
        self.voting_pipeline = Pipeline([('scaler', RobustScaler()), ('model', voting_reg)])
        self.results[split_name]['Voting Regressor'] = self.evaluate_model(self.voting_pipeline, 
                                                                         X_train, X_test, y_train, y_test)
        if test_size == 0.2:
            self.voting_pipeline.fit(X_train, y_train)

    def run(self, X, y):

        """Run regression Models for 80/20 and 70/30 splits and combine every model with a voting regressor"""
        
        self.feature_names = load_diabetes().feature_names
        for test_size, split_name in [(0.2, '80/20'), (0.3, '70/30')]:
            self.analyze_split(X, y, test_size, split_name)

    def plot_results(self, split_name):
        
        """Visualize regression performance metrics such as RMSE Score,MAE and R2 Score using Seaborn bar plots."""

        df = pd.DataFrame(self.results[split_name]).T.round(3).reset_index().rename(columns={'index': 'Model'})
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Regression Model Performance ({split_name} Split)', fontsize=16)
        
        sns.barplot(data=df, x='Model', y='RMSE', ax=axes[0], palette='Blues_d')
        axes[0].set_title('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='Model', y='MAE', ax=axes[1], palette='Greens_d')
        axes[1].set_title('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='Model', y='R2Score', ax=axes[2], palette='Reds_d')
        axes[2].set_title('R2 Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def predict(self, input_data):

        """Function Predict continuous disease progression using the Voting Regressor."""
       
        if self.voting_pipeline:
            if len(input_data) != len(self.feature_names):
                return None
            try:
                input_array = np.array([input_data], dtype=float)
                return self.voting_pipeline.predict(input_array)[0]
            except ValueError:
                return None
        return None

# Classification analyzer using multiple models for training but only Logistic Regression for prediction.
class DiabetesClassificationAnalyzer:
    def __init__(self, random_state=5):

        """Initialize classification models, pipelines, and and Implementing hyperparameter tuning with GridSearchCV."""
        
        self.random_state = random_state
        self.feature_names = None
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'XGBoost': XGBClassifier(random_state=self.random_state, objective='binary:logistic')
        }
        # Using RobustScaler for treating Outliers 
        self.pipelines = {
            name: Pipeline([('scaler', RobustScaler()), ('model', model)])
            for name, model in self.models.items()
        }
        # Initializing Grids
        self.param_grids = {
            'Random Forest': {
                'model__n_estimators': [50, 200],
                'model__max_depth': [10],
                'model__min_samples_split': [2, 5]
            },
            'XGBoost': {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 4]
            },
            'Logistic Regression': {
                'model__C': [0.1, 1.0, 10.0],
                'model__penalty': ['l2']
            }
        }
        self.results = {}
        self.best_params = {}
        self.logistic_pipeline = None
        self.threshold = None
        self.logistic_confusion_matrix = {}  # Store Logistic Regression confusion matrix
        self.logistic_roc_data = {}  # Store Logistic Regression ROC data 

    def evaluate_model(self, pipeline, X_train, X_test, y_train, y_test):

        """Evaluate classification model performance using accuracy, precision, recall, and F1-score."""
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        if pipeline.steps[-1][1].__class__.__name__ == 'LogisticRegression':
            cm = confusion_matrix(y_test, y_pred)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            return {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1Score': f1,
                'ConfusionMatrix': cm,
                'FPR': fpr,
                'TPR': tpr,
                'AUC': auc
            }
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1Score': f1}

    def analyze_split(self, X, y, test_size, split_name):
        """Train and evaluate all classification models for a specified train-test split."""
        # Stratified split ensures balanced classes.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=self.random_state, stratify=y)
        self.results[split_name] = {}
        self.logistic_confusion_matrix[split_name] = None
        self.logistic_roc_data[split_name] = None

        for name, pipeline in self.pipelines.items():
            if name in self.param_grids:
                grid_search = GridSearchCV(pipeline, self.param_grids[name], cv=5, 
                                          scoring='f1', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                self.best_params[name] = {k.split('__')[1]: v for k, v in grid_search.best_params_.items()}
                self.results[split_name][name] = self.evaluate_model(grid_search.best_estimator_, 
                                                                    X_train, X_test, y_train, y_test)
                if name == 'Logistic Regression':
                    self.logistic_confusion_matrix[split_name] = self.results[split_name][name].pop('ConfusionMatrix', None)
                    self.logistic_roc_data[split_name] = {
                        'FPR': self.results[split_name][name].pop('FPR', None),
                        'TPR': self.results[split_name][name].pop('TPR', None),
                        'AUC': self.results[split_name][name].pop('AUC', None)
                    }
            else:
                self.results[split_name][name] = self.evaluate_model(pipeline, X_train, X_test, y_train, y_test)
                if name == 'Logistic Regression':
                    self.logistic_confusion_matrix[split_name] = self.results[split_name][name].pop('ConfusionMatrix', None)
                    self.logistic_roc_data[split_name] = {
                        'FPR': self.results[split_name][name].pop('FPR', None),
                        'TPR': self.results[split_name][name].pop('TPR', None),
                        'AUC': self.results[split_name][name].pop('AUC', None)
                    }

        # Stacking Classifier combines all models for training evaluation.
        stacking_clf = StackingClassifier(
            estimators=[
                ('lr', self.models['Logistic Regression']),
                ('rf', RandomForestClassifier(**self.best_params.get('Random Forest', 
                                                                    {'n_estimators': 100, 'max_depth': 10, 
                                                                     'min_samples_split': 2, 'random_state': self.random_state}))),
                ('xgb', XGBClassifier(**self.best_params.get('XGBoost', 
                                                             {'n_estimators': 100, 'learning_rate': 0.1, 
                                                              'max_depth': 3, 'random_state': self.random_state})))
            ],
            final_estimator=LogisticRegression(random_state=self.random_state)
        )
        stacking_pipeline = Pipeline([('scaler', RobustScaler()), ('model', stacking_clf)])
        self.results[split_name]['Stacking Classifier'] = self.evaluate_model(stacking_pipeline, 
                                                                            X_train, X_test, y_train, y_test)
        
        # Store Logistic Regression pipeline for predictions.
        if test_size == 0.2:
            logistic_grid = GridSearchCV(self.pipelines['Logistic Regression'], 
                                        self.param_grids['Logistic Regression'], 
                                        cv=5, scoring='f1', n_jobs=-1)
            logistic_grid.fit(X_train, y_train)
            self.logistic_pipeline = logistic_grid.best_estimator_
            self.logistic_pipeline.fit(X_train, y_train)

    def run(self, X, y):
        """Execute classification analysis for 80/20 and 70/30 splits."""
        self.feature_names = load_diabetes().feature_names
        for test_size, split_name in [(0.2, '80/20'), (0.3, '70/30')]:
            self.analyze_split(X, y, test_size, split_name)

    def plot_results(self, split_name):
        """Visualize classification performance metrics and AUC-ROC curve using Seaborn bar plots."""
        df = pd.DataFrame(self.results[split_name]).T.round(3).reset_index().rename(columns={'index': 'Model'})
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Classification Model Performance ({split_name} Split)', fontsize=16)
        
        # Plot performance metrics
        sns.barplot(data=df, x='Model', y='Accuracy', ax=axes[0, 0], palette='Blues_d')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='Model', y='Precision', ax=axes[0, 1], palette='Greens_d')
        axes[0, 1].set_title('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='Model', y='Recall', ax=axes[0, 2], palette='Reds_d')
        axes[0, 2].set_title('Recall')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        sns.barplot(data=df, x='Model', y='F1Score', ax=axes[1, 0], palette='Purples_d')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot AUC-ROC curve for Logistic Regression
        if split_name in self.logistic_roc_data and self.logistic_roc_data[split_name] is not None:
            fpr = self.logistic_roc_data[split_name]['FPR']
            tpr = self.logistic_roc_data[split_name]['TPR']
            auc = self.logistic_roc_data[split_name]['AUC']
            axes[1, 1].plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', color='blue')
            axes[1, 1].plot([0, 1], [0, 1], 'k--')  # Diagonal line
            axes[1, 1].set_xlim([0.0, 1.0])
            axes[1, 1].set_ylim([0.0, 1.05])
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('ROC Curve: Logistic Regression')
            axes[1, 1].legend(loc='lower right')
        else:
            axes[1, 1].text(0.5, 0.5, 'ROC Curve Unavailable', ha='center', va='center')
            axes[1, 1].set_title('ROC Curve: Logistic Regression')
        
        axes[1, 2].axis('off')  # Empty subplot for layout
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def plot_confusion_matrix(self, split_name):
        """Plot confusion matrix for Logistic Regression."""
        cm = self.logistic_confusion_matrix[split_name]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
        ax.set_title(f'Confusion Matrix: Logistic Regression ({split_name} Split)')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        return fig

    def predict(self, input_data):
        """Predict binary disease progression (Low or High) using Logistic Regression."""
        if self.logistic_pipeline:
            if len(input_data) != len(self.feature_names):
                return None
            try:
                input_array = np.array([input_data], dtype=float)
                # Output "Low" or "High" based on binary prediction.
                prediction = self.logistic_pipeline.predict(input_array)[0]
                return "High" if prediction == 1 else "Low"
            except ValueError:
                return None
        return None

def main():
    """Main Streamlit application for regression and classification analysis of diabetes progression."""
    st.set_page_config(page_title="Diabetes Regression & Classification Analysis", layout="wide")
    
    # Configure sidebar navigation for the four required pages.
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a Page", [
        "Regression Training & Results",
        "Regression Prediction",
        "Classification Training & Results",
        "Classification Prediction"
    ])

    # Load diabetes dataset with unscaled features, consistent with provided code.
    diabetes = load_diabetes(scaled=False)
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name='Progression')
    # Generate binary labels using median threshold for classification.
    threshold = y.median()
    y_binary = (y > threshold).astype(int)
    df = X.copy()
    df['Progression'] = y
    df['High_Progression'] = y_binary

    # Initialize analyzers in session state for persistence.
    if 'reg_analyzer' not in st.session_state:
        st.session_state.reg_analyzer = DibetesAnalyzer()
    if 'clf_analyzer' not in st.session_state:
        st.session_state.clf_analyzer = DiabetesClassificationAnalyzer()
        st.session_state.clf_analyzer.threshold = threshold

    if page == "Regression Training & Results":
        st.title("Regression Model Training and Results")
        st.markdown("""
            This page displays the diabetes dataset and evaluates regression models for predicting disease progression. 
            Unscaled features are preprocessed with RobustScaler within model pipelines to address outliers. 
           
        """)

        st.subheader("Dataset Preview (Raw Data)")
        st.dataframe(df.drop(columns=['High_Progression']).head())

        if st.button("Train Regression Models"):
            with st.spinner("Training regression models..."):
                st.session_state.reg_analyzer.run(X, y)
            
            for split_name in st.session_state.reg_analyzer.results:
                st.subheader(f"{split_name} Split Results")
                st.dataframe(pd.DataFrame(st.session_state.reg_analyzer.results[split_name]).T.round(3))
                st.pyplot(st.session_state.reg_analyzer.plot_results(split_name))
                st.write(f"Best Parameters for {split_name}:", st.session_state.reg_analyzer.best_params)

    elif page == "Regression Prediction":
        st.title("Regression Prediction")
        st.markdown("""
            This page enables input of unscaled feature values to predict continuous disease progression using the Voting Regressor. 
            The model pipeline incorporates RobustScaler for consistent preprocessing.
        """)

        if not st.session_state.reg_analyzer.voting_pipeline:
            st.warning("Please train the regression models first on the 'Regression Training & Results' page.")
        else:
            st.subheader("Enter Unscaled Feature Values")
            input_data = []
            for feature in st.session_state.reg_analyzer.feature_names:
                min_val, max_val = df[feature].min(), df[feature].max()
                input_data.append(st.number_input(f"{feature} (Range: {min_val:.4f} to {max_val:.4f})", 
                                                 value=0.0, format="%.6f"))
            
            if st.button("Predict"):
                prediction = st.session_state.reg_analyzer.predict(input_data)
                if prediction is not None:
                    st.success(f"Predicted Disease Progression: {prediction:.2f}")
                else:
                    st.error("Prediction failed. Please ensure all inputs are valid numbers and models are trained.")

    elif page == "Classification Training & Results":
        st.title("Classification Model Training and Results")
        st.markdown(f"""
            This page presents the diabetes dataset with binary labels (High for progression > {threshold:.2f}, Low otherwise) 
            and evaluates multiple classification models, including Logistic Regression, Random Forest, XGBoost, and a Stacking Classifier. 
            Unscaled features are processed with RobustScaler, ensuring consistency with regression pipelines.
        """)

        st.subheader("Dataset Preview (Unscaled Features, Binary Labels)")
        st.dataframe(df.drop(columns=['Progression']).head())

        if st.button("Train Classification Models"):
            with st.spinner("Training classification models..."):
                st.session_state.clf_analyzer.run(X, y_binary)
            
            for split_name in st.session_state.clf_analyzer.results:
                st.subheader(f"{split_name} Split Results")
                st.dataframe(pd.DataFrame(st.session_state.clf_analyzer.results[split_name]).T.round(3))
                st.pyplot(st.session_state.clf_analyzer.plot_results(split_name))
                st.write(f"Best Parameters for {split_name}:", st.session_state.clf_analyzer.best_params)

                st.subheader(f"Confusion Matrix ({split_name} Split)")
                st.write("Logistic Regression was identified as the best-performing model.")
                if split_name in st.session_state.clf_analyzer.logistic_confusion_matrix and st.session_state.clf_analyzer.logistic_confusion_matrix[split_name] is not None:
                    st.pyplot(st.session_state.clf_analyzer.plot_confusion_matrix(split_name))
                else:
                    st.warning("Confusion matrix for Logistic Regression not available.")

    elif page == "Classification Prediction":
        st.title("Classification Prediction")
        st.markdown(f"""
            This page allows input of unscaled feature values to predict binary disease progression 
            (High for progression > {st.session_state.clf_analyzer.threshold:.2f}, Low otherwise) 
            using Logistic Regression, identified as the best-performing model. 
            The pipeline includes RobustScaler for preprocessing.
        """)

        if not st.session_state.clf_analyzer.logistic_pipeline:
            st.warning("Please train the classification models first on the 'Classification Training & Results' page.")
        else:
            st.subheader("Enter Unscaled Feature Values")
            input_data = []
            for feature in st.session_state.clf_analyzer.feature_names:
                min_val, max_val = df[feature].min(), df[feature].max()
                input_data.append(st.number_input(f"{feature} (Range: {min_val:.4f} to {max_val:.4f})", 
                                                 value=0.0, format="%.6f"))
            
            if st.button("Predict"):
                prediction = st.session_state.clf_analyzer.predict(input_data)
                if prediction is not None:
                    st.success(f"Predicted Disease Progression: {prediction}")
                    st.write(f"Logistic Regression was the best-performing model for this prediction.")
                    st.write(f"Interpretation: {prediction} indicates {'progression >' if prediction == 'High' else 'progression ≤'} {st.session_state.clf_analyzer.threshold:.2f}")
                else:
                    st.error("Prediction failed. Please ensure all inputs are valid numbers and the model is trained.")

if __name__ == "__main__":
    main()