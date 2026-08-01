# 📉 User-Friction-Analyzer

**Author:** Sana P
**MUID:** sana-1@mulearn

## 📖 Problem Statement
Customer churn is a critical metric for subscription-based businesses. The cost of acquiring a new customer is significantly higher than retaining an existing one. This project aims to build a machine learning pipeline that analyzes user behavior, financial commitments, and service friction to predict the mathematical probability of a customer abandoning the platform. By identifying "at-risk" profiles early, businesses can deploy targeted interventions.

## 📊 Data Dictionary
The model evaluates the human choice architecture based on the following input features:
* **Tenure (Months):** The duration of the user's relationship with the business. Measures habituation and loyalty.
* **Support Calls (Count):** The frequency of customer service interactions. Acts as a proxy for cognitive load and user frustration.
* **Payment Delay (Days):** Tracks billing delinquency, acting as a behavioral indicator of disengagement or financial stress.
* **Total Spend ($):** The lifetime financial value of the customer.
* **Subscription Type:** Categorical tier (e.g., Basic, Premium) indicating the user's perceived value and expectations.
* **Contract Length:** Categorical lock-in mechanism (e.g., Month-to-Month, Annual) defining the barrier to exit.

## 🔍 Key Observations & Exploratory Insights
During the Exploratory Data Analysis (EDA) and model training phases, several critical behavioral patterns emerged:
1. **The Vulnerability of Month-to-Month Contracts:** Customers without long-term commitments exhibited the highest churn volatility. Without a financial barrier to exit, they are highly sensitive to minor service disruptions.
2. **Support Calls as a Friction Threshold:** There is a direct, positive correlation between the number of support calls and churn probability. Users exceeding 3-4 calls show an exponential spike in churn risk, indicating that unresolved friction breaks user retention.
3. **Tenure as a Stabilizing Force:** Habituation is a powerful retention mechanism. Customers with a tenure exceeding 24 months are statistically highly unlikely to churn, even if they experience occasional payment delays or need support.
4. **Value Discrepancy:** High-spend users on basic subscription tiers showed elevated flight risks when encountering technical issues, suggesting a misalignment between cost and perceived service value.

## 🧠 Methodology & Machine Learning Approach
1. **Data Preprocessing:** 
   * Handled structural differences by isolating continuous and categorical variables.
   * Applied `LabelEncoder` to transform string-based categories (Contract Length, Subscription Type) into machine-readable numeric matrices.
   * Utilized `StandardScaler` to normalize continuous variables (Spend, Tenure) so higher magnitude numbers did not artificially skew the model's weight distribution.
2. **Model Selection:** 
   * Deployed a **Random Forest Classifier** (an ensemble learning method) due to its robustness against overfitting and its ability to capture non-linear relationships between behavioral variables.
3. **Probability Inference:** 
   * Rather than a simple binary output (0 or 1), the deployment utilizes the `.predict_proba()` function to output a granular risk percentage, allowing for tiered business interventions.

## 🏗️ Deployment Architecture
The transition from local notebook to cloud application was executed in three phases:
1. **Serialization:** Exported the trained Random Forest model, the fitted `StandardScaler`, and the `LabelEncoders` as `.pkl` objects using `joblib`. This ensures the web app processes new user inputs through the exact same mathematical parameters used during training.
2. **Frontend Development:** Constructed an interactive UI using `Streamlit`, mapping user input sliders and dropdowns directly to the expected Pandas DataFrame structure.
3. **Cloud Provisioning:** Deployed via Streamlit Community Cloud. Configured the environment using a `requirements.txt` file to automatically install `scikit-learn`, `pandas`, and `numpy` dependencies on the cloud server.

## ✨ Future Scope
* **Hyperparameter Tuning:** Implement GridSearchCV to optimize the Random Forest depth and estimator counts.
* **Visual Dashboards:** Integrate Matplotlib/Plotly within the Streamlit app to show real-time feature importance charts.
* **Intervention Prescriptions:** Program the logic to not just predict churn, but automatically recommend the most effective retention strategy (e.g., "Offer 20% discount" vs. "Assign VIP Support").

## 🚀 Live Links
* **Live Web Application:** https://user-friction-analyzer-ffcfmkmeww9a5kyuwjdusb.streamlit.app/
* **GitHub Repository:** https://github.com/sana499/user-friction-analyzer

## 💻 Local Installation
```bash
# Clone the repository
git clone [Your GitHub URL]

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
