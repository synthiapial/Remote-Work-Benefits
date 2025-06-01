import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ----------------------------
# 1. Load the Dataset
# ----------------------------
df = pd.read_csv("../data/Impact_of_Remote_Work_on_Mental_Health.csv")

# ----------------------------
# 2. Data Cleaning
# ----------------------------
# Drop rows with missing values in key columns
df.dropna(subset=['Work_Location', 'Productivity_Change', 'Mental_Health_Condition'], inplace=True)

# ----------------------------
# 3. Feature Engineering
# ----------------------------

# Create a binary column for remote workers
df['Remote'] = df['Work_Location'].apply(lambda x: 1 if 'remote' in x.lower() else 0)

# Convert satisfaction to numerical values
df['Satisfied'] = df['Satisfaction_with_Remote_Work'].map({
    'Satisfied': 1,
    'Neutral': 0,
    'Unsatisfied': -1
})

# Convert productivity change to numerical values
df['Productivity_Num'] = df['Productivity_Change'].map({
    'Increase': 1,
    'No Change': 0,
    'Decrease': -1
})

# ----------------------------
# 4. Exploratory Data Analysis
# ----------------------------

# Boxplot: Productivity by Work Location
sns.boxplot(x='Remote', y='Productivity_Num', data=df)
plt.title('Productivity Change: Remote vs On-Site')
plt.xlabel("Remote (1 = Remote, 0 = Onsite/Hybrid)")
plt.ylabel("Productivity Change (-1 = Decrease, 1 = Increase)")
plt.tight_layout()
plt.show()

# Histogram: Social Isolation by Work Location
sns.histplot(data=df, x='Social_Isolation_Rating', hue='Remote', kde=True, palette='Set2')
plt.title("Social Isolation by Work Type")
plt.xlabel("Social Isolation Rating (1 = Low, 5 = High)")
plt.tight_layout()
plt.show()

# ----------------------------
# 5. Statistical Testing
# ----------------------------

# T-test: Does remote work significantly affect productivity?
remote_group = df[df['Remote'] == 1]['Productivity_Num']
onsite_group = df[df['Remote'] == 0]['Productivity_Num']
t_stat, p_value = ttest_ind(remote_group, onsite_group)

print(f"📊 T-Test Results: t-statistic = {t_stat:.3f}, p-value = {p_value:.5f}")
if p_value < 0.05:
    print("✅ Significant difference in productivity between remote and onsite workers.")
else:
    print("❌ No significant difference in productivity between remote and onsite workers.")

# ----------------------------
# 6. Machine Learning Model: Predict Satisfaction
# ----------------------------

# Define features and target
X = df[['Remote', 'Productivity_Num', 'Social_Isolation_Rating']]
y = df['Satisfied']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# 7. Feature Importance
# ----------------------------

importances = clf.feature_importances_
features = X.columns

# Plot feature importance
sns.barplot(x=importances, y=features)
plt.title('Feature Importance for Predicting Remote Work Satisfaction')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
