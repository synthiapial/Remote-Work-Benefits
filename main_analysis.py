def run_analysis():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    # Load data
    df = pd.read_csv("../data/Impact_of_Remote_Work_on_Mental_Health.csv")
    df.dropna(subset=['Work_Location', 'Productivity_Change', 'Mental_Health_Condition'], inplace=True)

    # Feature engineering
    df['Remote'] = df['Work_Location'].apply(lambda x: 1 if 'remote' in x.lower() else 0)
    df['Satisfied'] = df['Satisfaction_with_Remote_Work'].map({'Satisfied': 1, 'Neutral': 0, 'Unsatisfied': -1})
    df['Productivity_Num'] = df['Productivity_Change'].map({'Increase': 1, 'No Change': 0, 'Decrease': -1})

    # EDA
    sns.boxplot(x='Remote', y='Productivity_Num', data=df)
    plt.title('Productivity Change: Remote vs On-Site')
    plt.tight_layout()
    plt.show()

    sns.histplot(data=df, x='Social_Isolation_Rating', hue='Remote', kde=True)
    plt.title("Social Isolation by Work Type")
    plt.tight_layout()
    plt.show()

    # T-test
    remote_group = df[df['Remote'] == 1]['Productivity_Num']
    onsite_group = df[df['Remote'] == 0]['Productivity_Num']
    t_stat, p_value = ttest_ind(remote_group, onsite_group)
    print(f"T-Test: t-stat = {t_stat:.3f}, p = {p_value:.5f}")

    # Modeling
    X = df[['Remote', 'Productivity_Num', 'Social_Isolation_Rating']]
    y = df['Satisfied']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = clf.feature_importances_
    features = X.columns
    sns.barplot(x=importances, y=features)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
