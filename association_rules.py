# Step 1: Import Libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Data Preprocessing
# Load the dataset
df = pd.read_csv('OnlineRetail.csv', header=None)

# Rename the column for clarity
df.columns = ['Items']

# Split the items into lists
df['Items'] = df['Items'].apply(lambda x: x.split(','))

# Create the Basket Format
# Create a TransactionEncoder object
te = TransactionEncoder()
te_ary = te.fit(df['Items']).transform(df['Items'])

# Create a DataFrame
basket = pd.DataFrame(te_ary, columns=te.columns_)

# Step 3: Association Rule Mining
# Implement the Apriori Algorithm
frequent_items = apriori(basket, min_support=0.01, use_colnames=True)

# Generate Association Rules
rules = association_rules(frequent_items, metric='lift', min_threshold=1)

# Step 4: Analysis and Interpretation
# Sort and analyze the rules
rules.sort_values(by='lift', ascending=False, inplace=True)
print("Top 10 Association Rules:")
print(rules.head(10))

# Step 5: Visualize the Rules
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rules, x='support', y='confidence', size='lift', sizes=(20, 200), hue='antecedents')
plt.title('Association Rules - Support vs Confidence')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.legend(title='Antecedents', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
