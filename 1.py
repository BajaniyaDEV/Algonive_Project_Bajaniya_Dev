import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("customers.csv")

# Display the first few rows
print("Sample Data:\n", df.head())

# Select relevant features for segmentation
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize the Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='Set1', s=100)
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.show()

# Save the results
df.to_csv("1.csv", index=False)
print("Segmented customer data saved as segmented_customers.csv")
