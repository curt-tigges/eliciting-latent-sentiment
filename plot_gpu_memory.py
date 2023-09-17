#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
# Read the CSV file
data = pd.read_csv('gpu_memory_usage.csv', parse_dates=['Timestamp'])
#%%
data[' GPU Memory Usage (MiB)'].describe()
#%%
# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(data['Timestamp'], data[' GPU Memory Usage (MiB)'], marker='o', linestyle='-')
plt.title('GPU Memory Usage Over Time')
plt.xlabel('Timestamp')
plt.ylabel('GPU Memory Usage (MiB)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# %%
