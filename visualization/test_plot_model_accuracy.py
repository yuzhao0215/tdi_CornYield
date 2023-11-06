import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

accuracy_df = pd.read_csv('../data/accuracy_df.csv', index_col=None)

accuracy_df = accuracy_df[accuracy_df['test_r2'] > 0]
accuracy_df = accuracy_df[accuracy_df['train_mape'] < 1]

avg_accuracy_df = accuracy_df.groupby('current_actual_month').agg(np.mean)
avg_accuracy_df['growing_month'] = avg_accuracy_df.index
avg_accuracy_df['growing_month'] = avg_accuracy_df['growing_month'].apply(lambda x: (x+1) %12 +1)
avg_accuracy_df.sort_values('growing_month', inplace=True)

# plt.plot(avg_accuracy_df['growing_month'], avg_accuracy_df['test_mape'])
plt.plot(avg_accuracy_df['growing_month'], avg_accuracy_df['train_r2'], label='train_r2')
plt.plot(avg_accuracy_df['growing_month'], avg_accuracy_df['test_r2'], label='test_r2')
plt.xticks(range(1, 13), labels=['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', "Jul", 'Aug', 'Sep', 'Oct'])
plt.ylabel("r2 score")
plt.xlabel("Model in Month")
plt.legend()
plt.show()
print(avg_accuracy_df)