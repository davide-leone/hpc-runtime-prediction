import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

class PrintSummary:

    def show_prediction_summary(self, df):
        '''

        Calculates and shows the prediction summary
        
        '''
        df['Prediction_Type'] = df.apply(
            lambda row: 'Exact' if row['Predicted_Runtime'] == row['Actual_Runtime'] 
                        else 'Overestimate' if row['Predicted_Runtime'] > row['Actual_Runtime'] 
                        else 'Underestimate', axis=1)
        
        # Calculate percentages and average error for each category
        summary_table = df.groupby('Prediction_Type').agg(
            Count=('Job_ID', 'count'),
            Percentage=('Job_ID', lambda x: (len(x) / len(df)) * 100),
            Average_Error=('Prediction_Error', 'mean')
        ).reset_index()
        
        return pd.DataFrame(summary_table)

 
    def show_runtime_frequency(self, df):
        ''' 
        
        Count unique predicted runtime and get their frequency
        
        '''
        unique_requested_runtimes = df['Requested_Runtime'].nunique()
        print(f"Number of unique requested runtimes: {unique_requested_runtimes}")
        requested_runtime_frequency = df['Requested_Runtime'].value_counts().reset_index()
        requested_runtime_frequency.columns = ['Requested_Runtime', 'Frequency']
    
        unique_actual_runtimes = df['Actual_Runtime'].nunique()
        print(f"Number of unique actual runtimes: {unique_actual_runtimes}")
        actual_runtime_frequency = df['Actual_Runtime'].value_counts().reset_index()
        actual_runtime_frequency.columns = ['Actual_Runtime', 'Frequency']
        
        unique_predicted_runtimes = df['Predicted_Runtime'].nunique()
        print(f"Number of unique predicted runtimes: {unique_predicted_runtimes}")
        predicted_runtime_frequency = df['Predicted_Runtime'].value_counts().reset_index()
        predicted_runtime_frequency.columns = ['Predicted_Runtime', 'Frequency']
        
        # Return the frequency tables
        return (requested_runtime_frequency, actual_runtime_frequency, predicted_runtime_frequency)


    def print_metrics(self, df):
        mae = mean_absolute_error(df['Actual_Runtime'], df['Predicted_Runtime'])
        rmse = np.sqrt(mean_squared_error(df['Actual_Runtime'], df['Predicted_Runtime']))
        
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")


class ShowPlots:

    def show_runtime_histogram(self, col1, col2, label1, label2, path=''):
        plt.figure(figsize=(12, 6))
        sns.histplot(col1, kde=True, color='blue', label=label1, alpha=0.5, binwidth=300)
        sns.histplot(col2, kde=True, color='orange', label=label2, alpha=0.5, binwidth=300)
        plt.title('Histogram of Runtimes', fontsize=16)
        plt.xlabel('Runtime', fontsize=14)
        plt.xticks(np.arange(0, 86401, 3600), rotation=45)
        plt.ylabel('Frequency',  fontsize=14)
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if path != '':
            plt.savefig(path)
        plt.show()


    def show_runtime_histogram_limited(self, col1, col2, label1, label2, path=''):
        plt.figure(figsize=(12, 6))
        sns.histplot(col1, kde=True, color='blue', label=label1, alpha=0.5, binwidth=300)
        sns.histplot(col2, kde=True, color='orange', label=label2, alpha=0.5, binwidth=300)
    
        plt.title('Histogram of Runtimes (0–7200)', fontsize=16)
        plt.xlabel('Runtime', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
    
        # Limit x-axis to [0, 7200] seconds
        plt.xlim(0, 7200)
        plt.xticks(np.arange(0, 7201, 300), rotation=45)  # ticks every 5 minutes
    
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    
        if path != '':
            plt.savefig(path)
    
        plt.show()

    def show_scatter_plot(self, df):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df['Actual_Runtime'], y=df['Predicted_Runtime'], label='Predicted vs Actual', color='orange')
        plt.plot([df['Actual_Runtime'].min(), df['Actual_Runtime'].max()],
                 [df['Actual_Runtime'].min(), df['Actual_Runtime'].max()], color='blue', linestyle='--', label='Ideal')
        plt.title('Predicted vs Actual Runtime')
        plt.xlabel('Actual Runtime')
        plt.ylabel('Predicted Runtime')
        plt.legend()
        plt.show()
