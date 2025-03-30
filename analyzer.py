import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ImprovedConcentrationAnalyzer:
    def __init__(self, model_data_path=None):
        self.model_data = None
        self.user_feedback = None
        
        # Load model data if path provided
        if model_data_path:
            self.load_data(model_data_path)
    
    def load_data(self, model_data_path):
        try:
            self.model_data = pd.read_csv(model_data_path)
            self.model_data.columns = self.model_data.columns.str.strip()
            
            # Force the default columns to be 'timestamp' and 'concentration'
            if len(self.model_data.columns) >= 2:
                new_columns = self.model_data.columns.tolist()
                new_columns[0] = 'timestamp'
                new_columns[1] = 'concentration'
                self.model_data.columns = new_columns
            else:
                raise ValueError("Model data must have at least two columns: timestamp and concentration")
            
            if 'timestamp' not in self.model_data.columns or 'concentration' not in self.model_data.columns:
                raise ValueError("Model data must contain 'timestamp' and 'concentration' columns")
            
            print(f"Successfully loaded model data: {len(self.model_data)} rows")
            self._display_data_summary()
            
        except Exception as e:
            print(f"Error loading model data: {str(e)}")
            raise
    
    def _display_data_summary(self):
        if self.model_data is not None:
            print("\nModel Data Summary:")
            print(f"- Number of records: {len(self.model_data)}")
            print(f"- Columns: {', '.join(self.model_data.columns)}")
            print(f"- Concentration range: {self.model_data['concentration'].min():.2f} to {self.model_data['concentration'].max():.2f}")
            print(f"- Concentration mean: {self.model_data['concentration'].mean():.2f}")
            print(f"- Timestamp range: {self.model_data['timestamp'].min()} to {self.model_data['timestamp'].max()}")
        
        if self.user_feedback is not None:
            print("\nUser Feedback Summary:")
            print(f"- Number of records: {len(self.user_feedback)}")
            print(f"- Columns: {', '.join(self.user_feedback.columns)}")
            print(f"- Concentration range: {self.user_feedback['concentration'].min():.2f} to {self.user_feedback['concentration'].max():.2f}")
            print(f"- Concentration mean: {self.user_feedback['concentration'].mean():.2f}")
            print(f"- Timestamp range: {self.user_feedback['timestamp'].min()} to {self.user_feedback['timestamp'].max()}")
    
    def compute_accuracy_metrics(self):
        if self.model_data is None or self.user_feedback is None:
            raise ValueError("Model data and user feedback must be loaded before computing metrics")
            
        if 'timestamp' in self.model_data.columns and 'timestamp' in self.user_feedback.columns:
            merged_data = pd.DataFrame()
            for _, user_row in self.user_feedback.iterrows():
                user_time = user_row['timestamp']
                time_diffs = abs(self.model_data['timestamp'] - user_time)
                closest_idx = time_diffs.idxmin()
                model_value = self.model_data.loc[closest_idx, 'concentration']
                new_row = {
                    'timestamp': user_time,
                    'user_concentration': user_row['concentration'],
                    'model_concentration': model_value,
                    'time_diff': time_diffs.min()
                }
                merged_data = pd.concat([merged_data, pd.DataFrame([new_row])], ignore_index=True)
            
            if len(merged_data) > 0:
                metrics = {
                    'mean_absolute_error': mean_absolute_error(
                        merged_data['user_concentration'], merged_data['model_concentration']),
                    'mean_squared_error': mean_squared_error(
                        merged_data['user_concentration'], merged_data['model_concentration']),
                    'r2_score': r2_score(
                        merged_data['user_concentration'], merged_data['model_concentration']) 
                        if len(merged_data) > 1 else 0,
                    'correlation': np.corrcoef(
                        merged_data['user_concentration'], merged_data['model_concentration'])[0, 1] 
                        if len(merged_data) > 1 else 0,
                    'average_time_diff': merged_data['time_diff'].mean()
                }
                return metrics, merged_data
            else:
                print("Warning: No matching points between datasets")
                return {}, pd.DataFrame()
        else:
            print("Warning: Timestamp columns required in both datasets for metrics calculation")
            return {}, pd.DataFrame()
    
    def visualize_comparison(self, title="Concentration Scores - Model vs User Feedback"):
        if self.model_data is None or self.user_feedback is None:
            raise ValueError("Model data and user feedback must be loaded before visualization")
            
        metrics, merged_data = self.compute_accuracy_metrics()
        plt.figure(figsize=(14, 8))
        
        model_data_sorted = self.model_data.sort_values('timestamp')
        user_feedback_sorted = self.user_feedback.sort_values('timestamp')
        
        plt.plot(model_data_sorted['timestamp'], model_data_sorted['concentration'], 
                 'b-', label='Model Prediction', alpha=0.7)
        plt.scatter(user_feedback_sorted['timestamp'], user_feedback_sorted['concentration'], 
                    color='red', s=100, label='User Feedback')
        
        for i, row in user_feedback_sorted.iterrows():
            plt.annotate(f"{row['concentration']:.2f}", 
                         (row['timestamp'], row['concentration']),
                         xytext=(0, 10),
                         textcoords='offset points',
                         fontsize=9,
                         ha='center')
        
        if not merged_data.empty:
            for _, row in merged_data.iterrows():
                plt.plot([row['timestamp'], row['timestamp']], 
                         [row['user_concentration'], row['model_concentration']], 
                         'k--', alpha=0.5)
                mid_point = (row['user_concentration'] + row['model_concentration']) / 2
                error = abs(row['user_concentration'] - row['model_concentration'])
                plt.annotate(f"Δ={error:.2f}", 
                             (row['timestamp'], mid_point),
                             xytext=(5, 0),
                             textcoords='offset points',
                             fontsize=8)
        
        if metrics:
            metrics_text = (
                f"MAE: {metrics.get('mean_absolute_error', 'N/A'):.4f}\n"
                f"MSE: {metrics.get('mean_squared_error', 'N/A'):.4f}\n"
                f"R²: {metrics.get('r2_score', 'N/A'):.4f}\n"
                f"Correlation: {metrics.get('correlation', 'N/A'):.4f}"
            )
            plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         fontsize=10, ha='left', va='top')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Concentration Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_model_with_moving_average(self, window_size=30):
        if self.model_data is None:
            raise ValueError("Model data must be loaded before plotting")
        
        # Ensure data is sorted by timestamp
        model_data_sorted = self.model_data.sort_values('timestamp')
        time_seconds = model_data_sorted['timestamp'].values
        predictions = model_data_sorted['concentration'].values
        
        # Calculate moving average using uniform filter
        moving_avg = uniform_filter1d(predictions, size=window_size, mode='nearest')
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_seconds, predictions, 'b-', alpha=0.5, label='Original Model Prediction')
        plt.plot(time_seconds, moving_avg, 'g-', linewidth=2, label=f'Moving Average (window={window_size})')
        
        # If user feedback is available, plot it
        if self.user_feedback is not None and not self.user_feedback.empty:
            feedback = self.user_feedback.sort_values('timestamp')
            feedback_times = feedback['timestamp'].values
            feedback_values = feedback['concentration'].values
            plt.scatter(feedback_times, feedback_values, color='red', s=50, label='User Feedback')
            for t, v in zip(feedback_times, feedback_values):
                plt.annotate(f"{v:.2f}", (t, v), xytext=(0, 7),
                             textcoords='offset points', ha='center')
            
            # Compute metrics between moving average and user feedback interpolated on the model timeline
            interp_vals = np.interp(feedback_times, time_seconds, moving_avg)
            mae = np.mean(np.abs(interp_vals - feedback_values))
            mse = np.mean((interp_vals - feedback_values) ** 2)
            correlation = np.corrcoef(interp_vals, feedback_values)[0, 1] if len(feedback_values) > 1 else 0
            
            plt.text(0.02, 0.95, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nCorrelation: {correlation:.4f}",
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Concentration Score')
        plt.title('Model Prediction with Moving Average')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.show()
        
        return moving_avg
    
    def generate_report(self):
        metrics, merged_data = self.compute_accuracy_metrics()
        if not metrics:
            return "Unable to generate report: No matching data points between datasets."
        
        report = f"""
Concentration Monitoring Accuracy Report
----------------------------------------
Data Summary:
- Model data points: {len(self.model_data)}
- User feedback points: {len(self.user_feedback)}
- Matched points for analysis: {len(merged_data)}

Accuracy Metrics:
- Mean Absolute Error: {metrics.get('mean_absolute_error', 'N/A'):.4f}
- Mean Squared Error: {metrics.get('mean_squared_error', 'N/A'):.4f}
- R² Score: {metrics.get('r2_score', 'N/A'):.4f}
- Correlation: {metrics.get('correlation', 'N/A'):.4f}
- Average time difference between matched points: {metrics.get('average_time_diff', 'N/A'):.2f} seconds

Interpretation:
- MAE: Average error between model predictions and user feedback
- MSE: Average squared error, penalizing larger errors more heavily
- R²: Proportion of variance in user feedback explained by the model
- Correlation: Strength and direction of the relationship

Point-by-Point Comparison:
-------------------------
"""
        if not merged_data.empty:
            for _, row in merged_data.iterrows():
                error = abs(row['user_concentration'] - row['model_concentration'])
                report += (f"Time {row['timestamp']}s: "
                           f"User={row['user_concentration']:.2f}, "
                           f"Model={row['model_concentration']:.2f}, "
                           f"Error={error:.2f}\n")
        
        return report

def main():
    model_data_path = r"C:\Users\richa\Downloads\Default Dataset (5).csv"
    
    try:
        analyzer = ImprovedConcentrationAnalyzer(model_data_path)
        
        # Load predefined user feedback data (manual input)
        user_data = collect_user_feedback()
        analyzer.user_feedback = pd.DataFrame(user_data)
        
        analyzer._display_data_summary()
        
        # Generate and print the accuracy report
        report = analyzer.generate_report()
        print(report)
        
        # Visualize comparison between model predictions and user feedback
        analyzer.visualize_comparison("Model vs. Predefined User Feedback")
        
        analyzer.plot_model_with_moving_average(window_size=30)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

def collect_user_feedback():

    # Input with own CSV data 
    timestamps = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    concentration_values = [0.8, 0.8, 0.8, 0.8, 0.9, 0.8, 0.8, 0.9, 0.9, 0.9]
    
    print("\nUsing predefined user feedback data:")
    for t, c in zip(timestamps, concentration_values):
        print(f"Time={t}s, Concentration={c}")
    
    return {'timestamp': timestamps, 'concentration': concentration_values}

if __name__ == '__main__':
    main()
