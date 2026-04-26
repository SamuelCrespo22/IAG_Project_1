import pandas as pd
import os

def analyze_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # --- MULTIMETRIC ANALYSIS ---
    # Calculate the ranking for each important metric.
    # rank(ascending=True) -> rank 1 for the lowest value (good for FID, KID)
    # rank(ascending=False) -> rank 1 for the highest value (good for IS)
    df['FID_Rank'] = df['FID_mean'].rank(ascending=True)
    df['KID_Rank'] = df['KID_mean'].rank(ascending=True)
    df['IS_Rank'] = df['IS_mean'].rank(ascending=False)

    # Calculate an "Overall Score" based on the average of the rankings. Lower is better.
    df['Overall_Rank_Score'] = df[['FID_Rank', 'KID_Rank', 'IS_Rank']].mean(axis=1)
    
    # Sort by overall score
    df_sorted_overall = df.sort_values(by='Overall_Rank_Score', ascending=True)

    # The best model is the first in the list after sorting
    best_model = df_sorted_overall.iloc[0]
    
    print("=" * 65)
    print("🏆 BEST MODEL (Combined Analysis of FID, KID and IS)")
    print("=" * 65)
    print(f"Experiment Name: {best_model['Experiment_Name']}")
    print(f"Model Family:    {best_model['Model'].upper()}")
    print("-" * 65)
    print("Main Metrics and Rankings:")
    print(f"  - FID:    {best_model['FID_mean']:.2f} (Rank: {int(best_model['FID_Rank'])} of {len(df)})")
    print(f"  - KID:    {best_model['KID_mean']:.4f} (Rank: {int(best_model['KID_Rank'])} of {len(df)})")
    print(f"  - IS:     {best_model['IS_mean']:.2f} (Rank: {int(best_model['IS_Rank'])} of {len(df)})")
    print("-" * 65)
    print(f"Hyperparameters:     Epochs: {best_model['Epochs']} | Batch Size: {best_model['Batch_Size']} | LR: {best_model['LR']} | Latent Dim: {best_model['Latent_Dim']}")
    print(f"Total Training Time: {best_model['Total_Time_s']:.2f} seconds")
    print(f"Notes:               {best_model['Notes']}")
    print("=" * 65)
    
    print("\n📊 TOP 5 MODELS (Combined Ranking):")
    # Select and format relevant columns for the table
    display_cols = ['Experiment_Name', 'Model', 'Overall_Rank_Score', 'FID_Rank', 'KID_Rank', 'IS_Rank']
    top5_df = df_sorted_overall[display_cols].head(5).copy()
    top5_df['Overall_Rank_Score'] = top5_df['Overall_Rank_Score'].round(1)
    top5_df[['FID_Rank', 'KID_Rank', 'IS_Rank']] = top5_df[['FID_Rank', 'KID_Rank', 'IS_Rank']].astype(int)
    
    print(top5_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("🏆 BEST MODEL PER FAMILY (Based on Overall Rank Score)")
    print("=" * 65)
    
    # Group by Model and take the first row of each group (since it's already sorted by Overall_Rank_Score)
    best_per_family = df_sorted_overall.groupby('Model').first().reset_index()
    
    for _, row in best_per_family.iterrows():
        print(f"\nModel Family: {row['Model'].upper()}")
        print(f"  Experiment:     {row['Experiment_Name']}")
        print(f"  Overall Score:  {row['Overall_Rank_Score']:.1f}")
        print(f"  Metrics:        FID: {row['FID_mean']:.2f} | KID: {row['KID_mean']:.4f} | IS: {row['IS_mean']:.2f}")
        print(f"  Hyperparams:    Epochs: {row['Epochs']} | Batch Size: {row['Batch_Size']} | LR: {row['LR']} | Latent Dim: {row['Latent_Dim']}")
    print("=" * 65)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "phase1_results.csv")
    analyze_results(csv_path)
