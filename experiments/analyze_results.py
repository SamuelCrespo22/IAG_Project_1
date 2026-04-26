import pandas as pd
import os

def analyze_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Erro: O ficheiro {csv_path} não foi encontrado.")
        return

    # Carregar o ficheiro CSV
    df = pd.read_csv(csv_path)

    # --- ANÁLISE MULTIMÉTRICA ---
    # Calcular o ranking para cada métrica importante.
    # rank(ascending=True) -> rank 1 para o menor valor (bom para FID, KID)
    # rank(ascending=False) -> rank 1 para o maior valor (bom para IS)
    df['FID_Rank'] = df['FID_mean'].rank(ascending=True)
    df['KID_Rank'] = df['KID_mean'].rank(ascending=True)
    df['IS_Rank'] = df['IS_mean'].rank(ascending=False)

    # Calcular um "Score Geral" baseado na média dos rankings. Menor é melhor.
    df['Overall_Rank_Score'] = df[['FID_Rank', 'KID_Rank', 'IS_Rank']].mean(axis=1)
    
    # Ordenar pelo score geral
    df_sorted_overall = df.sort_values(by='Overall_Rank_Score', ascending=True)

    # O melhor modelo é o primeiro da lista após a ordenação
    best_model = df_sorted_overall.iloc[0]
    
    print("=" * 65)
    print("🏆 MELHOR MODELO (Análise Combinada de FID, KID e IS)")
    print("=" * 65)
    print(f"Nome da Experiência: {best_model['Experiment_Name']}")
    print(f"Família do Modelo:   {best_model['Model'].upper()}")
    print("-" * 65)
    print("Métricas Principais e Rankings:")
    print(f"  - FID:    {best_model['FID_mean']:.2f} (Rank: {int(best_model['FID_Rank'])} de {len(df)})")
    print(f"  - KID:    {best_model['KID_mean']:.4f} (Rank: {int(best_model['KID_Rank'])} de {len(df)})")
    print(f"  - IS:     {best_model['IS_mean']:.2f} (Rank: {int(best_model['IS_Rank'])} de {len(df)})")
    print("-" * 65)
    print(f"Hiperparâmetros:     Epochs: {best_model['Epochs']} | Batch Size: {best_model['Batch_Size']} | LR: {best_model['LR']} | Latent Dim: {best_model['Latent_Dim']}")
    print(f"Tempo Total Treino:  {best_model['Total_Time_s']:.2f} segundos")
    print(f"Notas:               {best_model['Notes']}")
    print("=" * 65)
    
    print("\n📊 TOP 5 MODELOS (Ranking Combinado):")
    # Selecionar e formatar colunas relevantes para a tabela
    display_cols = ['Experiment_Name', 'Model', 'Overall_Rank_Score', 'FID_Rank', 'KID_Rank', 'IS_Rank']
    top5_df = df_sorted_overall[display_cols].head(5).copy()
    top5_df['Overall_Rank_Score'] = top5_df['Overall_Rank_Score'].round(1)
    top5_df[['FID_Rank', 'KID_Rank', 'IS_Rank']] = top5_df[['FID_Rank', 'KID_Rank', 'IS_Rank']].astype(int)
    
    print(top5_df.to_string(index=False))

if __name__ == "__main__":
    csv_path = "/Users/samuel/Data_science/IAG_Project_1/phase1_results.csv"
    analyze_results(csv_path)
