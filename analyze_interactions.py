import json

import numpy as np
import pandas as pd

# --- CONFIGURACIÓN ---
INTERACTIONS_LOG_PATH = "real_interactions_rating.json"

def analyze_interactions(log_path):
    """
    Carga y analiza las interacciones guardadas para mostrar un resumen.
    """
    print(f"--- Análisis de la Sesión de Interacción desde '{log_path}' ---")
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            interactions = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo de log no encontrado en '{log_path}'.")
        return
    except Exception as e:
        print(f"❌ ERROR: No se pudo leer el archivo de log. Detalles: {e}")
        return

    if not interactions:
        print("El archivo de log está vacío. No hay nada que analizar.")
        return

    # Usamos pandas para facilitar el análisis
    df = pd.DataFrame(interactions)
    
    # Extraer y formatear datos para una mejor visualización
    df['V_action'] = df['action'].apply(lambda x: f"{x[0]:.2f}")
    df['A_action'] = df['action'].apply(lambda x: f"{x[1]:.2f}")
    df['V_state'] = df['state'].apply(lambda x: f"{x[0]:.2f}")
    df['A_state'] = df['state'].apply(lambda x: f"{x[1]:.2f}")
    df['V_next_state'] = df['next_state'].apply(lambda x: f"{x[0]:.2f}")
    df['A_next_state'] = df['next_state'].apply(lambda x: f"{x[1]:.2f}")
    
    print("\n--- Resumen de la Sesión ---")
    print(df[['V_state', 'A_state', 'V_action', 'A_action', 'V_next_state', 'A_next_state', 'reward', 'done']].round(2))

    # --- Estadísticas Clave ---
    total_reward = df['reward'].sum()
    avg_reward = df['reward'].mean()
    num_steps = len(df)
    was_successful = df['done'].any()

    print("\n--- Estadísticas Finales ---")
    print(f"Pasos totales en la sesión: {num_steps}")
    print(f"Recompensa total acumulada: {total_reward:.3f}")
    print(f"Recompensa media por paso: {avg_reward:.3f}")
    print(f"Sesión completada con éxito: {'Sí' if was_successful else 'No'}")
    
    # Análisis de correlación simple
    # Necesitamos extraer el feedback (like/dislike) que no guardamos explícitamente,
    # pero podemos inferirlo de la recompensa.
    # Esta es una simplificación, pero útil para el análisis.
    
    print("\nEste resumen te proporciona los datos para analizar la estrategia del agente")
    print("y discutir su rendimiento en tu TFM.")


if __name__ == '__main__':
    analyze_interactions(INTERACTIONS_LOG_PATH)