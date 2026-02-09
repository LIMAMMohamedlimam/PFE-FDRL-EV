import numpy as np
import matplotlib.pyplot as plt

class EvalMetrics:
    def __init__(self , run_name='metrics_plot'):
        self.run_name = run_name
        # Convergence
        self.episode_rewards = []  # Liste des récompenses totales par épisode
        
        # Grid Stability
        self.grid_loads = []       # Historique de la charge totale (MW) par pas de temps
        self.grid_power_changes = [] # Historique des variations |P_t - P_{t-1}|
        
        # Generalization
        self.test_rewards = []     # Récompenses sur données de test

        self.satisfaction_history = [] # Historique des pénalités de satisfaction (pour analyse)

    def log_satisfaction(self, agent_satisfactions):
        """
        Enregistre la moyenne de satisfaction des agents pour l'épisode.
        agent_satisfactions : liste de ratios (ex: [1.0, 0.8, 1.0])
        """
        avg_sat = np.mean(agent_satisfactions)
        self.satisfaction_history.append(avg_sat)
        
    def log_step(self, total_grid_load):
        """
        Enregistre la charge totale du réseau à un pas de temps t.
        Utilisé pour calculer la stabilité.
        """
        if self.grid_loads:
            # Calcule le changement de puissance par rapport au pas précédent
            delta_p = total_grid_load - self.grid_loads[-1]
            self.grid_power_changes.append(delta_p)
        
        self.grid_loads.append(total_grid_load)

    def log_episode(self, total_reward, mode='train'):
        """
        Enregistre la récompense cumulée à la fin d'un épisode.
        """
        if mode == 'train':
            self.episode_rewards.append(total_reward)
        else:
            self.test_rewards.append(total_reward)

    def compute_stability_metric(self):
        """
        Calcule la stabilité du réseau sigma_g.
        Définie comme l'écart-type des changements de puissance.
        Plus bas = Plus stable (moins de pics soudains).
        """
        if not self.grid_power_changes:
            return 0.0
        # sigma_g = std(Delta P)
        sigma_g = np.std(self.grid_power_changes)
        return sigma_g

    def plot_metrics(self):
        """Génère les graphiques pour l'analyse visuelle."""
        plt.figure(figsize=(14, 4))
        
        # 1. Convergence
        plt.subplot(1, 4, 1)
        plt.plot(self.episode_rewards, label='Train Reward')
        plt.xlabel('Épisodes')
        plt.ylabel('Récompense Cumulée')
        plt.title('Convergence (Apprentissage)')
        plt.grid(True)
        
        # 2. Stabilité Grid
        plt.subplot(1, 4, 2)
        # On trace la courbe de charge pour visualiser les pics
        plt.plot(self.grid_loads[:100], label='Charge Réseau (MW)') # Zoom sur les 100 premiers pas
        plt.title(f'Stabilité Grid ($\sigma_g$={self.compute_stability_metric():.3f})')
        plt.xlabel('Temps')
        plt.grid(True)
        
        # 3. Généralisation (Si dispo)
        if self.test_rewards:
            plt.subplot(1, 4, 3)
            plt.boxplot([self.episode_rewards[-10:], self.test_rewards], labels=['Train (Fin)', 'Test'])
            plt.title('Généralisation')
        
        # 4. Satisfaction History
        if self.satisfaction_history:
            plt.subplot(1, 4, 4)
            plt.plot(self.satisfaction_history, color='green')
            plt.axhline(1.0, color='r', linestyle='--', label='Objectif (100%)')
            plt.title('Satisfaction Client Moyenne')
            plt.xlabel('Épisodes')
            plt.ylabel('Ratio (SOC_final / SOC_req)')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(f'results/{self.run_name}.png')
        plt.show()