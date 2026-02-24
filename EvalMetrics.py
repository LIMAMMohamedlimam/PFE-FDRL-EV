import numpy as np
import matplotlib.pyplot as plt
import json
import os

class EvalMetrics:
    def __init__(self, run_name='metrics_plot', config=None):
        """
        Args:
            run_name (str): Name of the file for the plot.
            config (dict): Dictionary containing all simulation hyperparameters.
        """
        self.run_name = run_name
        self.config = config if config is not None else {}
        
        # 1. Convergence (Reward)
        self.episode_rewards = []
        
        # 2. Economics (Cost)
        self.episode_costs = [] 
        
        # 3. Grid Physics
        self.grid_loads = []       
        self.grid_power_changes = [] 
        
        # 4. Generalization
        self.test_rewards = []     

        # 5. User Satisfaction
        self.satisfaction_history = [] 

    def log_satisfaction(self, agent_satisfactions):
        avg_sat = np.mean(agent_satisfactions)
        self.satisfaction_history.append(avg_sat)

    def log_cost(self, total_episode_cost):
        self.episode_costs.append(total_episode_cost)
        
    def log_step(self, total_grid_load):
        if self.grid_loads:
            delta_p = total_grid_load - self.grid_loads[-1]
            self.grid_power_changes.append(delta_p)
        self.grid_loads.append(total_grid_load)

    def log_episode(self, total_reward, mode='train'):
        if mode == 'train':
            self.episode_rewards.append(total_reward)
        else:
            self.test_rewards.append(total_reward)

    def compute_stability_metric(self):
        if not self.grid_power_changes:
            return 0.0
        return np.std(self.grid_power_changes)

    def _save_config_to_json(self, filename):
        """
        Saves the run configuration to a central JSON registry.
        """
        json_path = 'results/simulation_registry.json'
        
        # Ensure results directory exists
        if not os.path.exists('results'):
            os.makedirs('results')

        # Load existing data
        data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}

        # Add metrics summary to config
        run_summary = self.config.copy()
        run_summary.update({
            "final_train_reward_avg_5": np.mean(self.episode_rewards[-5:]) if len(self.episode_rewards) >= 5 else 0,
            "final_test_reward": self.test_rewards[-1] if self.test_rewards else 0,
            "grid_stability_sigma": float(self.compute_stability_metric()),
            "timestamp": self.run_name.split('_')[-2] + "_" + self.run_name.split('_')[-1] if "_" in self.run_name else "N/A"
        })

        # Update and save
        data[filename] = run_summary
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"-> Configuration saved to {json_path}")

    def plot_metrics(self):
        """
        Generates plots and saves config to JSON.
        """
        plt.figure(figsize=(15, 10)) 
        
        # --- ROW 1: Learning & Economics ---
        plt.subplot(2, 3, 1)
        plt.plot(self.episode_rewards, label='Train Reward', color='blue')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Convergence (Reward)')
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        if self.episode_costs:
            plt.plot(self.episode_costs, label='Avg Cost', color='orange')
            plt.xlabel('Episodes')
            plt.ylabel('Cost ($)')
            plt.title('Energy Cost Minimization')
            plt.grid(True)

        plt.subplot(2, 3, 3)
        if self.satisfaction_history:
            plt.plot(self.satisfaction_history, color='green')
            plt.axhline(1.0, color='r', linestyle='--', label='Target')
            plt.xlabel('Episodes')
            plt.ylabel('SOC Ratio')
            plt.title('Client Satisfaction')
            plt.legend()
            plt.grid(True)

        # --- ROW 2: Grid & Validation ---
        plt.subplot(2, 3, 4)
        display_steps = min(len(self.grid_loads), 100)
        plt.plot(self.grid_loads[:display_steps], color='purple')
        plt.title(f'Grid Stability ($\sigma_g$={self.compute_stability_metric():.3f})')
        plt.xlabel('Time (Steps)')
        plt.ylabel('Load (MW)')
        plt.grid(True)
        
        if self.test_rewards:
            plt.subplot(2, 3, 5)
            train_sample = self.episode_rewards[-20:] if len(self.episode_rewards) > 20 else self.episode_rewards
            plt.boxplot([train_sample, self.test_rewards], labels=['Train (End)', 'Test'])
            plt.title('Generalization Gap')
            plt.grid(True)

        plt.tight_layout()
        
        # Save Image
        filename = f'{self.run_name}.png'
        save_path = f'results/{filename}'
        if not os.path.exists('results'):
            os.makedirs('results')
            
        plt.savefig(save_path)
        print(f"-> Plot saved to {save_path}")
        
        # Save Config to JSON
        self._save_config_to_json(filename)
        
        plt.show()