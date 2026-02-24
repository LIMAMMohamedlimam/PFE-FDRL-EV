import numpy as np
import matplotlib.pyplot as plt
import json
import csv
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

    # ------------------------------------------------------------------
    # CSV Persistence
    # ------------------------------------------------------------------

    def save_csv(self):
        """
        Save all per-episode metrics to two CSV files in results/:
          - {run_name}_episodes.csv  — one row per episode
          - {run_name}_grid.csv      — one row per timestep (grid loads)

        Call this after training/testing is complete (already called
        automatically from plot_metrics()).
        """
        os.makedirs('results', exist_ok=True)

        # ---- Episode-level CSV ----------------------------------------
        ep_csv_path = f'results/{self.run_name}_episodes.csv'
        n_train = len(self.episode_rewards)
        n_test  = len(self.test_rewards)
        n_cost  = len(self.episode_costs)
        n_sat   = len(self.satisfaction_history)
        n_rows  = max(n_train, n_test, n_cost, n_sat)

        with open(ep_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode',
                'train_reward',
                'episode_cost_usd',
                'satisfaction_soc_ratio',
                'test_reward',
            ])
            for i in range(n_rows):
                writer.writerow([
                    i + 1,
                    self.episode_rewards[i]      if i < n_train else '',
                    self.episode_costs[i]        if i < n_cost  else '',
                    self.satisfaction_history[i] if i < n_sat   else '',
                    self.test_rewards[i]         if i < n_test  else '',
                ])
        print(f"-> Data saved to {ep_csv_path}")

        # ---- Timestep-level grid CSV -----------------------------------
        if self.grid_loads:
            grid_csv_path = f'results/{self.run_name}_grid.csv'
            with open(grid_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestep', 'grid_load_mw', 'delta_load_mw'])
                for i, load in enumerate(self.grid_loads):
                    delta = self.grid_power_changes[i - 1] if i > 0 else ''
                    writer.writerow([i, load, delta])
            print(f"-> Grid data saved to {grid_csv_path}")

        return ep_csv_path

    # ------------------------------------------------------------------

    def _save_config_to_json(self, filename):
        """
        Saves the run configuration to a central JSON registry.
        """
        json_path = 'results/simulation_registry.json'
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)

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
            "csv_episodes": f"results/{self.run_name}_episodes.csv",
            "csv_grid": f"results/{self.run_name}_grid.csv" if self.grid_loads else None,
            "timestamp": self.run_name.split('_')[-2] + "_" + self.run_name.split('_')[-1] if "_" in self.run_name else "N/A"
        })

        # Update and save
        data[filename] = run_summary
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"-> Configuration saved to {json_path}")

    def plot_metrics(self):
        """
        Generates plots, saves CSV data, and saves config to JSON.
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
        plt.title(f'Grid Stability ($\\sigma_g$={self.compute_stability_metric():.3f})')
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
        os.makedirs('results', exist_ok=True)
            
        plt.savefig(save_path)
        print(f"-> Plot saved to {save_path}")
        
        # Save CSV data (auto-called here so no call sites need to change)
        self.save_csv()

        # Save Config to JSON (includes csv_path references)
        self._save_config_to_json(filename)
        
        plt.close()
        # plt.show()
