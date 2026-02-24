import pandapower as pp
import pandapower.networks as nw
import pandas as pd
import numpy as np

class GridEnv:
    """
    Modélisation de l'environnement Grid (Niveau Edge/Distribution).
    Gère le calcul de flux de puissance (OPF/PF) et les contraintes de tension.
    """
    def __init__(self, network_type='case33bw'):
        # Création du réseau (IEEE 33-bus radial distribution network par défaut)
        # Note: 'case74' n'est pas standard dans pandapower, on utilise un équivalent distribution.
        if network_type == 'case33bw':
            self.net = nw.case33bw()
        else:
            # Placeholder pour charger un fichier json/sql personnalisé si besoin
            self.net = nw.case14() 
            
        # Configuration des limites de tension [cite: 33]
        self.v_min = 0.95
        self.v_max = 1.05
        
        # Mapping des bus de charge (exclure le bus de référence/slack)
        self.load_buses = self.net.load.bus.values
        
    def reset(self):
        """Réinitialise les charges du réseau à zéro"""
        self.net.load.p_mw = 0.0
        self.net.load.q_mvar = 0.0
        return self.get_grid_state()

    def step(self, ev_loads_dict, base_load_mw):
        """
        Applique les charges des VE et la charge de base, puis résout le Power Flow.
        
        Args:
            ev_loads_dict: Dict {bus_index: power_mw} (puissance agrégée par bus)
            base_load_mw: Charge de fond du réseau (non-VE)
        """
        # 1. Mise à jour des charges de base (réparties uniformément ou selon profil)
        self.net.load.p_mw = base_load_mw / len(self.net.load)

        # 2. Injection des charges VE aux nœuds correspondants
        # On suppose que les VEs sont connectés aux nœuds de charge existants
        for bus_idx, power_mw in ev_loads_dict.items():
            # Trouver l'index de charge correspondant au bus
            load_idx = self.net.load[self.net.load.bus == bus_idx].index
            if not load_idx.empty:
                self.net.load.at[load_idx[0], 'p_mw'] += power_mw

        # 3. Résolution du Power Flow (AC)
        try:
            pp.runpp(self.net, algorithm='nr')
            converged = True
            vm_pu = self.net.res_bus.vm_pu.values
        except pp.LoadflowNotConverged:
            converged = False
            # If PF fails, return a strong congestion signal
            vm_pu = np.ones(len(self.net.bus)) * np.nan

        # 4. Calcul des métriques et pénalités
        if converged:
            max_deviation = float(np.max(np.abs(vm_pu - 1.0)))
            voltage_violations = int(np.sum((vm_pu < self.v_min) | (vm_pu > self.v_max)))

            # Continuous lambda_grid: scale deviation by 0.05 p.u. band
            lambda_grid = float(np.clip(max_deviation / 0.05, 0.0, 2.0))
            min_v = float(np.min(vm_pu))
            max_v = float(np.max(vm_pu))
        else:
            voltage_violations = -1
            lambda_grid = 2.0
            min_v = float('nan')
            max_v = float('nan')
        info = {
            'converged': converged,
            'voltage_violations': voltage_violations,
            'min_voltage': min_v,
            'max_voltage': max_v
        }
        
        return lambda_grid, info

    def get_grid_state(self):
        return self.net.res_bus.vm_pu.values