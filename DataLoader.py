import matplotlib.pyplot as plt
import random
import numpy as np

# --- Générateurs de Données (Mock) ---

class DataGenerator:
    """Génère les données d'environnement (Prix et Comportement)"""
    
    @staticmethod
    def get_iso_ne_price(hour, mode='train'):
        """
        Si mode='test', on décale les prix ou on ajoute du bruit 
        pour simuler une autre saison/journée.
        """
        base_price = 0.15
        
        # Modification pour généralisation : décalage temporel ou bruit
        hour_adjusted = hour
        if mode == 'test':
            hour_adjusted = (hour + 2) % 24  # Décalage de pic (ex: hiver vs été)
            base_price += 0.02 # Prix globalement plus élevés
            
        if 8 <= hour_adjusted <= 10 or 18 <= hour_adjusted <= 21:
            return base_price * 1.5
        elif 0 <= hour_adjusted <= 5:
            return base_price * 0.6
        return base_price

    @staticmethod
    def get_nhts_profile(n_drivers):
        """
        Simule les profils NHTS 2017: Heure d'arrivée et de départ.
        Arrivée ~ Normal(18h, 2h), Départ ~ Normal(7h, 1h).
        """
        profiles = []
        for _ in range(n_drivers):
            # Simulation simplifiée sur 24h : Arrivée le soir, départ le lendemain matin
            t_arrival = int(np.random.normal(18, 2))  # ~18h00
            t_departure = int(np.random.normal(7, 1)) # ~07h00 (J+1)
            
            # Correction des bornes (0-23h)
            t_arrival = max(0, min(23, t_arrival))
            t_departure = max(0, min(23, t_departure))
            
            # Durée de connexion (si départ < arrivée, c'est le lendemain)
            stay_duration = (24 - t_arrival) + t_departure if t_departure < t_arrival else t_departure - t_arrival
            
            profiles.append({
                't_start': t_arrival,
                't_dep': t_departure, # Heure absolue de départ J+1 pour la simu
                'duration': stay_duration,
                'soc_init': np.random.uniform(0.1, 0.4), # Arrive avec batterie faible
                'soc_req': np.random.uniform(0.8, 1.0)   # Veut repartir plein
            })
        return profiles
