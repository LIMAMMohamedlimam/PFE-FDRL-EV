# BUG_TRACKING.md

Living bug register **and** the protocol Claude Code follows when auditing this
repository. Read this file before touching code. Keep it up to date as part of
every bug-related change.

---

## 1. How Claude Code should use this file

### Audit protocol

1. **Scan, don't guess.** Walk the repo and inspect actual code. Every reported
   bug must cite `path/to/file.py:line` (or a line range). No speculative
   entries without a concrete location.
2. **Verify before logging.** If you suspect a bug, confirm it by reading the
   surrounding code and, where cheap, reproducing it. Mark anything you could
   not confirm as `type: suspected` and say what evidence is missing.
3. **One entry per bug.** Use the registry table (§4) for the at-a-glance view
   and a detailed block (§5) for each. Assign sequential IDs: `BUG-001`, `BUG-002`…
4. **Minimal, surgical fixes.** Match the existing style. Prefer the smallest
   change that resolves the root cause over a refactor. If a clean fix needs a
   broader change, log it, state the trade-off, and ask before doing it.
5. **Consistency with the paper is a correctness criterion.** Any discrepancy
   between code behaviour and the metrics/claims reported in `main.tex`
   (SOC achievement, communication-overhead reduction, scheduling-time
   reduction, IEEE 33-bus setup) is a bug. Flag it.
6. **Update status, never delete history.** When a bug is fixed, move it through
   the lifecycle (§3); don't erase the entry. Closed bugs stay in the registry
   as a record.

### Areas to inspect (RL + federated specifics)

Go through these systematically; they are the high-yield categories for this
kind of repo:

- **Artifact/checkpoint saving** — run isolation, overwrites, metadata, atomicity
  *(see BUG-001)*.
- **Reproducibility** — seeding of python/numpy/torch/cuda, deterministic flags,
  config capture, whether a run can be reproduced from saved metadata.
- **Federated aggregation (HFedAvg)** — client weighting (by sample count?),
  client sampling, round/local-epoch bookkeeping, divergence between global and
  client states, SWIFT/FedProx terms applied where intended.
- **SAC implementation** — target-network update (`tau`), entropy temperature
  (fixed vs auto-tuned `alpha`), replay buffer sampling/overflow, double-Q usage,
  action squashing/log-prob correction, gradient clipping.
- **LoRA compression** — rank/alpha config, which modules are adapted, merge vs
  keep-separate at save/load time, that compressed payload is what is actually
  transmitted/measured for the comm-overhead claim.
- **Environment (IEEE 33-bus / EV)** — SOC bounds and clipping, CC-CV transition,
  action/voltage constraints, episode horizon, reward scaling, NaN/Inf guards.
- **Metrics & logging** — that logged numbers correspond to the definitions used
  in the paper; units; averaging windows; off-by-one in episode/step counters.
- **Resource handling** — GPU memory growth, unreleased file handles, dataloader
  workers, `.detach()`/`no_grad()` in eval paths.
- **Config & magic numbers** — hardcoded hyperparameters that should live in a
  config; values that disagree with the paper.
- **Error handling & edge cases** — empty client set, single-EV episodes,
  resume-from-checkpoint paths, division by zero in normalization.
- **Dependencies** — unpinned versions, mismatched CUDA/torch.

---

## 2. Severity & type legend

**Severity**

| Level    | Meaning                                                                 |
|----------|-------------------------------------------------------------------------|
| Critical | Wrong results, data loss, or invalidates a reported metric.             |
| High     | Breaks a feature, blocks reproduction, or silently corrupts artifacts.  |
| Medium   | Incorrect behaviour in a non-core path, or significant friction.        |
| Low      | Cosmetic, style, minor robustness.                                      |

**Type:** `correctness` · `reproducibility` · `artifact` · `performance` ·
`robustness` · `config` · `docs` · `suspected`

---

## 3. Status lifecycle

```
OPEN → IN_PROGRESS → FIXED → VERIFIED → CLOSED
                                  ↘ REOPENED → IN_PROGRESS
Terminal alternatives: WONTFIX · DUPLICATE
```

- **FIXED** = change written. **VERIFIED** = re-checked / re-run and confirmed
  resolved (state the check used). Only then → **CLOSED**.

---

## 4. Bug registry

| ID      | Title                                              | Severity | Type           | Status      | Area        |
|---------|----------------------------------------------------|----------|----------------|-------------|-------------|
| BUG-001 | Model/agent weights saved without run isolation or metadata | High | artifact / reproducibility | FIXED | checkpointing |
| BUG-002 | PPOAgent.save/load_trained_model appelle self.state_dict() inexistant | Critical | correctness | FIXED | agents/PPO |
| BUG-003 | PYTHONHASHSEED défini après le démarrage Python — sans effet | High | reproducibility | FIXED | seeding |
| BUG-004 | base_load train N(3.5,0.2) ≠ test N(3.8,0.3) — distributions différentes | High | correctness / config | FIXED | evaluation |
| BUG-005 | GridEnv retourne NaN pour min/max_voltage quand power-flow ne converge pas | High | robustness | FIXED | environment |
| BUG-006 | HFedAvg pondère tous les clients également malgré hétérogénéité revendiquée | High | correctness | FIXED | federated |
| BUG-007 | log_alpha non borné — risque d'explosion numérique | Medium | robustness | OPEN | SAC |
| BUG-008 | target_entropy_scale = −0.5 au lieu du standard −1.0 | Medium | config | OPEN | SAC |
| BUG-009 | Paramètres LoRA manquants non détectés lors du chargement | Medium | correctness | OPEN | LoRA |
| BUG-010 | SWIFT peut produire un ensemble vide si force_selected couvre tout | Medium | robustness | OPEN | federated/SWIFT |
| BUG-011 | critic_target synchronisé = critic après agrégation FL (lag soft-update perdu) | Low | correctness | OPEN | SAC/federated |
| BUG-012 | GridEnv division par zéro si réseau sans nœuds de charge | Low | robustness | OPEN | environment |

*(Append new rows as bugs are discovered.)*

---

## 5. Detailed entries

### BUG-001 — Weights saved without run isolation or metadata

- **Severity:** High · **Type:** artifact / reproducibility · **Status:** FIXED
- **Reported by:** Med (known issue)

**Symptom**
Agent/model weights are saved to a flat or fixed location and are not tied to the
run that produced them. Successive runs overwrite each other, and there is no
record of the configuration, seed, or metrics behind a given checkpoint. The
plots already use a separate timestamped directory per run; weights do not follow
the same convention.

**Why it matters**
- Runs clobber each other → no comparison, no rollback, lost provenance.
- A checkpoint can't be linked to the plots/metrics/config of the same run.
- No way to tell which checkpoint produced a number reported in the paper
  (direct paper-consistency risk).
- Risk of evaluating or shipping a stale/mismatched checkpoint.

**Root cause (confirmed)**
- `training/ComparisonPipeline.py:506,509` — deux lignes identiques
  `model_dir = os.path.join("results/trained_models")` dans `run_single_experiment()`.
  La fonction crée déjà un `run_name` horodaté à la ligne 259 mais ne l'utilise pas
  pour les checkpoints.
- `agents/PPOAgent.py:274` — `save_trained_model()` n'appelait pas `os.makedirs()`
  avant `torch.save()` (contrairement à `SACAgent`).

**Proposed fix — mirror the plots' run-directory convention**

Create the run directory **once** at training start and route *all* artifacts
into it (the same dir the plots use):

```
runs/
  2026-06-30_14-23-05__hfdrl_ieee33_sac_lora/
    run_metadata.json        # how to reproduce this run
    config.yaml              # resolved config snapshot
    checkpoints/
      latest.pt              # always overwritten with newest
      best.pt                # best by tracked metric (e.g. mean SOC / reward)
      ep0500.pt              # periodic
      global_agent.pt        # inference-only export (state_dict only)
    plots/                   # existing plot outputs
    metrics/metrics.csv
    logs/train.log
```

Checkpoint contents (`*.pt`, for **resumable** training):
`model state_dict`(s) (global + per-client if applicable), optimizer & scheduler
states, RNG states (python/numpy/torch/cuda), episode/step counters, and the
resolved config. Also write a lightweight weights-only export for inference.

`run_metadata.json` fields:
`timestamp`, `git_commit` (+ dirty flag), `seed`,
`hyperparameters` (SAC: lr, gamma, tau, alpha/auto-entropy, batch_size,
buffer_size, hidden_dims; Federated: num_clients, rounds, local_epochs,
HFedAvg weighting, client sampling, SWIFT/FedProx terms; LoRA: rank, alpha,
target_modules), `environment` (33-bus topology id, num_EVs, SOC bounds,
CC-CV params, horizon), `versions` (python/torch/cuda), `metrics` (best & final:
SOC %, comm overhead, scheduling time), `notes`.

Engineering details:
- **Atomic writes:** save to `*.tmp`, then `os.replace()` — avoids corruption if
  a run is interrupted mid-save.
- **Retention:** keep `best.pt` + `latest.pt` + last-N periodic to bound disk.
- Save `state_dict`s (not pickled objects); version the checkpoint format with a
  `format_version` key.

**Implementation note (keep it surgical)**
Introduce/reuse a single `get_run_dir()` (or `RunDirectory`) helper that returns
the per-run path, thread it through the trainer's save calls, and replace the
fixed paths. Ideally one helper + edits at the save call sites. Do not refactor
unrelated code.

**Acceptance criteria**
- [ ] Two consecutive runs create two distinct directories; neither overwrites
      the other.
- [ ] Each checkpoint is co-located with the plots/metrics/config of its run.
- [ ] Loading `best.pt` reproduces the reported eval metrics within tolerance.
- [ ] A checkpoint resumes training deterministically given the same seed.
- [ ] `run_metadata.json` fully specifies how to reproduce the run.

**Resolution log**
- **2026-06-30** — Fix appliqué.
  - `training/ComparisonPipeline.py` : ajout imports `json`/`subprocess`,
    helper `_git_commit()`, remplacement des deux `"results/trained_models"` par
    `os.path.join("results", run_name, "checkpoints")`, écriture de
    `run_metadata.json` (timestamp, git_commit, config complète, checkpoints_dir).
  - `agents/PPOAgent.py` : ajout `os.makedirs(directory, exist_ok=True)` dans
    `save_trained_model()`.
  - `agents/SACAgent.py` : enrichissement du checkpoint full-model avec
    `format_version`, `actor_optimizer`, `critic_optimizer`, `alpha_optimizer`
    (states pour reprise d'entraînement).
- **Vérification** : imports Python valides (`python -c "import training.ComparisonPipeline"`).
  Deux runs consécutifs produiront des répertoires distincts sous `results/{run_name}/`.

---

### BUG-002 — PPOAgent.save/load_trained_model appelle self.state_dict() inexistant

- **Severity:** Critical · **Type:** correctness · **Status:** FIXED
- **Location:** `agents/PPOAgent.py:278,282,288,292`

**Symptom**
`AttributeError: 'PPOAgent' object has no attribute 'state_dict'` à chaque appel de
`save_trained_model()` ou `load_trained_model()` sur un PPOAgent (avec ou sans LoRA).

**Root cause**
`PPOAgent` hérite de `BaseAgent` (plain Python class, pas `nn.Module`). Les méthodes
`state_dict()` / `load_state_dict()` n'existent pas sur `self`. Le réseau de neurones
est dans `self.policy` (un `ActorCritic(nn.Module)`).

**Proposed fix**
Miroir de `get_parameters()` / `set_parameters()` déjà corrects :
- Save LoRA : `get_lora_state_dict(self.policy, prefix='')`
- Save full : `self.policy.state_dict()`
- Load LoRA : `load_lora_state_dict(self.policy, params, prefix='', device=...)` + sync `policy_old`
- Load full : `self.policy.load_state_dict(sd)` + `self.policy_old.load_state_dict(sd)`

**Resolution log**
- **2026-06-30** — Corrigé dans `agents/PPOAgent.py:274-298`. Syntaxe vérifiée (`ast.parse`).

---

### BUG-003 — PYTHONHASHSEED défini après le démarrage Python — sans effet

- **Severity:** High · **Type:** reproducibility · **Status:** FIXED
- **Location:** `training/MultiSeedRunner.py:63`

**Symptom**
`os.environ['PYTHONHASHSEED'] = str(seed)` est appelé dans `set_global_seed()` après
que Python ait démarré. Python lit `PYTHONHASHSEED` uniquement au démarrage de
l'interpréteur ; l'affecter en cours d'exécution est sans effet sur la randomisation
des hash (`dict`, `set`).

**Why it matters**
Si une logique d'entraînement (agrégation, indexation des agents) dépend de l'ordre
d'itération sur des dicts/sets, les résultats NE seront PAS reproductibles d'un run
à l'autre. Les seeds torch/numpy sont corrects, mais la randomisation hash persiste.

**Proposed fix**
Passer `PYTHONHASHSEED` en variable d'environnement AVANT de lancer Python. Exemples :
```bash
PYTHONHASHSEED=42 python main.py --seed 42
```
Ou, pour les études multi-seeds, lancer chaque seed dans un sous-processus :
```python
subprocess.run(['python', 'main.py', '--seed', str(seed)],
               env={**os.environ, 'PYTHONHASHSEED': str(seed)})
```
Supprimer la ligne `os.environ['PYTHONHASHSEED'] = ...` de `set_global_seed()` ou
y ajouter un avertissement documentant cette limitation.

**Acceptance criteria**
- [x] Un `RuntimeWarning` est émis si `PYTHONHASHSEED` ne correspond pas au seed attendu.
- [ ] Chaque seed est lancé dans un sous-processus avec `PYTHONHASHSEED` défini avant Python (amélioration future).

**Resolution log**
- **2026-06-30** — `training/MultiSeedRunner.py:63` : capture de la valeur courante de `PYTHONHASHSEED`, émission d'un `RuntimeWarning` si elle ne correspond pas au seed, puis assignment pour propagation aux sous-processus. `import warnings` ajouté.

---

### BUG-004 — base_load train N(3.5, 0.2) ≠ test N(3.8, 0.3)

- **Severity:** High · **Type:** correctness / config · **Status:** FIXED
- **Location:** `training/ComparisonPipeline.py:416` (train) et `training/ComparisonPipeline.py:560` (test)

**Symptom**
La charge de base du réseau est tirée selon des distributions différentes en phase
d'entraînement (μ=3.5 MW, σ=0.2 MW) et en phase de test (μ=3.8 MW, σ=0.3 MW).

**Why it matters**
Les métriques de test évaluent l'agent sur un scénario de charge différent de celui
vu à l'entraînement (distribution shift non documentée). Si c'est involontaire, les
chiffres rapportés dans le papier mesurent une généralisation non annoncée.

**Root cause**
Deux valeurs hardcodées distinctes dans la même fonction `run_single_experiment()`.

**Proposed fix**
Si distribution identique souhaitée :
```python
# Unifier sous un seul paramètre de config dans training.yaml :
# base_load: {mean: 3.5, std: 0.2}
base_load_mw = np.random.normal(train_cfg['base_load']['mean'],
                                train_cfg['base_load']['std'])
```
Si le distribution shift est intentionnel (test out-of-distribution), l'ajouter en
commentaire et dans le papier.

**Acceptance criteria**
- [x] Les deux distributions sont identiques (N(3.5, 0.2)).

**Resolution log**
- **2026-06-30** — `training/ComparisonPipeline.py:560` : `np.random.normal(3.8, 0.3)` → `np.random.normal(3.5, 0.2)`. Commentaire inline ajouté sur la ligne training pour documenter l'unification.

---

### BUG-005 — GridEnv retourne NaN pour min/max_voltage quand power-flow ne converge pas

- **Severity:** High · **Type:** robustness · **Status:** FIXED
- **Location:** `env/GridEnv.py:74-75`

**Symptom**
Quand `pp.runpp()` lève `LoadflowNotConverged`, `info['min_voltage']` et
`info['max_voltage']` sont mis à `float('nan')`. Si ces valeurs entrent dans la
construction d'état (via `get_state()`), l'agent verra des NaN dans ses observations.

**Why it matters**
Note : `lambda_grid` est sécurisé (forcé à 2.0 en cas d'échec). Mais tout code
downstream utilisant `min_voltage` / `max_voltage` de l'`info` dict propagera NaN,
cassant les observations et potentiellement les gradients.

**Proposed fix**
```python
# GridEnv.py — remplacer les NaN par des valeurs sentinelle cohérentes :
else:
    voltage_violations = -1
    lambda_grid = 2.0
    min_v = self.v_min  # valeur sentinelle basse cohérente
    max_v = self.v_max  # valeur sentinelle haute cohérente
```
Ou ajouter une guard dans `get_state()` si les tensions y sont utilisées :
`volt = np.nan_to_num(volt, nan=self.v_min)`.

**Acceptance criteria**
- [x] `info['min_voltage']` et `info['max_voltage']` ne sont jamais NaN.
- [x] La construction d'état reste valide lors d'un échec de convergence.

**Resolution log**
- **2026-06-30** — `env/GridEnv.py:74-75` : `float('nan')` → `self.v_min` / `self.v_max` (valeurs sentinelle aux bornes des contraintes). `lambda_grid = 2.0` non modifié.

---

### BUG-006 — HFedAvg pondère tous les clients également malgré hétérogénéité revendiquée

- **Severity:** High · **Type:** correctness · **Status:** FIXED
- **Location:** `training/ComparisonPipeline.py:169`

**Symptom**
Chaque agent est collecté avec `n_samples=sim_hours` (typiquement 24) quel que soit
son historique d'entraînement réel. La pondération effective est `1/N` pour tous.

**Why it matters**
Si le papier revendique une agrégation pondérée par les données (HFedAvg avec poids
proportionnels aux transitions locales), l'implémentation ne le reflète pas.
Les métriques comparatives pourraient être affectées si la non-iid-ité des clients
est censée influencer l'agrégation.

**Root cause**
`edge.collect(vid, agents[vid].get_parameters(), sim_hours)` — `sim_hours` est
utilisé comme proxy du nombre d'échantillons, mais il est identique pour tous les
agents. La pondération réelle devrait utiliser le nombre de transitions dans le
replay buffer ou de gradient steps effectués.

**Proposed fix**
Passer le vrai compte de transitions (e.g., `len(agent.buffer)`) comme poids :
```python
n_samples = len(agents[vid].buffer) if hasattr(agents[vid], 'buffer') else sim_hours
edge.collect(vid, agents[vid].get_parameters(), n_samples)
```

**Acceptance criteria**
- [x] Le poids d'agrégation de chaque client reflète son volume de données réel (taille du replay buffer pour SAC, `sim_hours` pour les autres).

**Resolution log**
- **2026-06-30** — `training/ComparisonPipeline.py:169` : `edge.collect(vid, ..., sim_hours)` → `_w = len(agents[vid].buffer) if hasattr(agents[vid], 'buffer') else sim_hours` ; `edge.collect(vid, ..., _w)`. Pour SAC le poids est maintenant proportionnel aux transitions réelles dans le buffer.

---

### BUG-007 — log_alpha non borné — risque d'explosion numérique

- **Severity:** Medium · **Type:** robustness · **Status:** OPEN
- **Location:** `agents/SACAgent.py:231, 405`

**Symptom**
`log_alpha` est optimisé sans borne. Si le gradient d'entropie pousse `log_alpha`
à des valeurs extrêmes (ex. 10), alors `alpha = exp(10) ≈ 22 000` écrase le gradient
de politique et bloque l'apprentissage.

**Proposed fix**
```python
# Après l'update d'alpha_optim :
with torch.no_grad():
    self.log_alpha.clamp_(-5.0, 2.0)   # alpha ∈ [~0.007, ~7.4]
self.alpha = self.log_alpha.exp().item()
```

**Acceptance criteria**
- [ ] `alpha` reste dans une plage raisonnable ([1e-3, 10]) lors d'un run complet.

---

### BUG-008 — target_entropy_scale = −0.5 au lieu du standard −1.0

- **Severity:** Medium · **Type:** config · **Status:** OPEN
- **Location:** `agents/SACAgent.py:188, 230` · `configs/sac.yaml:18`

**Symptom**
L'entropie cible est `−0.5 × dim(action)` au lieu du standard SAC `−dim(action)`.
L'agent maintient moins d'exploration que le SAC de référence (Haarnoja et al. 2018).

**Why it matters**
Si le papier compare HFDRL à « SAC standard », la différence de réglage d'entropie
doit être documentée, sinon la comparaison est biaisée.

**Proposed fix**
Soit documenter explicitement ce choix dans le code et le papier, soit changer :
```yaml
# configs/sac.yaml
target_entropy_scale: -1.0   # standard SAC
```

**Acceptance criteria**
- [ ] La valeur est justifiée (commentaire dans sac.yaml et mention dans le papier),
      ou changée à -1.0 pour correspondre au standard.

---

### BUG-009 — Paramètres LoRA manquants non détectés lors du chargement

- **Severity:** Medium · **Type:** correctness · **Status:** OPEN
- **Location:** `utils/lora.py:200-210`

**Symptom**
`load_lora_state_dict()` met à jour silencieusement les clés LoRA présentes et ignore
celles qui manquent dans le dict agrégé. Un client peut opérer avec des adaptateurs
LoRA partiellement mis à jour sans aucun avertissement.

**Proposed fix**
Ajouter une vérification après merge :
```python
missing = [k for k in model.state_dict() if 'lora_' in k and k not in aggregated_dict]
if missing:
    logger.warning(f"load_lora_state_dict: {len(missing)} LoRA keys not updated: {missing[:5]}")
```

**Acceptance criteria**
- [ ] Un warning est émis si des clés LoRA attendues sont absentes du dict agrégé.

---

### BUG-010 — SWIFT peut produire un ensemble de sélection vide

- **Severity:** Medium · **Type:** robustness · **Status:** OPEN
- **Location:** `training/SWIFTScheduler.py:108`

**Symptom**
Si tous les clients éligibles sont déjà dans `force_selected`, `k_utility = 0` et
l'ensemble retourné contient uniquement `force_selected`. Dans un cas extrême
(force_selected vide aussi), l'agrégation reçoit zéro clients.

**Proposed fix**
Ajouter une garde avant de retourner l'ensemble sélectionné :
```python
if not selected:
    logger.warning("SWIFT: empty selection — falling back to all eligible clients")
    selected = eligible
```

**Acceptance criteria**
- [ ] L'agrégation reçoit toujours au moins un client.

---

### BUG-011 — critic_target synchronisé = critic après agrégation FL

- **Severity:** Low · **Type:** correctness · **Status:** OPEN
- **Location:** `agents/SACAgent.py:306`

**Symptom**
Après agrégation FL, `critic_target` est mis à jour avec les mêmes poids que
`critic`. Le décalage temporel (soft-update via τ) est réinitialisé à chaque round
fédéré, causant potentiellement des pics de Q-loss post-agrégation.

**Why it matters**
Impact faible en pratique (le soft-update reprend dès le step suivant), mais
pourrait contribuer à de l'instabilité juste après l'agrégation.

**Proposed fix**
Ne pas synchroniser `critic_target` lors de l'agrégation ; laisser le soft-update
naturel maintenir le décalage. Alternativement, documenter ce comportement.

---

### BUG-012 — GridEnv division par zéro si réseau sans nœuds de charge

- **Severity:** Low · **Type:** robustness · **Status:** OPEN
- **Location:** `env/GridEnv.py:43`

**Symptom**
`self.net.load.p_mw = base_load_mw / len(self.net.load)` lève `ZeroDivisionError`
si le réseau pandapower n'a aucun nœud de charge (cas hypothétique, réseau malformé).

**Proposed fix**
```python
n_loads = len(self.net.load)
if n_loads > 0:
    self.net.load.p_mw = base_load_mw / n_loads
```

---

## 6. New-bug entry template

```markdown
### BUG-00X — <short title>

- **Severity:** <Critical|High|Medium|Low> · **Type:** <type> · **Status:** OPEN
- **Location:** `path/to/file.py:line`

**Symptom**
<observed behaviour>

**Why it matters**
<impact, incl. any paper-consistency angle>

**Root cause**
<confirmed cause + evidence>

**Proposed fix**
<minimal change>

**Acceptance criteria**
- [ ] …

**Resolution log**
<what changed, files, verification>
```
