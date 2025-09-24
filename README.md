**What we’re deciding.** A practical recipe for hedging **Bitcoin options** when prices can **jump**.

**What varies in the recipe (three knobs):**
- **Model of the market:**  
  **MJD** (Merton Jump-diffusion), **SVJ** (stochastic volatility + jumps), **SVCJ** (stochastic volatility + contemporaneous jumps).
- **Hedging instruments:**  
  **Δ-neutral** (underlying only) vs **Jump-risk hedging** using **1 Call + 1 Put + Underlying**; we also check a larger set (**2C+2P+U**).
- **Implementation style:**  
  **Standard** (use the model’s own greeks) vs **BS-estimated** (use Black-Scholes greeks from market IVs).

**What “better” means.** We judge everything by **realized hedging error in dollars** (weekly absolute errors).  
Lower error = less cash bleed in real life.

---

## The workflow (simple)

1. **Calibration** → learn the model’s **hyperparameters** from data  
2. **Implementation & Result report** → plug those hyperparameters into a **notebook (.ipynb)**, run, and read the **results & takeaways**

**TL;DR:** *Calibrate → Implement → Read results.*

---

## Where to click first

- **Just want the results and story?**  
  Open the notebooks in **[Implementation & Result report/](Implementation%20%26%20Result%20report/)** (they are `.ipynb`).  
  You’ll see the hedging setup, main charts, and short conclusions.

- **Want to see where the numbers came from?**  
  Open the matching notebook in **[Calibration/](Calibration/)** for that model.  
  You’ll see what the model learned (**hyperparameters**) before we apply them.

---

## Step-by-step for any model (MJD / SVJ / SVCJ)

**1) Calibration (first step)**  
Open the model’s notebook in **`Calibration/`** and skim for:
- the data used,  
- the **final hyperparameters** (the numbers we carry forward).

**2) Implementation & Result report (second step)**  
Open the matching notebook in **`Implementation & Result report/`**:
- it **uses the calibrated hyperparameters**,  
- runs the hedging logic,  
- shows the **key charts** and **plain-English takeaways**.

> **Reading order tip:** start with **SVCJ** (most realistic), then **SVJ**, then **MJD** to see how simplifications change the outcome.

---

## What to look for in the results notebooks

- **Setup** — what’s being hedged and over what period  
- **Main charts** — behavior around jumps and overall hedging error  
- **Takeaways** — what improves, what doesn’t, and why it matters

---

## Data flow (at a glance)

**Calibration (learn hyperparameters)**  
→ **Implementation notebook (apply hyperparameters; run hedge)**  
→ **Results (read charts & conclusions in the notebook’s final sections)**
