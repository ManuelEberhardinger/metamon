# --- bandit.py ---
import numpy as np

class BetaBernoulliBandit:
    def __init__(self, n_arms, alpha0=1.0, beta0=1.0, verbose=True, print_every = 100):
        self.alpha = np.full(n_arms, alpha0, dtype=np.float64)
        self.beta  = np.full(n_arms, beta0,  dtype=np.float64)
        self.verbose = verbose
        
        self.print_every = print_every
        self.updates = 0

    def select(self) -> int:
        # Thompson sampling
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: int):
        self.alpha[arm] += reward
        self.beta[arm]  += (1 - reward)
        self.updates += 1

        if self.verbose and (self.updates % self.print_every == 0):
            print(self)

    def decay(self, rate=0.99):
        self.alpha = self.alpha * rate + (1 - rate)
        self.beta  = self.beta  * rate + (1 - rate)

    def means(self):
        return self.alpha / (self.alpha + self.beta)

    def __str__(self):
        means = np.round(self.means(), 3).tolist()
        return (f"[Bandit] step {self.updates} | "
                f"means={means} "
                f"alpha={np.round(self.alpha,1).tolist()} "
                f"beta={np.round(self.beta,1).tolist()}"
                f"best team={np.argmax(means)}"
                f"|Games player per team {self.alpha + self.beta}")