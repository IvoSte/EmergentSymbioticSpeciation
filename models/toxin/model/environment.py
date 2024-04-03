
class Environment:

    def __init__(self, n_toxins, toxin_base):
        self.n_toxins = n_toxins
        self.toxin_base = toxin_base
        self.toxins = [toxin_base] * n_toxins

    def step(self):
        self.reset_toxins()

    def reset_toxins(self):
        self.toxins = [self.toxin_base] * self.n_toxins

    def decrease_toxin(self, toxin_index, amount):
        self.toxins[toxin_index] -= amount
        if self.toxins[toxin_index] < 0:
            self.toxins[toxin_index] = 0

    def toxin_report(self):
        print(f"Toxins: {self.toxins}")