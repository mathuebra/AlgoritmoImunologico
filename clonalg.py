import numpy as np
import matplotlib.pyplot as plt

# --- Função Alpine2 (para maximização) ---
def alpine2(x):
    return np.prod(np.sqrt(x) * np.sin(x))

# --- Clonalg ---
def clonalg(
    func,
    n_dim=2,
    bounds=(0, 10),
    pop_size=50,
    n_generations=100,
    n_selected=10,
    beta=5,
    mutation_rate=0.2,
    replace_rate=0.2
):
    # Inicialização da população
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, n_dim))
    best_scores = []

    for gen in range(n_generations):
        # Avaliação
        fitness = np.array([func(ind) for ind in population])

        # Seleciona os melhores
        idx_best = np.argsort(fitness)[::-1]  # maior é melhor
        selected = population[idx_best[:n_selected]]
        selected_fitness = fitness[idx_best[:n_selected]]

        # Clonagem proporcional à afinidade
        clones = []
        for i in range(n_selected):
            n_clones = int(beta * (n_selected - i) / n_selected)
            clones.extend([selected[i]] * n_clones)
        clones = np.array(clones)

        # Mutação inversamente proporcional à afinidade
        clone_fitness = np.array([func(c) for c in clones])
        max_fit = np.max(clone_fitness)
        normalized_fit = clone_fitness / (max_fit + 1e-9)

        for i, c in enumerate(clones):
            # Quanto pior, mais muta
            mutation_strength = mutation_rate * (1 - normalized_fit[i])
            clones[i] = c + np.random.normal(0, mutation_strength, n_dim)
            clones[i] = np.clip(clones[i], bounds[0], bounds[1])

        # Avaliar clones mutados e unir com os selecionados
        all_candidates = np.vstack((population, clones))
        fitness_all = np.array([func(ind) for ind in all_candidates])

        # Selecionar nova população
        idx_sorted = np.argsort(fitness_all)[::-1]
        population = all_candidates[idx_sorted[:pop_size]]

        # Substituir parte por novos aleatórios (diversificação)
        n_replace = int(replace_rate * pop_size)
        population[-n_replace:] = np.random.uniform(bounds[0], bounds[1], (n_replace, n_dim))

        best_score = np.max(fitness_all)
        best_scores.append(best_score)

        if gen % 10 == 0:
            print(f"Geração {gen}: melhor f(x) = {best_score:.4f}")

    best = population[np.argmax([func(ind) for ind in population])]
    return best, best_scores

# --- Execução ---
best_sol, history = clonalg(alpine2)

print("\nMelhor solução encontrada:", best_sol)
print("Valor da função:", alpine2(best_sol))

# --- Gráfico da evolução ---
plt.plot(history)
plt.title("Evolução do valor máximo (Clonalg)")
plt.xlabel("Geração")
plt.ylabel("f(x)")
plt.grid()
plt.show()
