import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -------------------------------
# Parâmetros do algoritmo
# -------------------------------
POP_SIZE = 50         # tamanho da população
NUM_GENERATIONS = 50  # número de iterações
NUM_SELECTED = 10     # número de anticorpos selecionados
BETA = 0.5            # taxa de mutação
NEW_CELLS = 5         # número de novos anticorpos gerados por geração

# -------------------------------
# Funções auxiliares
# -------------------------------

def fitness(antibody, X, y):
    """Avalia o desempenho do anticorpo como um classificador linear."""
    preds = np.dot(X, antibody[:-1]) + antibody[-1]
    preds = (preds > np.median(preds)).astype(int)
    # Para o Iris com 3 classes, simplificaremos para binário (ou use one-vs-rest)
    acc = np.mean(preds == y)
    return acc

def clone_and_mutate(antibody, affinity, max_affinity):
    """Realiza clonagem e hipermutação inversamente proporcional à afinidade."""
    mutation_rate = (1 - affinity / (max_affinity + 1e-9)) * BETA
    clone = antibody + mutation_rate * np.random.randn(*antibody.shape)
    return clone

# -------------------------------
# Preparação dos dados
# -------------------------------

iris = load_iris()
X = iris.data
y = iris.target

# Para simplificar, faremos classificação binária (Iris Setosa vs Não-Setosa)
y = (y == 0).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Inicialização
# -------------------------------
antibodies = np.random.randn(POP_SIZE, X_train.shape[1] + 1)

# -------------------------------
# CLONALG principal
# -------------------------------
for gen in range(NUM_GENERATIONS):
    # Avaliar afinidade
    affinities = np.array([fitness(a, X_train, y_train) for a in antibodies])
    
    # Selecionar os melhores
    best_idx = np.argsort(-affinities)[:NUM_SELECTED]
    selected = antibodies[best_idx]
    best_aff = affinities[best_idx]
    
    # Clonagem e hipermutação
    max_aff = np.max(best_aff)
    clones = []
    for i, a in enumerate(selected):
        n_clones = int(best_aff[i] * 10) + 1
        for _ in range(n_clones):
            clone = clone_and_mutate(a, best_aff[i], max_aff)
            clones.append(clone)
    
    clones = np.array(clones)
    
    # Avaliar clones e formar nova população
    clone_affinities = np.array([fitness(c, X_train, y_train) for c in clones])
    combined = np.vstack((selected, clones))
    combined_aff = np.hstack((best_aff, clone_affinities))
    
    # Selecionar os melhores para a próxima geração
    next_gen_idx = np.argsort(-combined_aff)[:POP_SIZE - NEW_CELLS]
    next_gen = combined[next_gen_idx]
    
    # Adicionar novos anticorpos aleatórios
    new_cells = np.random.randn(NEW_CELLS, X_train.shape[1] + 1)
    antibodies = np.vstack((next_gen, new_cells))

# -------------------------------
# Avaliação final
# -------------------------------
best_antibody = antibodies[np.argmax([fitness(a, X_train, y_train) for a in antibodies])]
train_acc = fitness(best_antibody, X_train, y_train)
test_acc = fitness(best_antibody, X_test, y_test)

print(f"Acurácia de treino: {train_acc:.3f}")
print(f"Acurácia de teste: {test_acc:.3f}")
