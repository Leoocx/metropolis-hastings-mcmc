#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
METROPOLIS-HASTINGS - Explicação:

Imagine que você quer desenhar pontos (amostras) que sigam uma certa distribuição de probabilidade P(x),
mas você NÃO consegue gerar pontos diretamente dessa distribuição. No entanto, você consegue calcular
o valor de P(x) para qualquer x (mesmo que sem a constante de normalização).

O algoritmo Metropolis-Hastings cria um "passeio aleatório" (uma Cadeia de Markov) no espaço dos x,
com uma regra esperta: ele tende a aceitar passos que vão para regiões de maior probabilidade,
mas ocasionalmente também aceita passos para regiões de menor probabilidade, numa medida exata
que garante que, depois de muitos passos, os pontos visitados seguem exatamente a distribuição P(x).

O processo é:

1. Comece num ponto qualquer x0.
2. Proponha um novo ponto y, usando uma distribuição de proposta Q(x -> y) (ex.: sorteie um valor
   próximo de x, como uma Normal centrada em x).
3. Calcule a "razão de aceitação": alpha = min(1, [P(y) * Q(y->x)] / [P(x) * Q(x->y)]).
   Se a proposta for simétrica (Q(x->y) = Q(y->x)), alpha = min(1, P(y)/P(x)).
4. Aceite y com probabilidade alpha; senão, permaneça em x.
5. Repita.

O resultado: a sequência de pontos (após descartar um período inicial de "burn-in") é uma amostra
(embora correlacionada) da distribuição P(x).

"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. DEFINIÇÃO DA DISTRIBUIÇÃO ALVO P(X)
# ------------------------------------------------------------
def p(x):
    """
    Distribuição alvo: mistura de duas normais (não normalizada, mas a constante não importa).
    P(x) = 0.3 * N(x | -3, 0.5^2) + 0.7 * N(x | 2, 0.8^2)
    """
    comp1 = 0.3 * np.exp(-0.5 * ((x + 3) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi))
    comp2 = 0.7 * np.exp(-0.5 * ((x - 2) / 0.8)**2) / (0.8 * np.sqrt(2 * np.pi))
    return comp1 + comp2

# ------------------------------------------------------------
# 2. DISTRIBUIÇÃO PROPOSTA Q
# ------------------------------------------------------------
def Q_proposta(x):
    """
    Distribuição de proposta: normal centrada no valor atual x,
    com desvio padrão fixo = 1.0.
    Esta proposta é SIMÉTRICA: Q(x -> y) = Q(y -> x).
    """
    return np.random.normal(loc=x, scale=1.0)

# ------------------------------------------------------------
# 3. PARÂMETROS DO MCMC
# ------------------------------------------------------------
n_iteracoes = 50000   # número total de passos da cadeia
burn_in = 10000       # número de passos iniciais a descartar
x0 = 0.0             # valor inicial (pode ser qualquer um)

# ------------------------------------------------------------
# 4. EXECUÇÃO DO ALGORITMO METROPOLIS-HASTINGS
# ------------------------------------------------------------
amostras = []        # lista que armazenará todos os estados (incluindo burn-in)
x_atual = x0         # estado inicial

print("Simulando a cadeia...")
for passo in range(n_iteracoes):
    # ----- Passo 2: Gerar candidato y a partir da proposta -----
    y = Q_proposta(x_atual)

    # ----- Passo 3: Calcular a probabilidade de aceitação alpha -----
    # Como a proposta é simétrica, Q(y->x) / Q(x->y) = 1, então:
    # alpha = min(1, p(y) / p(x_atual))
    p_atual = p(x_atual)
    p_y = p(y)

    # Evitar divisão por zero (p_atual > 0 no nosso exemplo)
    alpha = min(1.0, p_y / p_atual)

    # ----- Passo 4: Decidir se aceita ou rejeita -----
    u = np.random.uniform(0, 1)   # sorteia um número uniforme [0,1]
    if u <= alpha:
        x_atual = y               # ACEITA: vai para y
    else:
        # REJEITA: permanece em x_atual 
        pass

    # Armazena o estado atual (seja novo ou repetido)
    amostras.append(x_atual)

print("Simulação concluída.")

# ------------------------------------------------------------
# 5. DESCARTE DO BURN-IN
# ------------------------------------------------------------
amostras_finais = amostras[burn_in:]   # descarta as primeiras 'burn_in' amostras
print(f"Total de amostras coletadas: {len(amostras_finais)}")

# ------------------------------------------------------------
# 6. VISUALIZAÇÃO: HISTOGRAMA DAS AMOSTRAS VS DENSIDADE VERDADEIRA
# ------------------------------------------------------------
x_grid = np.linspace(-6, 6, 200)       # grade de pontos para a curva real
p_grid = p(x_grid)                     # valores da densidade alvo (não normalizada)

# Normalizar a densidade teórica numericamente para comparar com o histograma
# Usamos np.trapezoid (substituto do antigo np.trapz) que calcula a integral pela regra do trapézio.
# Se ocorrer erro de atributo, tente: from numpy import trapz (antigo) ou use scipy.integrate.simps.
try:
    integral = np.trapezoid(p_grid, x_grid)
except AttributeError:
    # Fallback para versões muito antigas do numpy: usar np.trapz
    integral = np.trapz(p_grid, x_grid)
p_norm = p_grid / integral

# Plot do histograma das amostras (normalizado para ser densidade)
plt.figure(figsize=(10, 4))
plt.hist(amostras_finais, bins=50, density=True, alpha=0.6, color='skyblue', label='Amostras MCMC')

# Plot da densidade teórica normalizada
plt.plot(x_grid, p_norm, 'r-', linewidth=2, label='Densidade Alvo (teórica)')

plt.title("Metropolis-Hastings: Amostrando de uma mistura de duas Gaussianas")
plt.xlabel("x")
plt.ylabel("Densidade")
plt.legend()
plt.grid(alpha=0.3)
plt.show()