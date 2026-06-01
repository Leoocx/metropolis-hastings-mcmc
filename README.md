# metropolis-hastings-mcmc


# Algoritmos de Amostragem por Método de Monte Carlo

## Introdução

O método de Monte Carlo (Monte Carlo Simulation) é uma técnica de simulação estatística baseada em amostragem aleatória.

Dado um conjunto de amostras estatísticas, normalmente buscamos estimar a função densidade de probabilidade que gerou essas amostras. Métodos comuns incluem Máxima Verossimilhança (MLE) e Máxima Estimativa a Posteriori (MAP).

O problema inverso é:

> Dada uma distribuição de probabilidade `p(x)`, como gerar amostras que sigam essa distribuição?

Esse problema é conhecido como **amostragem (Sampling)**. Este documento apresenta dois algoritmos fundamentais:

- MCMC (Markov Chain Monte Carlo)
- Gibbs Sampling

---

## Sampling

A geração de números aleatórios com distribuição uniforme `Uniform(0,1)` é relativamente simples.

Um gerador congruencial linear pode ser definido por:

\[
x_{n+1}=(ax_n+c)\ mod\ m
\]

onde `a`, `c` e `m` são constantes apropriadas.

A partir de números uniformes, podemos gerar amostras de outras distribuições.

### Exemplo: distribuição binária

\[
P(X=0)=0.5,\quad P(X=1)=0.5
\]

Podemos gerar um número aleatório `r` entre 0 e 1:

- Se `r < 0.5`, então `X = 1`
- Caso contrário, `X = 0`

### Exemplo: lançamento de dado

\[
P(X=i)=1/6,\quad i\in\{1,2,\ldots,6\}
\]

Dividimos o intervalo `[0,1]` em seis partes iguais e verificamos em qual delas o valor sorteado cai.

### Distribuições multivariadas

Quando temos:

\[
P(X_1,X_2,\ldots,X_n)
\]

e as variáveis são independentes:

\[
P(X_1,X_2,\ldots,X_n)=P(X_1)P(X_2)\cdots P(X_n)
\]

podemos amostrar cada variável separadamente.

Entretanto, para distribuições complexas ou de alta dimensão, são necessários métodos mais sofisticados, como MCMC e Gibbs Sampling.

---

## Cadeias de Markov e Distribuições Estacionárias

Uma Cadeia de Markov satisfaz:

\[
P(X_{t+1}=x|X_t,X_{t-1},\ldots)=P(X_{t+1}=x|X_t)
\]

Ou seja, o próximo estado depende apenas do estado atual.

### Exemplo de mobilidade social

Considere três classes sociais:

1. Classe baixa
2. Classe média
3. Classe alta

A matriz de transição é:

\[
P=
\begin{bmatrix}
0.65 & 0.28 & 0.07\\
0.15 & 0.67 & 0.18\\
0.12 & 0.36 & 0.52
\end{bmatrix}
\]

Se a distribuição atual é:

\[
\pi_0=[\pi_0(1),\pi_0(2),\pi_0(3)]
\]

então:

\[
\pi_n=\pi_0P^n
\]

Após muitas iterações, a distribuição converge para um valor fixo, independentemente do estado inicial.

Essa distribuição limite é chamada de **distribuição estacionária**.

### Teorema da Cadeia de Markov

Se uma cadeia é:

- Irredutível (todos os estados se comunicam)
- Aperiódica

então existe uma distribuição estacionária única `π`, tal que:

\[
\pi P=\pi
\]

e:

\[
\lim_{n\to\infty}P^n
\]

converge para uma matriz cujas linhas são iguais a `π`.

Esse resultado é a base teórica dos métodos MCMC.

---

## Algoritmo MCMC

Desejamos gerar amostras de uma distribuição alvo `p(x)`.

A ideia é construir uma cadeia de Markov cuja distribuição estacionária seja exatamente `p(x)`.

Se a cadeia convergir, os estados visitados após a convergência podem ser considerados amostras de `p(x)`.

Essa ideia foi introduzida por Leonard Metropolis em 1953.

### Condição de Balanço Detalhado

Uma distribuição `π(x)` é estacionária se:

\[
\pi(i)P_{ij}=\pi(j)P_{ji}
\]

para todos os estados `i` e `j`.

Essa condição é chamada de **Detailed Balance Condition**.

### Ajuste da matriz de transição

Suponha uma cadeia com transições:

\[
q(i,j)
\]

Normalmente:

\[
p(i)q(i,j)\neq p(j)q(j,i)
\]

Introduzimos então uma taxa de aceitação `α(i,j)`:

\[
p(i)q(i,j)\alpha(i,j)
=
p(j)q(j,i)\alpha(j,i)
\]

Obtendo uma nova cadeia que satisfaz o balanço detalhado.

---

## Algoritmo Metropolis-Hastings

Para melhorar a eficiência do MCMC, utiliza-se:

\[
\alpha(i,j)
=
\min
\left\{
\frac{p(j)q(j,i)}
{p(i)q(i,j)},
1
\right\}
\]

Essa modificação aumenta a taxa de aceitação e acelera a convergência.

O algoritmo Metropolis-Hastings tornou-se uma das técnicas mais importantes para amostragem de distribuições complexas.

---

## Gibbs Sampling

Em espaços de alta dimensão, o Metropolis-Hastings pode se tornar ineficiente.

O Gibbs Sampling resolve esse problema realizando amostragens sucessivas das distribuições condicionais.

Considere uma distribuição conjunta:

\[
p(x,y)
\]

Em vez de amostrar diretamente dela, alternamos:

\[
y \sim p(y|x)
\]

e depois:

\[
x \sim p(x|y)
\]

Esse procedimento gera uma cadeia cuja distribuição estacionária é a distribuição conjunta original.

### Caso geral

Para:

\[
(x_1,x_2,\ldots,x_n)
\]

amostramos cada variável condicionalmente às demais:

\[
p(x_i|x_1,\ldots,x_{i-1},x_{i+1},\ldots,x_n)
\]

repetindo o processo iterativamente.

O Gibbs Sampling foi proposto por Stuart Geman e Donald Geman em 1984 e é amplamente utilizado em inferência Bayesiana moderna.

---

## Referências

1. Monte Carlo Sampling Algorithms.
2. Survey and Explanation of Random Sampling Methods (MCMC, Gibbs Sampling etc.).
