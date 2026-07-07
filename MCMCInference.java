import java.util.*;

/**
 * Classe responsável pela Inferência Bayesiana via MCMC (Metropolis-Hastings)
 * para estimar a matriz de transição de uma Cadeia de Markov.
 * 
 * Utiliza a classe MarkovChain para calcular a verossimilhança.
 * O prior é uma Dirichlet(alpha) para cada linha da matriz.
 * A proposta é uma Dirichlet centrada na matriz atual (random walk).
 */
public class MCMCInference {

    private static final Random random = new Random();

    // 1. GERADOR GAMMA (Marsaglia & Tsang)
    // // Gera uma variável aleatória com Distribuição Gamma via método de Marsaglia e Tsang (2000).
    private static double nextGamma(double shape, double scale) {
        if (shape < 1) {
            double u = random.nextDouble();
            return nextGamma(1 + shape, scale) * Math.pow(u, 1.0 / shape);
        }
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        while (true) {
            double x = random.nextGaussian();
            double v = 1.0 + c * x;
            if (v <= 0) continue;
            v = v * v * v;
            double u = random.nextDouble();
            if (u < 1.0 - 0.0331 * Math.pow(x, 4)) {
                return d * v * scale;
            }
            if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) {
                return d * v * scale;
            }
        }
    }

    // 2. AMOSTRADOR DIRICHLET
    // // Gera um vetor de proporções aleatórias cuja soma é igual a 1 via Distribuição Dirichlet.
    private static double[] nextDirichlet(double[] alphas) {
        double[] gammas = new double[alphas.length];
        double soma = 0.0;
        for (int i = 0; i < alphas.length; i++) {
            gammas[i] = nextGamma(alphas[i], 1.0);
            soma += gammas[i];
        }
        for (int i = 0; i < gammas.length; i++) {
            gammas[i] /= soma;
        }
        return gammas;
    }

    // ------------------------------------------------------------
    // 3. FUNÇÕES DE PROPOSTA, PRIOR E LIKELIHOOD
    // ------------------------------------------------------------

    /**
     * Calcula o log-prior para a matriz.
     * Prior: cada linha i segue uma Dirichlet(alphaPrior).
     * log(prior) = sum_{i,j} (alphaPrior - 1) * log(P[i][j])
     */
    private static double logPrior(double[][] matrix, double alphaPrior) {
        double logp = 0.0;
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                logp += (alphaPrior - 1.0) * Math.log(matrix[i][j] + 1e-12);
            }
        }
        return logp;
    }

    /**
     * Calcula o log-likelihood dos dados observados, dada a matriz de transição.
     * Usa a classe MarkovChain para calcular a probabilidade da sequência.
     */
    private static double logLikelihood(double[][] matrix, List<String> estados, List<String> observacoes) {
        MarkovChain mc = new MarkovChain(estados);
        mc.definirMatriz(matrix, estados);
        double prob = mc.probabilidadeSequencia(observacoes);
        return Math.log(prob + 1e-300); // serve de proteção contra log(0)
    }

    /**
     * Gera uma proposta de nova matriz.
     * Para cada linha i, amostra uma Dirichlet com parâmetros:
     * alphas[j] = concentration * current[i][j] + 1e-6
     * O parâmetro 'concentration' controla o tamanho do passo.
     * Quanto maior, mais próximas as propostas ficam da atual.
     */
    private static double[][] proporMatriz(double[][] current, double concentration) {
        int n = current.length;
        double[][] proposta = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[] alphas = new double[n];
            for (int j = 0; j < n; j++) {
                alphas[j] = concentration * current[i][j] + 1e-6;
            }
            proposta[i] = nextDirichlet(alphas);
        }
        return proposta;
    }

    private static double[][] copiarMatriz(double[][] original) {
        double[][] copia = new double[original.length][];
        for (int i = 0; i < original.length; i++) {
            copia[i] = original[i].clone();
        }
        return copia;
    }

    // ------------------------------------------------------------
    // 4. EXECUÇÃO DO MCMC (METROPOLIS-HASTINGS)
    // ------------------------------------------------------------

    /**
     * Executa o MCMC para estimar a matriz de transição.
     *
     * @param estados             Lista de nomes dos estados.
     * @param observacoes         Sequência de dados observados.
     * @param numIteracoes        Número total de iterações do MCMC.
     * @param burnIn              Número de iterações descartadas no início (burn-in).
     * @param thinning            Guarda apenas 1 a cada 'thinning' amostras.
     * @param alphaPrior          Parâmetro do prior Dirichlet (ex: 1.0 para uniforme).
     * @param concentration       Parâmetro de passo da proposta (ex: 10~20).
     * @return Um objeto ResultadoMCMC contendo as amostras e estatísticas.
     */
    public static ResultadoMCMC executar(
            List<String> estados,
            List<String> observacoes,
            int numIteracoes,
            int burnIn,
            int thinning,
            double alphaPrior,
            double concentration) {

        int n = estados.size();

        // --- Inicialização: matriz uniforme ---
        double[][] atual = new double[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(atual[i], 1.0 / n);
        }

        // --- Log-posterior inicial ---
        double logPostAtual = logPrior(atual, alphaPrior) + logLikelihood(atual, estados, observacoes);

        // --- Estruturas para guardar as amostras ---
        List<double[][]> amostras = new ArrayList<>();
        List<Double> logPosts = new ArrayList<>();

        // --- Loop principal ---
        for (int iter = 0; iter < numIteracoes; iter++) {

            // 1. Gerar proposta
            double[][] proposta = proporMatriz(atual, concentration);

            // 2. Calcular log-posterior da proposta
            double logPostProposta = logPrior(proposta, alphaPrior) + logLikelihood(proposta, estados, observacoes);

            // 3. Critério de Aceitação de Metropolis
            // Aceita se log(random) < (logPostProposta - logPostAtual)
            // Ou seja, se a proposta for melhor (diferença positiva) ou, se for pior,
            // aceita com probabilidade proporcional à razão.
            if (Math.log(random.nextDouble()) < (logPostProposta - logPostAtual)) {
                atual = proposta;
                logPostAtual = logPostProposta;
            }

            // 4. Armazenar amostra (após burn-in e respeitando o thinning)
            if (iter >= burnIn && (iter - burnIn) % thinning == 0) {
                amostras.add(copiarMatriz(atual));
                logPosts.add(logPostAtual);
            }
        }

        return new ResultadoMCMC(amostras, logPosts, estados);
    }

    // ------------------------------------------------------------
    // 5. CLASSE PARA ARMAZENAR RESULTADOS
    // ------------------------------------------------------------

    public static class ResultadoMCMC {
        private final List<double[][]> amostras;
        private final List<Double> logPosts;
        private final List<String> estados;
        private final int numEstados;
        private final int numAmostras;

        public ResultadoMCMC(List<double[][]> amostras, List<Double> logPosts, List<String> estados) {
            this.amostras = amostras;
            this.logPosts = logPosts;
            this.estados = estados;
            this.numEstados = estados.size();
            this.numAmostras = amostras.size();
        }

        /**
         * Retorna a média posterior (estimativa pontual de Bayes).
         */
        public double[][] getMediaPosterior() {
            double[][] media = new double[numEstados][numEstados];
            for (double[][] mat : amostras) {
                for (int i = 0; i < numEstados; i++) {
                    for (int j = 0; j < numEstados; j++) {
                        media[i][j] += mat[i][j];
                    }
                }
            }
            for (int i = 0; i < numEstados; i++) {
                for (int j = 0; j < numEstados; j++) {
                    media[i][j] /= numAmostras;
                }
            }
            return media;
        }

        /**
         * Retorna o percentil 'percentil' (ex: 2.5 para o limite inferior de 95%).
         */
        private double[][] getPercentil(double percentil) {
            double[][] result = new double[numEstados][numEstados];
            for (int i = 0; i < numEstados; i++) {
                for (int j = 0; j < numEstados; j++) {
                    double[] valores = new double[numAmostras];
                    for (int k = 0; k < numAmostras; k++) {
                        valores[k] = amostras.get(k)[i][j];
                    }
                    Arrays.sort(valores);
                    int idx = (int) Math.round((percentil / 100.0) * (numAmostras - 1));
                    result[i][j] = valores[Math.max(0, Math.min(idx, numAmostras - 1))];
                }
            }
            return result;
        }

        public double[][] getLimiteInferior95() { return getPercentil(2.5); }
        public double[][] getLimiteSuperior95() { return getPercentil(97.5); }

        public List<Double> getLogPosts() { return logPosts; }
        public int getNumAmostras() { return numAmostras; }

        public void imprimirResultados() {
            double[][] media = getMediaPosterior();
            double[][] lower = getLimiteInferior95();
            double[][] upper = getLimiteSuperior95();

            System.out.println("=".repeat(60));
            System.out.println("MATRIZ DE TRANSIÇÃO ESTIMADA (MÉDIA POSTERIOR):");
            System.out.println("=".repeat(60));
            System.out.printf("%-12s", "Origem \\ Dest");
            for (String s : estados) {
                System.out.printf("%10s", s);
            }
            System.out.println();
            for (int i = 0; i < numEstados; i++) {
                System.out.printf("%-12s", estados.get(i));
                for (int j = 0; j < numEstados; j++) {
                    System.out.printf("%10.3f", media[i][j]);
                }
                System.out.println();
            }

            System.out.println("\n" + "=".repeat(60));
            System.out.println("INTERVALOS DE CREDIBILIDADE 95%:");
            System.out.println("=".repeat(60));
            for (int i = 0; i < numEstados; i++) {
                for (int j = 0; j < numEstados; j++) {
                    System.out.printf("P(%s -> %s): [%6.3f, %6.3f]\n",
                            estados.get(i), estados.get(j),
                            lower[i][j], upper[i][j]);
                }
            }
            System.out.println("\nNúmero de amostras efetivas: " + numAmostras);
        }
    }
}
