import java.awt.*;          // Para desenhar gráficos (cores, fontes, formas)
import java.awt.image.BufferedImage;  // Para criar a imagem em memória
import java.io.File;        // Para salvar o arquivo no disco
import java.io.IOException; // Para tratar erros de arquivo
import java.util.Arrays;    // Para ordenar vetores e calcular médias
import java.util.Random;    // Para gerar números aleatórios
import javax.imageio.ImageIO; // Para escrever a imagem em PNG

/**
 * =================================================================
 * TÍTULO: Inferência Bayesiana usando MCMC (Metropolis-Hastings)
 * AUTOR:  (seu nome)
 * DESCRIÇÃO: Este programa estima a média (μ) de uma população
 *            a partir de 10 observações, usando o Teorema de Bayes
 *            e o algoritmo de Metropolis-Hastings (MCMC).
 *            Também gera um gráfico com dois painéis:
 *            - Trace plot: mostra a evolução da cadeia do MCMC.
 *            - Histograma: mostra a distribuição a posteriori,
 *              comparada com a solução analítica exata.
 * =================================================================
 */
public class BayesianInferenceMCMC {

    // =============================================================
    // BLOCO 1: OS DADOS OBSERVADOS (o que medimos na vida real)
    // =============================================================
    // São 10 medições de algo, por exemplo, altura de plantas,
    // temperatura em °C, ou qualquer outra variável contínua.
    // Em problemas reais, esses dados viriam de um arquivo ou banco.
    private static final double[] DADOS = {1.8, 2.1, 1.9, 2.3, 2.0, 2.2, 1.7, 2.4, 2.1, 1.9};

    // Número de observações (tamanho da amostra)
    private static final int N = DADOS.length;

    // Média aritmética simples dos dados (usada depois para validar)
    private static final double MEDIA_AMOSTRAL = Arrays.stream(DADOS).average().orElse(0.0);

    // =============================================================
    // BLOCO 2: O MODELO BAYESIANO (nossas suposições sobre o mundo)
    // =============================================================
    // Aqui definimos duas coisas:
    // 1) Verossimilhança (Likelihood): como os dados são gerados,
    //    dado um valor de μ. Assumimos que cada dado vem de uma
    //    distribuição Normal com média μ e desvio padrão 1.0.
    // 2) Prior (distribuição a priori): nossa crença inicial sobre μ
    //    antes de ver os dados. Usamos uma Normal com média 0 e
    //    desvio padrão 10 (pouco informativa, deixa os dados falarem).

    private static final double PRIOR_MEAN = 0.0;   // média do prior
    private static final double PRIOR_SD = 10.0;    // desvio do prior (grande → pouco informativo)
    private static final double LIKELIHOOD_SD = 1.0; // desvio da verossimilhança (fixo, conhecido)

    // =============================================================
    // BLOCO 3: PARÂMETROS DO ALGORITMO MCMC
    // =============================================================
    // O MCMC faz um passeio aleatório pelo espaço dos possíveis μ.
    // Estes parâmetros controlam esse passeio:
    private static final double PROPOSAL_SD = 0.5;   // "tamanho do passo" – quanto o μ pode pular
    private static final int NUM_ITERACOES = 10000;  // número total de passos do passeio
    private static final int BURN_IN = 1000;         // número de passos iniciais que descartamos
    // (o passeio precisa "esquecer" o chute inicial)

    // Gerador de números aleatórios (com semente fixa para dar o mesmo resultado sempre)
    private static final Random RANDOM = new Random(42);

    // =============================================================
    // BLOCO 4: FUNÇÕES MATEMÁTICAS (log das densidades)
    // =============================================================
    // Para calcular probabilidades, usamos o LOG porque multiplicar
    // várias probabilidades pequenas pode causar underflow (número
    // muito próximo de zero). Soma de logs é estável.

    /**
     * Calcula o log da função densidade de probabilidade (PDF)
     * de uma distribuição Normal com média 'mean' e desvio 'sd'
     * no ponto 'x'.
     *
     * Exemplo: logNormalPDF(2.0, 1.5, 1.0) → log da densidade
     * de uma Normal(1.5, 1) no valor 2.0.
     */
    private static double logNormalPDF(double x, double mean, double sd) {
        double variance = sd * sd;  // variância = desvio²
        // Fórmula do log da Normal:
        // -0.5 * ln(2πσ²) - (x-μ)²/(2σ²)
        return -0.5 * Math.log(2 * Math.PI * variance) - Math.pow(x - mean, 2) / (2 * variance);
    }

    /**
     * Versão SEM log da PDF Normal (usada para desenhar a curva
     * no gráfico, pois queremos valores de densidade reais).
     */
    private static double normalPDF(double x, double mean, double sd) {
        double variance = sd * sd;
        return (1.0 / Math.sqrt(2 * Math.PI * variance)) * Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
    }

    /**
     * Log da distribuição a priori: μ ~ Normal(PRIOR_MEAN, PRIOR_SD)
     */
    private static double logPrior(double mu) {
        return logNormalPDF(mu, PRIOR_MEAN, PRIOR_SD);
    }

    /**
     * Log da verossimilhança: soma dos logs das densidades de cada dado
     * individual. Dados | μ ~ Normal(μ, LIKELIHOOD_SD)
     */
    private static double logLikelihood(double mu) {
        double sum = 0.0;
        for (double x : DADOS) {
            sum += logNormalPDF(x, mu, LIKELIHOOD_SD);
        }
        return sum;
    }

    /**
     * Log da distribuição a posteriori (não normalizada).
     * Pelo Teorema de Bayes: P(μ|dados) ∝ P(dados|μ) * P(μ)
     * Em log: log posteriori = log verossimilhança + log prior
     */
    private static double logPosterior(double mu) {
        return logPrior(mu) + logLikelihood(mu);
    }

    // =============================================================
    // BLOCO 5: MÉTODO PRINCIPAL (onde tudo acontece)
    // =============================================================
    public static void main(String[] args) {
        // Exibe a média dos dados para referência
        System.out.printf("Média dos dados observados: %.3f%n", MEDIA_AMOSTRAL);

        // ----------------------------------------------------------
        // SUB-BLOCO 5.1: EXECUÇÃO DO MCMC (Metropolis-Hastings)
        // ----------------------------------------------------------
        // Vetor que guardará todos os valores de μ visitados (cadeia)
        double[] cadeia = new double[NUM_ITERACOES];

        // Chute inicial para μ (pode ser qualquer valor, escolhemos 0.0)
        double atual = 0.0;

        // Log da posteriori no valor atual (calculamos uma vez para economizar)
        double logPostAtual = logPosterior(atual);

        // Contador de quantas propostas foram aceitas (para calcular a taxa)
        int aceitos = 0;

        System.out.println("\nExecutando o MCMC...");

        // Laço principal: cada iteração é um passo do passeio aleatório
        for (int i = 0; i < NUM_ITERACOES; i++) {

            // --- Passo 1: Gerar uma PROPOSTA (novo valor candidato) ---
            // A proposta é o valor atual + um ruído aleatório.
            // RANDOM.nextGaussian() gera um número com distribuição Normal(0,1).
            // Multiplicamos por PROPOSAL_SD para controlar o tamanho do passo.
            double proposta = atual + RANDOM.nextGaussian() * PROPOSAL_SD;

            // --- Passo 2: Calcular a probabilidade da proposta ---
            // Queremos saber se a proposta é mais ou menos provável que o atual.
            double logPostProposta = logPosterior(proposta);

            // Razão de aceitação em log:
            // Se logRazao > 0, a proposta é MAIS provável (pois logPostProposta > logPostAtual)
            // Se logRazao < 0, a proposta é MENOS provável.
            double logRazao = logPostProposta - logPostAtual;

            // --- Passo 3: Critério de aceitação (Metropolis-Hastings) ---
            // Se logRazao > 0, aceitamos SEMPRE (andamos para uma região melhor).
            // Se logRazao ≤ 0, aceitamos com probabilidade exp(logRazao).
            // Para decidir, sorteamos um número uniforme entre 0 e 1.
            // Se o log do sorteio for menor que logRazao, aceitamos.
            // Isso permite que, às vezes, aceitemos propostas piores,
            // o que é essencial para explorar toda a distribuição.
            if (logRazao > 0 || Math.log(RANDOM.nextDouble()) < logRazao) {
                // Aceita a proposta: o atual passa a ser a proposta
                atual = proposta;
                logPostAtual = logPostProposta;
                aceitos++; // contabiliza a aceitação
            }
            // Se a condição for falsa, a proposta é rejeitada,
            // e o valor atual permanece o mesmo.

            // --- Passo 4: Armazenar o valor atual na cadeia ---
            // Se rejeitou, armazenamos o mesmo valor novamente (isso é correto).
            cadeia[i] = atual;
        }

        // Calcula a taxa de aceitação (ideal entre 20% e 50%)
        double taxaAceitacao = (double) aceitos / NUM_ITERACOES;
        System.out.printf("Taxa de aceitação: %.1f%%%n", taxaAceitacao * 100);

        // ----------------------------------------------------------
        // SUB-BLOCO 5.2: DESCARTAR O BURN-IN
        // ----------------------------------------------------------
        // Os primeiros BURN_IN passos ainda estão influenciados pelo chute inicial
        // (atual = 0.0). Vamos jogá-los fora e ficar só com o resto.
        int tamanhoAmostras = NUM_ITERACOES - BURN_IN;
        double[] amostras = new double[tamanhoAmostras];
        // Copia a partir da posição BURN_IN até o final
        System.arraycopy(cadeia, BURN_IN, amostras, 0, tamanhoAmostras);

        // ----------------------------------------------------------
        // SUB-BLOCO 5.3: ESTATÍSTICAS DA POSTERIORI (VIA MCMC)
        // ----------------------------------------------------------
        // Agora as amostras representam a distribuição a posteriori.
        // Podemos calcular média, desvio padrão e intervalos de credibilidade.
        double mediaPosterior = Arrays.stream(amostras).average().orElse(0.0);

        // Cálculo da variância amostral (com correção de Bessel, divisor n-1)
        double varPosterior = 0.0;
        for (double v : amostras) {
            varPosterior += Math.pow(v - mediaPosterior, 2);
        }
        varPosterior /= (tamanhoAmostras - 1);
        double dpPosterior = Math.sqrt(varPosterior);

        // Para o intervalo de credibilidade de 95%, ordenamos as amostras
        // e pegamos os percentis 2.5% e 97.5%.
        Arrays.sort(amostras);
        double lower = amostras[(int) (0.025 * tamanhoAmostras)];
        double upper = amostras[(int) (0.975 * tamanhoAmostras)];

        // Exibe os resultados
        System.out.println("\n--- RESULTADOS DA INFERÊNCIA (MCMC) ---");
        System.out.printf("Média a posteriori de μ: %.3f%n", mediaPosterior);
        System.out.printf("Desvio padrão a posteriori: %.3f%n", dpPosterior);
        System.out.printf("Intervalo de Credibilidade (95%%): [%.3f, %.3f]%n", lower, upper);

        // ----------------------------------------------------------
        // SUB-BLOCO 5.4: SOLUÇÃO ANALÍTICA (para validar o MCMC)
        // ----------------------------------------------------------
        // Este modelo (Normal com prior Normal) é um caso "conjugado",
        // ou seja, a posteriori também é uma Normal e tem fórmula fechada.
        // Vamos calcular essa fórmula para comparar com o MCMC.
        double precisaoPrior = 1.0 / (PRIOR_SD * PRIOR_SD);   // precisão = 1/variância
        double precisaoDados = N / (LIKELIHOOD_SD * LIKELIHOOD_SD);

        // A posteriori é Normal( mediaPostAnalitica, varPostAnalitica )
        double varPostAnalitica = 1.0 / (precisaoPrior + precisaoDados);
        double mediaPostAnalitica = (precisaoDados * MEDIA_AMOSTRAL + precisaoPrior * PRIOR_MEAN)
                / (precisaoPrior + precisaoDados);

        System.out.println("\n--- SOLUÇÃO ANALÍTICA (para validação) ---");
        System.out.printf("Média exata: %.3f%n", mediaPostAnalitica);
        System.out.printf("Desvio exato: %.3f%n", Math.sqrt(varPostAnalitica));

        // ----------------------------------------------------------
        // SUB-BLOCO 5.5: GERAR GRÁFICOS E SALVAR PNG
        // ----------------------------------------------------------
        try {
            desenharGraficos(cadeia, amostras, mediaPostAnalitica, Math.sqrt(varPostAnalitica));
            System.out.println("\n Gráfico salvo como 'mcmc_resultados.png'");
        } catch (IOException e) {
            System.err.println("Erro ao salvar a imagem: " + e.getMessage());
        }
    }

    // =============================================================
    // BLOCO 6: FUNÇÃO PARA DESENHAR OS GRÁFICOS (USANDO JAVA2D)
    // =============================================================
    // Esta função cria uma imagem PNG com dois gráficos lado a lado
    // (na verdade, empilhados verticalmente) para visualizar:
    //  - A cadeia do MCMC (trace plot) – mostra a convergência.
    //  - O histograma das amostras finais – mostra a posteriori.
    private static void desenharGraficos(double[] cadeia, double[] amostras,
                                         double mediaAnalitica, double dpAnalitica) throws IOException {

        // Dimensões da imagem final
        int largura = 1200;
        int altura = 800;

        // Cria uma imagem em memória (RGB, 8 bits por canal)
        BufferedImage imagem = new BufferedImage(largura, altura, BufferedImage.TYPE_INT_RGB);

        // Objeto gráfico que desenha sobre a imagem
        Graphics2D g2d = imagem.createGraphics();

        // Ativa suavização (anti-aliasing) para bordas mais bonitas
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Fundo branco
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, largura, altura);

        // ----------------------------------------------------------
        // DEFINIÇÃO DAS MARGENS E TAMANHO DOS GRÁFICOS
        // ----------------------------------------------------------
        int margemEsq = 80, margemDir = 50;
        int margemTop = 60, margemBot = 70;
        int largPlot = largura - margemEsq - margemDir;      // largura útil de cada plot
        int altPlot = (altura - margemTop - margemBot - 40) / 2; // altura de cada plot (com 40px entre eles)

        // =============================================================
        // GRÁFICO 1: TRACE PLOT (evolução da cadeia ao longo do tempo)
        // =============================================================
        int y1 = margemTop; // coordenada Y do canto superior esquerdo do primeiro gráfico

        // Título
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 16));
        g2d.drawString("Evolução da Cadeia (Trace Plot)", margemEsq + 20, y1 - 10);

        // Encontrar os valores mínimo e máximo da cadeia para definir a escala Y
        double minCadeia = Double.MAX_VALUE;
        double maxCadeia = -Double.MAX_VALUE;
        for (double v : cadeia) {
            if (v < minCadeia) minCadeia = v;
            if (v > maxCadeia) maxCadeia = v;
        }
        // Damos uma folga de 10% para não cortar os pontos nas bordas
        double rangeY = maxCadeia - minCadeia;
        double yMinPlot = minCadeia - 0.1 * rangeY;
        double yMaxPlot = maxCadeia + 0.1 * rangeY;

        // Desenha a caixa (retângulo) do gráfico
        g2d.setColor(Color.LIGHT_GRAY);
        g2d.drawRect(margemEsq, y1, largPlot, altPlot);

        // --- Linha vertical vermelha indicando o fim do Burn-in ---
        int xBurn = margemEsq + (int) ((double) BURN_IN / NUM_ITERACOES * largPlot);
        g2d.setColor(Color.RED);
        g2d.setStroke(new BasicStroke(2, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{6, 4}, 0));
        g2d.drawLine(xBurn, y1, xBurn, y1 + altPlot);
        g2d.setStroke(new BasicStroke(1)); // volta ao traço contínuo
        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        g2d.drawString("Burn-in", xBurn - 20, y1 - 5);

        // --- Desenha a cadeia como uma linha azul ---
        g2d.setColor(new Color(30, 80, 200)); // azul escuro
        g2d.setStroke(new BasicStroke(1.2f));
        // Primeiro ponto (iteração 0)
        int xAnterior = margemEsq;
        int yAnterior = y1 + altPlot - (int) ((cadeia[0] - yMinPlot) / (yMaxPlot - yMinPlot) * altPlot);
        // Percorre os pontos seguintes e traça linhas entre eles
        for (int i = 1; i < NUM_ITERACOES; i++) {
            int xAtual = margemEsq + (int) ((double) i / NUM_ITERACOES * largPlot);
            int yAtual = y1 + altPlot - (int) ((cadeia[i] - yMinPlot) / (yMaxPlot - yMinPlot) * altPlot);
            g2d.drawLine(xAnterior, yAnterior, xAtual, yAtual);
            xAnterior = xAtual;
            yAnterior = yAtual;
        }

        // --- Rótulos do eixo Y ---
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        g2d.drawString(String.format("%.2f", yMaxPlot), margemEsq - 45, y1 + 15);
        g2d.drawString(String.format("%.2f", yMinPlot), margemEsq - 45, y1 + altPlot);
        // Nome do eixo Y (rotacionado, mas vamos apenas escrever na vertical mesmo)
        g2d.drawString("μ", margemEsq - 30, y1 + altPlot / 2);

        // =============================================================
        // GRÁFICO 2: HISTOGRAMA DA POSTERIORI + CURVA ANALÍTICA
        // =============================================================
        int y2 = y1 + altPlot + 40; // posição Y do segundo gráfico (com espaço)

        // Título
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 16));
        g2d.drawString("Distribuição a Posteriori (Histograma vs Analítico)", margemEsq + 20, y2 - 10);

        // --- Preparar o histograma a partir das amostras finais ---
        int numBins = 50; // número de barras
        double minAmostra = Double.MAX_VALUE;
        double maxAmostra = -Double.MAX_VALUE;
        for (double v : amostras) {
            if (v < minAmostra) minAmostra = v;
            if (v > maxAmostra) maxAmostra = v;
        }
        double rangeAmostra = maxAmostra - minAmostra;
        double binWidth = rangeAmostra / numBins;

        int[] counts = new int[numBins];
        for (double v : amostras) {
            int bin = (int) ((v - minAmostra) / binWidth);
            if (bin == numBins) bin = numBins - 1; // segurança para o último valor
            counts[bin]++;
        }

        // Calcular a densidade máxima (para escalonar a altura das barras)
        double maxDensidade = 0;
        double total = amostras.length;
        for (int count : counts) {
            double dens = count / (total * binWidth); // densidade = frequência relativa / largura do bin
            if (dens > maxDensidade) maxDensidade = dens;
        }
        double yMaxHist = maxDensidade * 1.2; // deixa 20% de folga no topo

        // Desenha a caixa do gráfico
        g2d.setColor(Color.LIGHT_GRAY);
        g2d.drawRect(margemEsq, y2, largPlot, altPlot);

        // --- Desenha as barras (verde translúcido) ---
        g2d.setColor(new Color(50, 180, 100, 150)); 
        for (int i = 0; i < numBins; i++) {
            double dens = counts[i] / (total * binWidth);
            int alturaBarra = (int) ((dens / yMaxHist) * altPlot);
            int xBarra = margemEsq + (int) ((double) i / numBins * largPlot);
            int largBarra = (int) ((double) largPlot / numBins) + 1; // +1 para evitar espaços
            g2d.fillRect(xBarra, y2 + altPlot - alturaBarra, largBarra, alturaBarra);
        }

        // --- Desenha a CURVA ANALÍTICA (vermelha) sobre o histograma ---
        g2d.setColor(Color.RED);
        g2d.setStroke(new BasicStroke(2.5f));
        // Vamos desenhar a curva no intervalo [média - 4*desvio, média + 4*desvio]
        double xMinCurva = mediaAnalitica - 4 * dpAnalitica;
        double xMaxCurva = mediaAnalitica + 4 * dpAnalitica;
        int pontosCurva = 200;
        double passoCurva = (xMaxCurva - xMinCurva) / pontosCurva;

        // Primeiro ponto
        int xCurvaAnt = margemEsq + (int) ((xMinCurva - minAmostra) / rangeAmostra * largPlot);
        double densAnt = normalPDF(xMinCurva, mediaAnalitica, dpAnalitica);
        int yCurvaAnt = y2 + altPlot - (int) ((densAnt / yMaxHist) * altPlot);

        // Laço para os demais pontos
        for (int i = 1; i <= pontosCurva; i++) {
            double xAtualCurva = xMinCurva + i * passoCurva;
            double densAtual = normalPDF(xAtualCurva, mediaAnalitica, dpAnalitica);
            int xCurvaAtual = margemEsq + (int) ((xAtualCurva - minAmostra) / rangeAmostra * largPlot);
            int yCurvaAtual = y2 + altPlot - (int) ((densAtual / yMaxHist) * altPlot);

            // Só desenha se estiver dentro dos limites do gráfico
            if (xCurvaAtual >= margemEsq && xCurvaAtual <= margemEsq + largPlot) {
                g2d.drawLine(xCurvaAnt, yCurvaAnt, xCurvaAtual, yCurvaAtual);
            }
            xCurvaAnt = xCurvaAtual;
            yCurvaAnt = yCurvaAtual;
        }

        // --- Rótulos do eixo Y (densidade) ---
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        g2d.drawString(String.format("%.2f", yMaxHist), margemEsq - 45, y2 + 15);
        g2d.drawString("0.00", margemEsq - 45, y2 + altPlot);
        g2d.drawString("Densidade", margemEsq - 55, y2 + altPlot / 2);

        // --- Rótulo do eixo X (μ) ---
        g2d.drawString("μ", margemEsq + largPlot / 2 - 5, y2 + altPlot + 30);

        // --- Legenda (dentro do gráfico, no canto superior direito) ---
        g2d.setFont(new Font("Arial", Font.PLAIN, 13));
        g2d.setColor(new Color(50, 180, 100));
        g2d.drawString("--- Histograma (MCMC)", margemEsq + largPlot - 200, y2 + 20);
        g2d.setColor(Color.RED);
        g2d.drawString("--- Curva Analítica Exata", margemEsq + largPlot - 200, y2 + 40);

        // =============================================================
        // FINALIZAÇÃO: SALVAR O ARQUIVO PNG
        // =============================================================
        g2d.dispose(); // libera recursos gráficos
        ImageIO.write(imagem, "png", new File("mcmc_resultados.png"));
    }
}
