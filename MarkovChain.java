import java.util.*;

/**
 * Classe que implementa uma Cadeia de Markov de primeira ordem.
 * Estados são representados por Strings. A matriz de transição pode ser
 * aprendida a partir de dados ou definida manualmente.
 */
public class MarkovChain {
    private List<String> estados;                          // Lista ordenada dos estados
    private Map<String, Integer> indice;                  // Mapeia estado -> índice (para acesso rápido)
    private Map<String, Map<String, Double>> matriz;      // matriz[origem][destino] = probabilidade
    private Map<String, Map<String, Integer>> contagens;  // contagens brutas para treino

    /**
     * Construtor vazio. Os estados serão adicionados dinamicamente durante o treino.
     */
    public MarkovChain() {
        this.estados = new ArrayList<>();
        this.indice = new HashMap<>();
        this.matriz = new HashMap<>();
        this.contagens = new HashMap<>();
    }

    /**
     * Construtor que já define a lista de estados.
     * Útil quando você já conhece todos os estados antecipadamente.
     */
    public MarkovChain(List<String> estados) {
        this();
        for (String s : estados) {
            adicionarEstado(s);
        }
    }

    // ---------- Métodos auxiliares internos ----------

    /**
     * Adiciona um estado se ele ainda não existir.
     * Inicializa suas estruturas de contagem e matriz.
     */
    private void adicionarEstado(String estado) {
        if (!indice.containsKey(estado)) {
            indice.put(estado, estados.size());
            estados.add(estado);
            contagens.put(estado, new HashMap<>());
            matriz.put(estado, new HashMap<>());
        }
    }

    /**
     * Converte as contagens brutas em probabilidades.
     * É chamado automaticamente após o treino.
     */
    private void computarProbabilidades() {
        for (String origem : estados) {
            Map<String, Integer> counts = contagens.get(origem);
            int total = counts.values().stream().mapToInt(Integer::intValue).sum();
            Map<String, Double> probs = matriz.get(origem);
            probs.clear();

            if (total == 0) {
                // Se não há transições, assume distribuição uniforme
                for (String destino : estados) {
                    probs.put(destino, 1.0 / estados.size());
                }
            } else {
                for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                    probs.put(entry.getKey(), entry.getValue() / (double) total);
                }
                // Garante que todos os estados apareçam (mesmo com probabilidade zero)
                for (String destino : estados) {
                    probs.putIfAbsent(destino, 0.0);
                }
            }
        }
    }

    // ---------- Métodos públicos principais ----------

    /**
     * Treina a cadeia a partir de uma sequência de observações.
     * Conta todas as transições ocorridas e atualiza a matriz.
     */
    public void treinar(List<String> sequencia) {
        for (int i = 0; i < sequencia.size() - 1; i++) {
            String atual = sequencia.get(i);
            String proximo = sequencia.get(i + 1);
            adicionarEstado(atual);
            adicionarEstado(proximo);
            Map<String, Integer> map = contagens.get(atual);
            map.put(proximo, map.getOrDefault(proximo, 0) + 1);
        }
        computarProbabilidades();
    }

    /**
     * Define a matriz de transição manualmente a partir de um mapa de mapas.
     * Exemplo: { "Sol": {"Sol":0.7, "Chuva":0.3}, "Chuva": {"Sol":0.4, "Chuva":0.6} }
     */
    public void definirMatriz(Map<String, Map<String, Double>> matrizManual) {
        this.matriz = matrizManual;
        // Atualiza a lista de estados a partir das chaves da matriz
        estados.clear();
        indice.clear();
        for (String origem : matrizManual.keySet()) {
            adicionarEstado(origem);
            for (String destino : matrizManual.get(origem).keySet()) {
                adicionarEstado(destino);
            }
        }
    }

    /**
     * Define a matriz de transição manualmente a partir de um array 2D.
     * As linhas correspondem à origem, colunas ao destino, na ordem da lista de estados.
     */
    public void definirMatriz(double[][] matrizArray, List<String> listaEstados) {
        this.estados = new ArrayList<>(listaEstados);
        this.indice.clear();
        for (int i = 0; i < estados.size(); i++) {
            indice.put(estados.get(i), i);
        }
        this.matriz.clear();
        for (int i = 0; i < estados.size(); i++) {
            String origem = estados.get(i);
            Map<String, Double> linha = new HashMap<>();
            for (int j = 0; j < estados.size(); j++) {
                linha.put(estados.get(j), matrizArray[i][j]);
            }
            matriz.put(origem, linha);
        }
    }

    /**
     * Retorna o próximo estado aleatório, dado o estado atual,
     * conforme a matriz de transição.
     */
    public String proximoEstado(String estadoAtual) {
        if (!matriz.containsKey(estadoAtual)) {
            throw new IllegalArgumentException("Estado desconhecido: " + estadoAtual);
        }
        Map<String, Double> probs = matriz.get(estadoAtual);
        double r = Math.random();
        double acum = 0.0;
        for (Map.Entry<String, Double> entry : probs.entrySet()) {
            acum += entry.getValue();
            if (r <= acum) {
                return entry.getKey();
            }
        }
        // Fallback (caso de arredondamento)
        return estados.get(estados.size() - 1);
    }

    /**
     * Gera uma sequência de comprimento 'tamanho' a partir de um estado inicial.
     */
    public List<String> gerarSequencia(String estadoInicial, int tamanho) {
        if (!indice.containsKey(estadoInicial)) {
            throw new IllegalArgumentException("Estado inicial inválido: " + estadoInicial);
        }
        List<String> seq = new ArrayList<>();
        seq.add(estadoInicial);
        for (int i = 1; i < tamanho; i++) {
            seq.add(proximoEstado(seq.get(i - 1)));
        }
        return seq;
    }

    /**
     * Calcula a probabilidade de uma sequência observada,
     * multiplicando as probabilidades de cada transição.
     */
    public double probabilidadeSequencia(List<String> sequencia) {
        if (sequencia.size() < 2) return 1.0;
        double prob = 1.0;
        for (int i = 0; i < sequencia.size() - 1; i++) {
            String origem = sequencia.get(i);
            String destino = sequencia.get(i + 1);
            Map<String, Double> linha = matriz.get(origem);
            if (linha == null) return 0.0;
            Double p = linha.get(destino);
            if (p == null || p == 0.0) return 0.0;
            prob *= p;
        }
        return prob;
    }


    public List<String> getEstados() {
        return estados;
    }

    public Map<String, Map<String, Double>> getMatriz() {
        return matriz;
    }
}