import java.util.*;

public class Main {
    public static void main(String[] args) {
        // 1. Definir os estados e os dados observados
        List<String> estados = Arrays.asList("Sol", "Chuva", "Nublado");
        List<String> observacoes = Arrays.asList(
                "Sol", "Sol", "Chuva", "Nublado", "Chuva", "Sol",
                "Nublado", "Nublado", "Chuva", "Sol", "Sol", "Chuva"
        );

        // 2. Configurando o MCMC
        int iteracoes = 50000;
        int burnIn = 10000;
        int thinning = 10;
        double alphaPrior = 2.0;   // (Dirichlet com alfa=2)
        double concentracao = 15.0; // Tamanho do passo

        MCMCInference.ResultadoMCMC resultado = MCMCInference.executar(
                estados, observacoes, iteracoes, burnIn, thinning, alphaPrior, concentracao
        );

        resultado.imprimirResultados();

        double[][] media = resultado.getMediaPosterior();
        MarkovChain mc = new MarkovChain(estados);
        mc.definirMatriz(media, estados);

        List<String> previsao = mc.gerarSequencia("Sol", 15);
        System.out.println("\nPrevisão de 15 dias (usando a média posterior): " + previsao);

        List<String> teste = Arrays.asList("Sol", "Chuva", "Chuva");
        double prob = mc.probabilidadeSequencia(teste);
        System.out.printf("Probabilidade de %s: %.4f\n", teste, prob);
    }
}
