#include <stdio.h>
#include "pico/stdlib.h"

#include "tflm_wrapper.h"   
#include "wine_dataset.h"      // O dataset de teste (gerado no Colab)
#include "wine_params.h"    // Médias e Desvios (gerado no Colab)

// Definições do Dataset Wine
#define NUM_FEATURES 13
#define NUM_CLASSES 3

// Matriz de Confusão (Linhas = Real, Colunas = Predito)
int confusion_matrix[NUM_CLASSES][NUM_CLASSES] = {0};

// Função auxiliar para encontrar o maior valor (Argmax)
int get_max_index(const float *probs, int size) {
    int max_idx = 0;
    float max_val = probs[0];
    for (int i = 1; i < size; i++) {
        if (probs[i] > max_val) {
            max_val = probs[i];
            max_idx = i;
        }

    }
    return max_idx;
}

int main() {
    stdio_init_all();
    
    // Aguarda conexão USB para não perdermos o começo do print
    sleep_ms(3000); 

    // 1. Inicializar o Modelo TensorFlow Lite Micro
    if (tflm_init_model() != 0) {
        printf("[ERRO] Falha ao inicializar o modelo TFLM!\n");
        while (1) tight_loop_contents(); // Trava o código
    }
    printf("[OK] Modelo inicializado com sucesso.\n");

    int acertos = 0;
    int total_amostras = NUM_SAMPLES; // Vem do wine_data.h

    printf("Iniciando testes com %d amostras de vinho...\n", total_amostras);

    // 2. Loop Principal: Testar cada vinho do dataset
    for (int i = 0; i < total_amostras; i++) {
        
        // --- ETAPA A: PREPARAÇÃO (NORMALIZAÇÃO) ---
        // O modelo aprendeu com dados normalizados, então precisamos
        // transformar os dados brutos da mesma forma.
        float entrada_normalizada[NUM_FEATURES];
        
        for (int j = 0; j < NUM_FEATURES; j++) {
            // Fórmula: (ValorBruto - Média) / DesvioPadrão
            entrada_normalizada[j] = (wine_features[i][j] - wine_means[j]) / wine_stds[j];
        }

        // --- ETAPA B: INFERÊNCIA (O MODELO PENSA) ---
        float probabilidades[NUM_CLASSES];
        tflm_infer(entrada_normalizada, probabilidades);

        // --- ETAPA C: PÓS-PROCESSAMENTO ---
        // Descobrir qual classe tem a maior probabilidade
        int predito = get_max_index(probabilidades, NUM_CLASSES);
        int real = wine_labels[i];

        // Contabilizar estatísticas
        if (predito == real) {
            acertos++;
        }
        
        // Atualiza a Matriz de Confusão
        confusion_matrix[real][predito]++;

        // Opcional: Imprimir detalhes das 5 primeiras amostras para debug
        if (i < 5) {
            printf("Amostra %03d | Real: %d | Predito: %d | Confiança: %.2f%%\n", 
                   i, real, predito, probabilidades[predito] * 100.0f);
        }
    }

    printf("\n================ RELATORIO FINAL ================\n");
    printf("Acuracia: %.2f%% (%d/%d)\n", 
           (float)acertos / total_amostras * 100.0f, acertos, total_amostras);
    
    printf("\n--- Matriz de Confusao ---\n");
    printf("       | Pred 0 | Pred 1 | Pred 2 |\n");
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("Real %d |", i);
        for (int j = 0; j < NUM_CLASSES; j++) {
            printf("   %4d |", confusion_matrix[i][j]);
        }
        printf("\n");
    }
    printf("=================================================\n");

    
    while (true) {
        tight_loop_contents();
    }
}