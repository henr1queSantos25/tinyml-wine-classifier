#include <cstdio>
#include "pico/stdlib.h"

// -------------------------------------------------------------------
// TensorFlow Lite Micro
// -------------------------------------------------------------------
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// IMPORTANTE: Este arquivo deve conter o array 'unsigned char model_tflite[]'
// gerado pelo comando xxd no Python.
#include "model_data.h"

#include "tflm_wrapper.h"

namespace {

// Arena de memória para o TensorFlow (8KB costuma ser suficiente para modelos pequenos)
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Resolver de operações: ajustado para 4 operações (FC, Relu, Softmax, Reshape)
// Se seu modelo usar mais tipos de camadas, aumente este número.
static tflite::MicroMutableOpResolver<10> resolver; 

}  // namespace

// -------------------------------------------------------------------
// Inicializa o modelo TFLM
// -------------------------------------------------------------------
int tflm_init_model(void) {
    // Carrega o modelo do array gerado (model_tflite é o nome da variável no model_data.h)
    model = tflite::GetModel(model_tflite);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Erro: versao do schema do modelo incompativel.\n");
        return -1;
    }

    // Registra as operações necessárias para rodar a rede neural.
    // O modelo do Wine usa basicamente Dense (FullyConnected) e ativações.
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();   // Adicionado por precaução se houver quantização
    resolver.AddDequantize(); // Adicionado por precaução

    // Cria o intérprete
    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize,
        nullptr,
        nullptr
    );
    interpreter = &static_interpreter;

    // Aloca memória para os tensores na arena
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("Erro: AllocateTensors falhou.\n");
        return -2;
    }

    // Obtém ponteiros para entrada e saída
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    if (!input_tensor || !output_tensor) {
        printf("Erro: Nao foi possivel obter tensores de entrada/saida.\n");
        return -3;
    }

    // (Opcional) Debug: imprime o tamanho esperado pelo modelo
    // printf("Input size esperado: %d\n", input_tensor->dims->data[1]); // Deve ser 13

    return 0;
}

// -------------------------------------------------------------------
// Executa a inferência
// -------------------------------------------------------------------
int tflm_infer(const float in_features[13], float out_scores[3]) {
    if (!interpreter || !input_tensor || !output_tensor) {
        return -1;
    }

    // Copia as 13 features normalizadas para o tensor de entrada do modelo
    for (int i = 0; i < 13; i++) {
        input_tensor->data.f[i] = in_features[i];
    }

    // Roda o modelo
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Erro: Invoke falhou.\n");
        return -2;
    }

    // Copia os resultados (probabilidades) para o array de saída
    for (int i = 0; i < 3; i++) {
        out_scores[i] = output_tensor->data.f[i];
    }

    return 0;
}