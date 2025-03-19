#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdbool.h>

#define HAUL_IMPLEMENTATION
#include "vector.h"

#define LINE_SIZE 256
#define POINT_DIMENSION 9
#define WEIGHTS_COUNT POINT_DIMENSION + 1
#define M_Ef 2.7182818284590452354f

#define PRINT_METADATA
#define PRINT_TABLE
#define DEBUG

typedef struct {
    float point[POINT_DIMENSION];
    float class;
} Sample;

Sample *parseDataLine(char *line) {
    Sample *sample = malloc(sizeof(Sample));

    char *token = strtok(line, ",");
    token = strtok(NULL, ","); // Skip the first column
    for (int i = 0; i < POINT_DIMENSION; ++i) {
        if (*token == '?') {
            free(sample);
            return NULL;
        }

        sample->point[i] = atoi(token);
        token = strtok(NULL, ",");
    }

    int class = atoi(token);
    // Change the class number from 2;4 to 0;1
    if (class == 2) {
        sample->class = 0;
    } else {
        sample->class = 1;
    }

    return sample;
}

void shuffle_samples(vector_t *samples) {
    srand(21);

    for (int i = 1; i < samples->stored; ++i) {
        int j = rand() % (i + 1);

        Sample *temp = vector_get(samples, i);
        vector_set(samples, i, vector_get(samples, j));
        vector_set(samples, j, temp);
    }
}

void split_dataset(vector_t *samples, vector_t *learning_data, vector_t *validation_data, vector_t *test_data) {
    // Split initial dataset 80:10:10
    int learning_data_size = 0.8 * samples->stored;
    int validation_data_size = 0.1 * samples->stored;

    for (int i = 0; i < learning_data_size; ++i) {
        vector_push(learning_data, samples->items[i]);
    }

    for (int i = learning_data_size; i < learning_data_size + validation_data_size; ++i) {
        vector_push(validation_data, samples->items[i]);
    }

    for (int i = learning_data_size + validation_data_size; i < samples->stored; ++i) {
        vector_push(test_data, samples->items[i]);
    }
}

void generate_random_weights(float *weights, int seed) {
    srand(seed);
    for (int i = 0; i < WEIGHTS_COUNT; ++i) {
        float lower = -3.0, upper = 3.0;
        weights[i] = lower + ((float)rand() / RAND_MAX) * (upper - lower);
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + powf(M_Ef, -x));
}

float neuron(float *point, float *weights) {
    float result = 1.0f * weights[0]; // += bias

    for (int i = 0; i < POINT_DIMENSION; ++i) {
        result += point[i] * weights[i + 1]; // += weights
    }

    return result;
}

float evaluate(float *point, float *weights) {
    return sigmoid(neuron(point, weights));
}

float calculate_error(vector_t *data, float *weights) {
    float error = 0;

    for (int i = 0; i < data->stored; ++i) {
        Sample *sample = data->items[i];
        float t = sample->class;
        float y = evaluate(sample->point, weights);
        error += powf(y - t, 2);
    }

    return error / data->stored;
}

float calculate_accuracy(vector_t *data, float *weights) {
    int correct_samples = 0;

    for (int i = 0; i < data->stored; ++i) {
        Sample *sample = data->items[i];
        float t = sample->class;
        float y = roundf(evaluate(sample->point, weights));

        if (t == y) {
            ++correct_samples;
        }
    }

    return (float) correct_samples / (float) data->stored;
}

typedef struct {
    float learning_data_error;
    float learning_data_accuracy;

    float validation_data_error;
    float validation_data_accuracy;

    float test_data_error;
    float test_data_accuracy;

    float weights[WEIGHTS_COUNT];
} LearningResult;

#define BASIC_LEARNING_RATE 0.5f
#define EPOCHS 100

LearningResult packet_gradient_descent(vector_t *learning_data, vector_t *validation_data, vector_t *test_data, float *weights, float learning_rate, bool silent) {
    clock_t start = clock();

    float total_error = FLT_MAX;
    int epoch = 0;

    float learning_data_errors[EPOCHS];
    float validation_data_errors[EPOCHS];
    float learning_data_accuracy[EPOCHS];
    float validation_data_accuracy[EPOCHS];

    while (total_error > 0 && epoch < EPOCHS) {
        total_error = 0;
        float gradient_sum[WEIGHTS_COUNT] = {0};

        for (int i = 0; i < learning_data->stored; ++i) {
            Sample *sample = learning_data->items[i];
            float t = sample->class;
            float y = evaluate(sample->point, weights);

            for (int j = 0; j < WEIGHTS_COUNT; ++j) {
                float x = j == 0 ? 1 : sample->point[j - 1]; // First value is 1 for bias
                gradient_sum[j] += (y - t) * y * (1 - y) * x;
            }

            total_error += powf(t - y, 2);
        }

        for (int j = 0; j < WEIGHTS_COUNT; ++j) {
            weights[j] -= learning_rate * (gradient_sum[j] / learning_data->stored);
        }

        learning_data_accuracy[epoch] = calculate_accuracy(learning_data, weights);
        validation_data_accuracy[epoch] = calculate_accuracy(validation_data, weights);

        learning_data_errors[epoch] = total_error / learning_data->stored;
        validation_data_errors[epoch] = calculate_error(validation_data, weights);
        ++epoch;
    }

    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000; // Convert to ms

    if (!silent) {
        #ifdef PRINT_METADATA
            printf("Packet gradient descent:\n");
            printf("\tTime: %.5f ms\n", elapsed_time);
            printf("\tLearning errors:\n\t\t");
            for (int i = 0; i < EPOCHS; ++i) {
                printf("(%d, %.5f) ", i + 1, learning_data_errors[i]);
            }
            printf("\n\tValidation errors:\n\t\t");
            for (int i = 0; i < EPOCHS; ++i) {
                printf("(%d, %.5f) ", i + 1, validation_data_errors[i]);
            }
            printf("\n");
            printf("\tLearning accuracy:\n\t\t");
            for (int i = 0; i < EPOCHS; ++i) {
                printf("(%d, %.5f) ", i + 1, learning_data_accuracy[i]);
            }
            printf("\n");
            printf("\tValidation accuracy:\n\t\t");
            for (int i = 0; i < EPOCHS; ++i) {
                printf("(%d, %.5f) ", i + 1, validation_data_accuracy[i]);
            }
            printf("\n");
            printf("\tBias: %.5f\n", weights[0]);
            printf("\tWeights: ");
            for (int i = 1; i < WEIGHTS_COUNT; ++i) {
                printf("%.5f, ", weights[i]);
            }
            printf("\n");
            printf("\tTest data error: %.5f\n", calculate_error(test_data, weights));
            printf("\tTest data accuracy: %.5f\n", calculate_accuracy(test_data, weights));
        #endif
    }

    LearningResult result;

    result.learning_data_error = learning_data_errors[EPOCHS - 1];
    result.learning_data_accuracy = learning_data_accuracy[EPOCHS - 1];

    result.validation_data_error = validation_data_errors[EPOCHS - 1];
    result.validation_data_accuracy = validation_data_accuracy[EPOCHS - 1];

    result.test_data_error = calculate_error(test_data, weights);
    result.test_data_accuracy = calculate_accuracy(test_data, weights);

    for (int i = 0; i < WEIGHTS_COUNT; ++i) {
        result.weights[i] = weights[i];
    }

    return result;
}

LearningResult stochastic_gradient_descent(vector_t *learning_data, vector_t *validation_data, vector_t *test_data, float *weights, float learning_rate, bool silent) {
    clock_t start = clock();

    float total_error = FLT_MAX;
    int epoch = 0;

    float learning_data_errors[EPOCHS];
    float validation_data_errors[EPOCHS];
    float learning_data_accuracy[EPOCHS];
    float validation_data_accuracy[EPOCHS];

    while (total_error > 0 && epoch < EPOCHS) {
        total_error = 0;

        for (int i = 0; i < learning_data->stored; ++i) {
            Sample *sample = learning_data->items[i];
            float t = sample->class;
            float y = evaluate(sample->point, weights);

            for (int j = 0; j < WEIGHTS_COUNT; ++j) {
                float x = j == 0 ? 1 : sample->point[j - 1]; // First value is 1 for bias
                weights[j] -= learning_rate * (y - t) * y * (1 - y) * x;
            }

            total_error += powf(t - y, 2);
        }

        learning_data_accuracy[epoch] = calculate_accuracy(learning_data, weights);
        validation_data_accuracy[epoch] = calculate_accuracy(validation_data, weights);

        learning_data_errors[epoch] = total_error / learning_data->stored;
        validation_data_errors[epoch] = calculate_error(validation_data, weights);
        ++epoch;
    }

    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000; // Convert to ms

    if (!silent) {
        #ifdef PRINT_METADATA
            printf("Stochastic gradient descent:\n");
            printf("\tTime: %.5f ms\n", elapsed_time);
            printf("\tLearning errors:\n\t\t");
            for (int i = 0; i < EPOCHS; ++i) {
                printf("(%d, %.5f) ", i + 1, learning_data_errors[i]);
            }
            printf("\n\tValidation errors:\n\t\t");
            for (int i = 0; i < EPOCHS; ++i) {
                printf("(%d, %.5f) ", i + 1, validation_data_errors[i]);
            }
            printf("\n");
            printf("\tLearning accuracy:\n\t\t");
            for (int i = 0; i < EPOCHS; ++i) {
                printf("(%d, %.5f) ", i + 1, learning_data_accuracy[i]);
            }
            printf("\n");
            printf("\tValidation accuracy:\n\t\t");
            for (int i = 0; i < EPOCHS; ++i) {
                printf("(%d, %.5f) ", i + 1, validation_data_accuracy[i]);
            }
            printf("\n");
            printf("\tBias: %.5f\n", weights[0]);
            printf("\tWeights: ");
            for (int i = 1; i < WEIGHTS_COUNT; ++i) {
                printf("%.5f, ", weights[i]);
            }
            printf("\n");
            printf("\tTest data error: %.5f\n", calculate_error(test_data, weights));
            printf("\tTest data accuracy: %.5f\n", calculate_accuracy(test_data, weights));
        #endif
    }

    LearningResult result;

    result.learning_data_error = learning_data_errors[EPOCHS - 1];
    result.learning_data_accuracy = learning_data_accuracy[EPOCHS - 1];

    result.validation_data_error = validation_data_errors[EPOCHS - 1];
    result.validation_data_accuracy = validation_data_accuracy[EPOCHS - 1];

    result.test_data_error = calculate_error(test_data, weights);
    result.test_data_accuracy = calculate_accuracy(test_data, weights);

    for (int i = 0; i < WEIGHTS_COUNT; ++i) {
        result.weights[i] = weights[i];
    }

    return result;
}

typedef LearningResult (*GDFunction)(vector_t *learning_data, vector_t *validation_data, vector_t *test_data, float *weights, float learning_rate, bool silent);

#define LEARNING_RATES_COUNT 4
float learning_rates[LEARNING_RATES_COUNT] = { 0.2f, 0.4f, 0.6f, 0.8f };

void analyze_learning_rates(GDFunction f, vector_t *learning_data, vector_t *validation_data, vector_t *test_data) {
    float weights[WEIGHTS_COUNT];
    
    LearningResult results[LEARNING_RATES_COUNT];

    for (int i = 0; i < LEARNING_RATES_COUNT; ++i) {
        generate_random_weights(weights, 42);
        results[i] = f(learning_data, validation_data, test_data, weights, learning_rates[i], true);
    }
    
    #ifdef PRINT_METADATA
        printf("\tTest data errors: ");
        printf("\t\t");
        for (int i = 0; i < LEARNING_RATES_COUNT; ++i) {
            printf("(%.1f, %.5f) ", learning_rates[i], results[i].test_data_error);
        }
        printf("\n");

        printf("\tTest data accuracies: ");
        printf("\t\t");
        for (int i = 0; i < LEARNING_RATES_COUNT; ++i) {
            printf("(%.1f, %.5f) ", learning_rates[i], results[i].test_data_accuracy);
        }
        printf("\n");
    #endif

    // Find the best result
    int best_result_index = 0;
    for (int i = 0; i < LEARNING_RATES_COUNT; ++i) {
        if (results[i].test_data_accuracy > results[best_result_index].test_data_accuracy) {
            best_result_index = i;
        }
    }

    #ifdef PRINT_METADATA
        printf("\tBest result:\n");
        printf("\tBias: %.5f\n", results[best_result_index].weights[0]);
        printf("\tWeights: ");
        for (int i = 1; i < WEIGHTS_COUNT; ++i) {
            printf("%.5f, ", results[best_result_index].weights[i]);
        }
        printf("\n");
        printf("\tEpochs: %d\n", EPOCHS);
        printf("\tLearning data error at last epoch: %.5f\n", results[best_result_index].learning_data_error);
        printf("\tValidation data error at last epoch: %.5f\n", results[best_result_index].validation_data_error);
        printf("\tTest data error: %.5f\n", results[best_result_index].test_data_error);
        printf("\tLearning data accuracy at last epoch: %.5f\n", results[best_result_index].learning_data_accuracy);
        printf("\tValidation data accuracy at last epoch: %.5f\n", results[best_result_index].validation_data_accuracy);
        printf("\tTest data accuracy: %.5f\n", results[best_result_index].test_data_accuracy);

        #ifdef PRINT_TABLE
            printf("\tCompare test dataset values with real data: \n");
            for (int i = 0; i < test_data->stored; ++i) {
                Sample *sample = test_data->items[i];

                float t = sample->class;
                float y = roundf(evaluate(sample->point, results[best_result_index].weights));

                printf("%d & %.1f & %.1f \\\\\n", i + 1, t, y);
            }
            printf("\n");
        #endif
    #endif
}

int main() {
    FILE *file = fopen("breast-cancer-wisconsin.data", "r");
    if (file == NULL) {
        perror("Error opening file\n");
        return 1;
    }

    vector_t samples;
    create_vector(&samples, 16);

    char line[LINE_SIZE];
    while (fgets(line, sizeof(line), file)) {
        Sample *s = parseDataLine(line);

        if (s != NULL) {
            vector_push(&samples, s);
        }
    }

    shuffle_samples(&samples);

    vector_t learning_data, validation_data, test_data;
    create_vector(&learning_data, 8);
    create_vector(&validation_data, 8);
    create_vector(&test_data, 8);
    split_dataset(&samples, &learning_data, &validation_data, &test_data);

    float weights[WEIGHTS_COUNT];

    generate_random_weights(weights, 42);
    #ifdef DEBUG
        printf("[DEBUG] Packet gradient descent initial weights: ");
        for (int i = 0; i < WEIGHTS_COUNT; ++i) {
            printf("%.5f ", weights[i]);
        }
        printf("\n");
        packet_gradient_descent(&learning_data, &validation_data, &test_data, weights, BASIC_LEARNING_RATE, false);
    #endif

    generate_random_weights(weights, 42);
    #ifdef DEBUG
        printf("[DEBUG] Stochastic gradient descent initial weights: ");
        for (int i = 0; i < WEIGHTS_COUNT; ++i) {
            printf("%.5f ", weights[i]);
        }
        printf("\n");
    #endif
    stochastic_gradient_descent(&learning_data, &validation_data, &test_data, weights, BASIC_LEARNING_RATE, false);

    printf("Analyze packet gradient descent different learning rates:\n");
    analyze_learning_rates(packet_gradient_descent, &learning_data, &validation_data, &test_data);
    printf("Analyze stochastic gradient descent different learning rates:\n");
    analyze_learning_rates(stochastic_gradient_descent, &learning_data, &validation_data, &test_data);

    fclose(file);
    free_vector_content(&samples);
    free_vector(&samples);
    free_vector(&learning_data);
    free_vector(&validation_data);
    free_vector(&test_data);
    return 0;
}