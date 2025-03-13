#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define HAUL_IMPLEMENTATION
#include "vector.h"

#define LINE_SIZE 256
#define POINT_DIMENSION 9
#define M_Ef 2.7182818284590452354f

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

float sigmoid(float x) {
    return 1.0f / (1 + powf(M_Ef, -x));
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

float loss_function(Sample *sample, float *weights) {
    float y = evaluate(sample->point, weights);
    float t = sample->class;
    return pow(y - t, 2);
}

void generate_random_weights(float *weights) {
    srand(42);
    for (int i = 0; i < POINT_DIMENSION; ++i) {
        weights[i] = (float) rand() / (RAND_MAX + 1.0f);
        printf("%f\n", weights[i]);
    }
}

#define LEARNING_RATE 2
#define EPOCHS 10

void packet_gradient_descent(vector_t *learning_data, float *weights) {
    float total_error = FLT_MAX;
    int epoch = 0;

    while (total_error > 0 && epoch < EPOCHS) {
        total_error = 0;

        for (int i = 0; i < learning_data->stored; ++i) {
            Sample *sample = learning_data->items[i];
            
            float y = evaluate(sample->point, weights);
            float t = sample->class;
        }
    }



    total_error /= learning_data->stored;
}

void stochastic_gradient_descent() {

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

    float weights[POINT_DIMENSION + 1];
    generate_random_weights(weights);

    fclose(file);
    free_vector_content(&samples);
    free_vector(&samples);
    free_vector(&learning_data);
    free_vector(&validation_data);
    free_vector(&test_data);
    return 0;
}