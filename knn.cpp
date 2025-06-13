 #include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <queue>
#include <vector>
#include <cctype>
#include <iostream>

#define MAX_SAMPLES 200
#define MAX_LINE_LEN 256
#define MAX_TEST_NUM 256
#define K 10

typedef struct {
    double features[4];
    char label[32];
} IrisSample;

typedef struct {
    double distance;
    char label[32];
} Neighbor;
bool operator < (const Neighbor &x, const Neighbor &y) {
    return x.distance < y.distance;
}

IrisSample trainingSet[MAX_SAMPLES];
int trainingSize = 0;

int split(const char* line, char tokens[][64], char delimiter) {
    int count = 0;
    const char* start = line;
    const char* ptr = line;
    while (*ptr) {
        if (*ptr == delimiter) {
            strncpy(tokens[count], start, ptr - start);
            tokens[count][ptr - start] = '\0';
            count++;
            start = ptr + 1;
        }
        ptr++;
    }
    if (ptr > start) {
        strcpy(tokens[count++], start);
    }
    return count;
}

void loadTrainingData(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open training file.\n");
        return;
    }

    char line[MAX_LINE_LEN];
    int lineNum = 0;
    while (fgets(line, sizeof(line), file)) {
        lineNum++;
        if (line[strlen(line) - 1] == '\n')
            line[strlen(line) - 1] = '\0';

        char parts[6][64];
        int partCount = split(line, parts, ',');

        if (lineNum == 1 && (strcmp(parts[0], "Sepal.Length") == 0 || strcmp(parts[0], "Species") == 0)) {
            continue;
        }

        if (partCount < 6) continue;

        IrisSample sample;
        for (int i = 0; i < 4; i++) {
            sample.features[i] = atof(parts[i + 1]);
        }
        strncpy(sample.label, parts[5], sizeof(sample.label));
        for(int i = 0; sample.label[i]; i++)
        if(!isalpha(sample.label[i]))
            sample.label[i] = 0;
        trainingSet[trainingSize++] = sample;
    }

    fclose(file);
}

double computeDistance(const double* a, const double* b) {
    double sum = 0.0;
    for (int i = 0; i < 4; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int compareNeighbors(const void* a, const void* b) {
    Neighbor* na = (Neighbor*)a;
    Neighbor* nb = (Neighbor*)b;
    return (na->distance > nb->distance) - (na->distance < nb->distance);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    size = 4;
    rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double pred[MAX_TEST_NUM][4];
    int res[MAX_TEST_NUM];

    int TEST_NUM = 1;
    if (rank == 0) {
        loadTrainingData("data.csv");
        for(int i = 0; i < TEST_NUM; i++)
            for(int j = 0; j < 4; j++)
                scanf("%lf", &pred[i][j]);
    }
    // 请在该段补充代码

    std::priority_queue <Neighbor> topKneighbor;
    for(int i = rank; i < TEST_NUM; i += size) {
        for(int k = 0; k < trainingSize; k ++) {
            Neighbor tmp;
            tmp.distance = computeDistance(trainingSet[k].features, pred[i]);
            printf("%.4f\n",tmp.distance);
            strcpy(tmp.label, trainingSet[k].label);
            topKneighbor.push(tmp);

            if(topKneighbor.size() > K)
                topKneighbor.pop();
        }
        int versi_cnt = 0, setos_cnt = 0, virgi_cnt = 0;
        double sum1 = 0, sum2 = 0, sum3 = 0;
        while(topKneighbor.size()) {
            Neighbor tmp = topKneighbor.top();
            topKneighbor.pop();
            // printf("# %d %s\n", strlen(tmp.label), tmp.label);
            if(strcmp(tmp.label, "versicolor") == 0) ++versi_cnt, sum1 += tmp.distance;
            else if(strcmp(tmp.label, "setosa") == 0) ++setos_cnt, sum2 += tmp.distance;
            else if(strcmp(tmp.label, "virginica") == 0) ++virgi_cnt, sum3 += tmp.distance;
            else {
                std::cerr << "unexpected label:" << tmp.label << std::endl;
                exit(0);
            }
        }
        int pred_label = -1;
        if(versi_cnt == setos_cnt && setos_cnt == virgi_cnt) {
            if(sum1 < std::min(sum2, sum3))
                pred_label = 0;
            else if(sum2 < std::min(sum1, sum3))
                pred_label = 1;
            else
                pred_label = 2;
        }
        else if(versi_cnt > std::max(setos_cnt, virgi_cnt))
            pred_label = 0;
        else if(setos_cnt > std::max(versi_cnt, virgi_cnt))
            pred_label = 1;
        else
            pred_label = 2;
        if(rank == 0)
            res[i] = pred_label;
        else
            MPI_Send(&pred_label, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
        // if(pred_label == 0)
        //     strcpy(res[i], "versicolor");
        // else if(pred_label == 1)
        //     strcpy(res[i], "setosa");
        // else
        //     strcpy(res[i], "virginica");
    }
    if(rank == 0) {
        int pred_res[MAX_TEST_NUM];
        for(int i = 1; i < size; i++) {
            int cnt;
            MPI_Status st;
            MPI_Recv(pred_res, TEST_NUM, MPI_INT, i, i, MPI_COMM_WORLD, &st);
            MPI_Get_count(&st, MPI_INT, &cnt);
            for(int k = i, j = 0; k < TEST_NUM; k += size, j++)
                res[k] = pred_res[j];    
        }
    }
    if(rank == 0)
        for(int i = 0; i < TEST_NUM; i++)
        if(res[i] == 0) puts("versicolor");
        else if(res[i] == 1) puts("setosa");
        else puts("virginica");
    MPI_Finalize();
    return 0;
}
