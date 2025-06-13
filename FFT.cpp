#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>

const double PI = std::acos(-1.0);

using Complex = std::complex<double>;
// 位反转置换（Bit-reversal Permutation）
void bit_reverse(std::vector<std::complex<double>>& data) {
    int n = data.size();
    std::vector<int> rev(n);
    for(int i = 0; i < n; i++) {
        rev[i] = rev[i >> 1] >> 1;
        if(i & 1)
            rev[i] |= (n >> 1);
    }
    // #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++)
        if(i < rev[i])
            std::swap(data[i], data[rev[i]]);
    return ;
}

void fft(std::vector<std::complex<double>>& data) {
    bit_reverse(data);
    int n = data.size();
    for(int h = 2; h <= n; h <<= 1) {
        Complex wn(cos(2 * PI / h), sin(2 * PI / h));
        for(int j = 0; j < n; j += h) {
            Complex w(1, 0);
            for(int k = j; k < j + h / 2; k++) {
                Complex u = data[k];
                Complex t = w * data[k + h / 2];
                data[k] = u + t;
                data[k + h / 2] = u - t;
                printf("# %.3f %.3f\n", w.real(), w.imag());
                w = w * wn;
            }
        }
        printf("##############\n");
        for(int i = 0; i < n; i++)
            printf("%.3f %.3f\n", data[i].real(), data[i].imag());
    }
    return ;
}

// 并行 FFT（Cooley-Tukey，使用 OpenMP）,请补充
void fft_openmp(std::vector<std::complex<double>>& data) {
    bit_reverse(data);
    int n = data.size();
    static std::vector<Complex> w;
    if(w.size() < n / 2) {
        w.resize(n / 2);
        #pragma omp parallel for schedule(static)
        for(int j = 0; j < n / 2; j++) {
            double angle = (2 * PI / n) * j;
            w[j] = Complex(cos(angle), sin(angle));
        }
        
            
    }
    #pragma omp parallel
    {
        for(int h = 2; h <= n; h <<= 1) {
            int len = n / h, half = h / 2;
            #pragma omp for schedule(static)
            for(int j = 0; j < n; j += h) {
                #pragma omp simd
                for(int k = 0; k < half; k++) {
                    Complex u = data[j + k];
                    Complex t = w[k * len] * data[j + k + half];
                    data[j + k] = u + t;
                    data[j + k + half] = u - t;
                }
            }
        }
    }
    return ;
}

int main() {
    int n;
    std::cin >> n;
    if ((n & (n - 1)) != 0) {
        std::cerr << "Error: n must be a power of 2.\n";
        return 1;
    }

    std::vector<std::complex<double>> data(n);
    double real, imag;

    for (int i = 0; i < n; ++i) {
        std::cin >> real;
        data[i] = std::complex<double>(real, 0.0);
    }
    for (int i = 0; i < n; ++i) {
        std::cin >> imag;
        data[i].imag(imag);
    }
    // 执行 FFT
    double st = omp_get_wtime();
    // fft_openmp(data);
    fft(data);
    double en = omp_get_wtime();
    // std::cerr << en - st << std::endl;

    // 输出结果
    for (int i = 0; i < n; ++i) {
        printf("%.0f %.0f\n", data[i].real(), data[i].imag());
    }

    return 0;
}
