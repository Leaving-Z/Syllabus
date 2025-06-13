#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <queue>
#include <mpi.h>

/* ---------- 帮助函数 ---------- */
uint32_t read_uint32_be(std::istream& in) {
    uint8_t b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}
#define K 10
/* ---------- 读取图像 ---------- */
std::vector<std::vector<uint8_t>>
load_mnist_images(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path);

    uint32_t magic = read_uint32_be(in);
    if (magic != 2051) throw std::runtime_error("Wrong magic (images)");

    uint32_t num   = read_uint32_be(in);
    uint32_t rows  = read_uint32_be(in);
    uint32_t cols  = read_uint32_be(in);

    std::vector<std::vector<uint8_t>> images(num, std::vector<uint8_t>(rows * cols));
    for (uint32_t i = 0; i < num; ++i)
        in.read(reinterpret_cast<char*>(images[i].data()), rows * cols);

    return images;
}
struct node{
    double dis;
    int label;
};
bool operator < (const node &x, const node &y) {
    return x.dis < y.dis;
}
/* ---------- 读取标签 ---------- */
std::vector<uint8_t>
load_mnist_labels(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path);

    uint32_t magic = read_uint32_be(in);
    if (magic != 2049) throw std::runtime_error("Wrong magic (labels)");

    uint32_t num = read_uint32_be(in);

    std::vector<uint8_t> labels(num);
    in.read(reinterpret_cast<char*>(labels.data()), num);

    return labels;
}
double calc(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
    double res = 0;
    for(int i = 0; i < a.size(); i++)
        res += (a[i] / 255.0 - b[i] / 255.0) * (a[i] / 255.0 - b[i] / 255.0);
    return res;
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double t0 = MPI_Wtime();
    std::vector<std::vector<uint8_t>> train_imgs, test_imgs;
    std::vector<uint8_t> train_lbls, test_lbls, train_imgs_flat, test_imgs_flat;
    int demensions, test_num, train_num;

    std::cerr << "Hello" << std::endl;





    if(rank == 0) {
        train_imgs = load_mnist_images("train-images-idx3-ubyte");
        train_lbls = load_mnist_labels("train-labels-idx1-ubyte");
        train_num = train_imgs.size();
        for(auto &img:train_imgs) 
            for(auto &x:img)
                train_imgs_flat.push_back(x);
        test_imgs = load_mnist_images("t10k-images-idx3-ubyte");
        test_lbls = load_mnist_labels("t10k-labels-idx1-ubyte");
        test_num = test_imgs.size();
        for(auto &img:test_imgs) 
            for(auto &x:img)
                test_imgs_flat.push_back(x);
        demensions = train_imgs[0].size();

        std::cerr << "rand[" << rank << "] " << train_num << std::endl;
        std::cerr << "rand[" << rank << "] " << test_num << std::endl;
        std::cerr << "rand[" << rank << "] " << demensions << std::endl;

        MPI_Bcast(&demensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&test_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&train_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::cerr << "rand[" << rank << "] " << train_imgs_flat.size() << std::endl;
        std::cerr << "rand[" << rank << "] " << test_imgs_flat.size() << std::endl;
        MPI_Bcast(train_imgs_flat.data(), train_imgs_flat.size(), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(test_imgs_flat.data(), test_imgs_flat.size(), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(train_lbls.data(), train_lbls.size(), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(test_lbls.data(), test_lbls.size(), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        
    } else {
        MPI_Bcast(&demensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&test_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&train_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        train_imgs_flat.resize(demensions * train_num);
        test_imgs_flat.resize(demensions * test_num);
        train_lbls.resize(train_num);
        test_lbls.resize(test_num);
        MPI_Bcast(train_imgs_flat.data(), train_imgs_flat.size(), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(test_imgs_flat.data(), test_imgs_flat.size(), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(train_lbls.data(), train_lbls.size(), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(test_lbls.data(), test_lbls.size(), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        std::cerr << "rand[" << rank << "] " << train_imgs_flat.size() << std::endl;
        std::cerr << "rand[" << rank << "] " << test_imgs_flat.size() << std::endl;
        train_imgs.resize(train_num);
        for(int samplenum = 0; samplenum < train_num; samplenum++) {
            train_imgs[samplenum].resize(demensions);
            for(int k = 0; k < demensions; k++)
                train_imgs[samplenum][k] = train_imgs_flat[samplenum * demensions + k];
        }
        test_imgs.resize(test_num);
        for(int samplenum = 0; samplenum < test_num; samplenum++) {
            test_imgs[samplenum].resize(demensions);
            for(int k = 0; k < demensions; k++)
                test_imgs[samplenum][k] = test_imgs_flat[samplenum * demensions + k];
        }
        std::cerr << "rand[" << rank << "] " << "process Complete." << std::endl;
    }
    
    std::string log_filename = "thread_" + std::to_string(rank) + "_log.txt";
    std::ofstream log_file(log_filename);
     if (!log_file) {
        std::cerr << "无法打开文件 " << log_filename << " 进行写入。" << std::endl;
        MPI_Finalize();
        return 0;
    }
    log_file << "START LOG, ThreadID = " << rank << std::endl;

    int correct_num = 0;
    for(int i = rank; i < test_num; i += size) {
        std::priority_queue <node> topK;
        uint8_t mxlabel = 0;
        for(int j = 0; j < train_num; j++) {
            node tmp;
            tmp.label = train_lbls[j];
            mxlabel = std::max(mxlabel, train_lbls[j]);
            tmp.dis = calc(test_imgs[i], train_imgs[j]);
            topK.push(tmp);
            if(topK.size() > K)
                topK.pop();
        }
        // std::cerr << "###" << int(mxlabel) << std::endl;
        double totaldis[20];
        int book[20];
        memset(totaldis, 0, sizeof(totaldis));
        memset(book, 0 ,sizeof(book));
        while(topK.size()) {
            auto tmp = topK.top();
            topK.pop();
            totaldis[tmp.label] += tmp.dis;
            book[tmp.label] ++;
        }
        int res = 0;
        for(int i = 1; i < 10; i++)
        if(book[i] > book[res] || book[i] == book[res] && totaldis[i] < totaldis[res])
            res = i;
        if(res == test_lbls[i])
            correct_num++;
        log_file << "Test " << i << " result " << res << std::endl;
    }
    int total_correct = 0;
    MPI_Reduce(&correct_num, &total_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0) {
        double t1 = MPI_Wtime();
        printf("%d testsamples, %d trainsamples\n", test_num, train_num);
        printf("Precision: %7d / %7d %.3f%%\n", total_correct, test_num, 100.0* total_correct / test_num);
        printf("Total wall time %.3f s\n", t1 - t0);
    }
    log_file.close();
    MPI_Finalize();
    return 0;
}
