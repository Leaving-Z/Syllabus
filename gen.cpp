#include<bits/stdc++.h>
using namespace std;
int main() {
    int len = 1 << 21;
    const int lim = 50;
    printf("%d\n", len);
    mt19937 rnd(time(0));
    for(int i = 0; i < len; i++) {
        printf("%d%c", rnd() % lim, " \n"[i == len - 1]);
    }
    for(int i = 0; i < len; i++) {
        printf("%d%c", rnd() % lim, " \n"[i == len - 1]);
    }
    return 0;
}