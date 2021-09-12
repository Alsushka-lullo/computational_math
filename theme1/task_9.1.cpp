#include <iostream>
#include <cmath>
#include <vector>

double MaclaurinMethod(float x) {
   float sum = 0, term = 1;
   int step = 1;

   while (sum + term != sum) {
       //std::cout << sum << " " << term << "\n";
       sum += term;
       term *= x / float(step);
       step++;
   }

   return sum;
}

int main() {
    std::vector<float> x_values = {1, 5, 10, 15, 20, 25, -1, -5, -10, -15, -20, -25};

    for(auto x : x_values) {
        std::cout << x << " " << MaclaurinMethod(x) << " " << exp(x) << "\n";
    }

    /* Результаты:
        1 2.71828 2.71828
        5 148.413 148.413
        10 22026.5 22026.5
        15 3.26902e+06 3.26902e+06
        20 4.85165e+08 4.85165e+08
        25 7.20049e+10 7.20049e+10
        -1 0.367879 0.367879
        -5 0.00673714 0.00673795
        -10 -5.23423e-05 4.53999e-05
        -15 -0.0223869 3.05902e-07
        -20 -1.79703 2.06115e-09
        -25 -737.664 1.38879e-11

        Как можно видеть, для положительных x экспонента вычисляется рядом маклорена очень хорошо, потому что
        все слагаемые в в ряду положительные и по модулю меньше предыдущего (убывает очень быстро как a^n/n!).
        Для больших отрицательных x результат получается очень сильно отличным от реального,
        потому что ряд теперь знакопеременный, т.е мы начинаем вычитать друг из друга большие числа, рассчитывая
        получить маленькие, что получается плохо из-за ошибки округления.

        Можно предложить простой метод вычисления экспоненты отрицательного аргумента: вычислять exp(-x), а затем
        exp(x) = 1 / exp(-x)
     */

    std::cout << "Now let's calculate exp(x) as exp(x) = 1 / exp(-x) \n";

    for(auto x : x_values) {
        if (x < 0) {
            std::cout << x << " " << 1/MaclaurinMethod(-x) << " " << exp(x) << "\n";
        }
    }

    /* Результаты:
        -1 0.367879 0.367879
        -5 0.00673795 0.00673795
        -10 4.53999e-05 4.53999e-05
        -15 3.05902e-07 3.05902e-07
        -20 2.06115e-09 2.06115e-09
        -25 1.38879e-11 1.38879e-11

        Вычисленные таким способом значения оказались равны (с нашей точностью) истинным.
    */

    return 0;
}