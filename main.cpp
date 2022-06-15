//#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/highgui.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
//#include <stdio.h>
#include <vector>       // std::vector
#include <cstdlib>
#include <sstream>

//using namespace cv;
using namespace std;
using std::vector;

namespace cpp_secrets{
///Runnable: A class which has a valid and public default ctor and a "run()" function.
///BenchmarkingTimer tests the "run()" function of Runnable
///num_run_cycles: It is the number of times run() needs to be run for a single test.
///One Runnable object is used for a single test.
///Note: if the run() function is statefull then it can only be run once for an object in order
///to get meaningful results.
///num_tests: It is the number of tests that need to be run.
    template <typename Runnable, int num_run_cycles = 1000000, int num_tests = 10>

        struct BenchmarkingTimer{
            /// runs the run() function of the Runnable object and captures timestamps around each test
            void run(){
                for(int i = 0; i < num_tests; i++){
                    Runnable runnable_object_{};
                    Timer t{intervals_[i].first, intervals_[i].second};
                    for(int i = 0; i < num_run_cycles; i++){
                        runnable_object_.run();
                    }
                }
            }

            ///utility function to print durations of all tests
            std::string durations() const{
                std::stringstream ss;
                int i{1};
                for(const auto& interval: intervals_){
                    ss << "Test-" << i++  << " duration = " << (interval.second - interval.first) * 0.001 << " ms" << std::endl;
                }
                return ss.str();
            }

            ///utility function to print average duration of all tests
            double average_duration(){
                auto duration_sum{0.0};
                for(const auto& interval: intervals_){
                    duration_sum += (interval.second - interval.first) * 0.001;
                }
                if (num_tests) return (duration_sum/num_tests);
                return 0;
            }

            private:
            std::array<std::pair<double, double>, num_tests> intervals_{};

            struct Timer{
                Timer(double& start, double& finish):finish_(finish) { start = now(); }
                ~Timer() { finish_ = now(); }

                private:
                double& finish_;
                double now(){
                         ///utility function to return current time in microseconds since epoch
                    return std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
                }
            };

        };
}

///sample class which has a statefull run().
//run() function is stateful because it is not meaningful to sort a sorted array.
//that's why num_run_cycles = 1 in this case.
struct randomly_sorted{
    randomly_sorted(){
        srand(time(0));
        for(int i=0;i<1000000;i++){
            arr_.emplace_back(rand());
                  // making a vector filled with random elements
        }
    }

    void run(){
        sort(arr_.begin(), arr_.end(), std::less<int>());
    }
    private:
    std::vector<int>arr_;
};

//-----------------Funcion clahe-----------------------
int claheGO(vector<vector<int>>src,int _step = 8)
{
    //Mat CLAHE_GO = src.clone();
    vector<vector<int>> CLAHE_GO(src.size());
    copy(src.begin(), src.end(), CLAHE_GO.begin());
    int block = _step;//pblock
    int filas = (sizeof(src)/sizeof(src[0]));
    int columnas = (sizeof(src[0])/sizeof(src[0][0]));
    int width = filas;
    int height= columnas;
    int width_block = width/block;
    int height_block = height/block;

    // almacenar cada histograma
    int tmp2[8*8][256] ={0};
    float C2[8*8][256] = {0.0};
    //Bloque
    int total = width_block * height_block;
    //#pragma omp parallel for collapse (2)

    for (int i=0;i<block;i++)
    {
        for (int j=0;j<block;j++)
        {
            int start_x = i*width_block;
            int end_x = start_x + width_block;
            int start_y = j*height_block;
            int end_y = start_y + height_block;
            int num = i+block*j;
            // Recorrer los bloques pequeños y calcular el histograma
            for(int ii = start_x ; ii < end_x ; ii++)
            {
                for(int jj = start_y ; jj < end_y ; jj++)
                {
                    //int index[i][j] = src[jj][ii];
                    int index =src[jj][ii];
                    tmp2[num][index]++;
                }
            }
            //Operaciones de cortar y sumar, es decir, la parte cl en clahe
            //Los parámetros aquí corresponden a "Gem" por encima de fCliplimit = 4 , uiNrBins = 255
            int average = width_block * height_block / 255;
            //Es necesario discutir cómo elegir los parámetros. Discutir los diferentes resultados.
            //Cuando se trata de la situación global, es necesario discutir cómo calcular este cl aquí
            int LIMIT = 40 * average;
            int steal = 0;
            for(int k = 0 ; k < 256 ; k++)
            {
                if(tmp2[num][k] > LIMIT){
                    steal += tmp2[num][k] - LIMIT;
                    tmp2[num][k] = LIMIT;
                }
            }
            int bonus = steal/256;
            // repartir los robos en promedio
            for(int k = 0 ; k < 256 ; k++)
            {
                tmp2[num][k] += bonus;
            }
            //Calcular el histograma de distribución acumulada
            for(int k = 0 ; k < 256 ; k++)
            {
                if( k == 0)
                    C2[num][k] = 1.0f * tmp2[num][k] / total;
                else
                    C2[num][k] = C2[num][k-1] + 1.0f * tmp2[num][k] / total;
            }
        }
    }
    //Calcular el valor del píxel transformado
    //Según la posición del píxel, elija un método de cálculo diferente
    for(int  i = 0 ; i < width; i++)
    {
        for(int j = 0 ; j < height; j++)
        {
            //four coners //cuatro conos
            if(i <= width_block/2 && j <= height_block/2)
            {
                int num = 0;
                CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 255);
            }else if(i <= width_block/2 && j >= ((block-1)*height_block + height_block/2)){
                int num = block*(block-1);
                CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 255);
            }else if(i >= ((block-1)*width_block+width_block/2) && j <= height_block/2){
                int num = block-1;
                CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 255);
            }else if(i >= ((block-1)*width_block+width_block/2) && j >= ((block-1)*height_block + height_block/2)){
                int num = block*block-1;
                CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 255);
            }
            //four edges except coners
            //cuatro aristas excepto conos
            else if( i <= width_block/2 )
            {
                //Interpolación linear
                int num_i = 0;
                int num_j = (j - height_block/2)/height_block;
                int num1 = num_j*block + num_i;
                int num2 = num1 + block;
                float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);
                float q = 1-p;
                CLAHE_GO[j][i] = (int)((q*C2[num1][CLAHE_GO[j][i]]+ p*C2[num2][CLAHE_GO[j][i]])* 255);
            }else if( i >= ((block-1)*width_block+width_block/2)){
                //Interpolación linear
                int num_i = block-1;
                int num_j = (j - height_block/2)/height_block;
                int num1 = num_j*block + num_i;
                int num2 = num1 + block;
                float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);
                float q = 1-p;
                CLAHE_GO[j][i] = (int)((q*C2[num1][CLAHE_GO[j][i]]+ p*C2[num2][CLAHE_GO[j][i]])* 255);
            }else if( j <= height_block/2 ){
                //Interpolación linear
                int num_i = (i - width_block/2)/width_block;
                int num_j = 0;
                int num1 = num_j*block + num_i;
                int num2 = num1 + 1;
                float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);
                float q = 1-p;
                CLAHE_GO[j][i] = (int)((q*C2[num1][CLAHE_GO[j][i]]+ p*C2[num2][CLAHE_GO[j][i]])* 255);
            }else if( j >= ((block-1)*height_block + height_block/2) ){
                //Interpolación linear
                int num_i = (i - width_block/2)/width_block;
                int num_j = block-1;
                int num1 = num_j*block + num_i;
                int num2 = num1 + 1;
                float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);
                float q = 1-p;
                CLAHE_GO[j][i] = (int)((q*C2[num1][CLAHE_GO[j][i]]+ p*C2[num2][CLAHE_GO[j][i]])* 255);
            }
            // interpolación bilineal
            else{
                int num_i = (i - width_block/2)/width_block;
                int num_j = (j - height_block/2)/height_block;
                int num1 = num_j*block + num_i;
                int num2 = num1 + 1;
                int num3 = num1 + block;
                int num4 = num2 + block;
                float u = (i - (num_i*width_block+width_block/2))/(1.0f*width_block);
                float v = (j - (num_j*height_block+height_block/2))/(1.0f*height_block);
                CLAHE_GO[j][i] = (int)((u*v*C2[num4][CLAHE_GO[j][i]] +
                    (1-v)*(1-u)*C2[num1][CLAHE_GO[j][i]] +
                    u*(1-v)*C2[num2][CLAHE_GO[j][i]] +
                    v*(1-u)*C2[num3][CLAHE_GO[j][i]]) * 255);
            }
            //smooth  //suavizamiento
            CLAHE_GO[j][i] = CLAHE_GO[j][i] + (CLAHE_GO[j][i] << 8) + (CLAHE_GO[j][i] << 16);
        }
    }
  return 0;
}


int main()
{

vector<vector<int>> src{ { 1, 2, 3 },
                         { 4, 5, 6 },
                         { 7, 8, 9, 4 } };
vector<vector<int>> original(src.size());

int filas = (sizeof(src)/sizeof(src[0]));
int columnas = (sizeof(src[0])/sizeof(src[0][0]));


cpp_secrets::BenchmarkingTimer<randomly_sorted, 1, 10> test; // randomly_sorted structure run function is run 10 time and average output is given.
        test.run();
        std::cout << test.durations() << std::endl; // outputs the duration of every test.
        std::cout << "average duration = " << test.average_duration() << " ms" << std::endl;
cout << "proceso finalizado " << endl;
claheGO(src);

return 0;
}
