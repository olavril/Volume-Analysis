#include <tuple>
#include <vector>
#include <numbers>
#include <stdint.h>
#include <cmath>
#include <iostream>

std::tuple<int,int,int> select_start(const uint64_t iTest)
{   
    // start offsets that will be selected
    int ixstart = 0;
    int iystart = 0;
    int izstart = 0;
    // compute the modulo to decide which start setting should be used
    uint64_t select = iTest % 8;
    // select the start setting
    switch (select)
    {
    case 1:
        ixstart = 1;
        break;
    case 2:
        iystart = 1;
        break;
    case 3:
        ixstart = 1;
        iystart = 1;
        break;
    case 4:
        izstart = 1;
        break;
    case 5:
        ixstart = 1;
        izstart = 1;
        break;
    case 6:
        iystart = 1;
        izstart = 1;
        break;
    case 7:
        ixstart = 1;
        iystart = 1;
        izstart = 1;
        break;
    default:
        break;
    }
    return std::make_tuple(ixstart,iystart,izstart);
}

std::tuple<int,int,int> select_start2(const uint64_t iTest, const uint64_t num)
{   
    // start offsets that will be selected
    int izstart = std::floor(iTest / (num*num));
    int iystart = std::floor((iTest - izstart*(num*num)) / num);
    int ixstart = static_cast<int>(iTest - izstart*(num*num) - iystart*num);
    izstart = izstart % num;

    return std::make_tuple(ixstart,iystart,izstart);
}

int select_bin(double i){
    int min = std::floor(i);
    int max = min + 1;

    double d1 = i - static_cast<double>(min);
    double d2 = static_cast<double>(max) - i;

    int result = -1;
    if (d1 < d2){
        result = min;
    } else {
        result = max;
    }
    return result;
}

int main(){
    std::vector<uint64_t> test(60);
    uint64_t num = 0;
    // for (size_t i = 0; i < test.size(); i++){
    //     test[i] = num;
    //     auto [ix,iy,iz] = select_start(num); 
    //     auto [ix2,iy2,iz2] = select_start2(num,3); 
    //     // std::cout << num << "\t:\t" << ix << "/" << ix2 << "\t" << iy << "/" << iy2 << "\t" << iz << "/" << iz2 << std::endl;
    //     std::cout << num << "\t:\t" << ix2 << "\t" << iy2 << "\t" << iz2 << std::endl;
    //     num++;
    // }

    // std::cout << "Pi: " << std::numbers << std::endl;
    double i = 2.55;
    std::cout << i << " : " << select_bin(i) << std::endl;
    return 0;
}