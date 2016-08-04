//-----------------------------------------------------------------------------
// 作    者：ZWZ
// 描    述：
// 版    本：
//-----------------------------------------------------------------------------
// 历史更新纪录 Public element had changed by ZWZ,Please check it carefully.
//-----------------------------------------------------------------------------
// 版    本：           修改时间：2016/8/4           修改人：ZWZ          
// 修改内容：
//-----------------------------------------------------------------------------
// Copyright (C) 2016-ZHBITZWZ
//-----------------------------------------------------------------------------
#include <chrono>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include<iterator>

class Timing{

public:
	
	//std::chrono::system_clock::time_point timePoint_t = std::chrono::time_point<std::chrono::system_clock>;
	/*
	//Public element had changed by ZWZ,Please check it carefully.
	*/
	//std::chrono::time_point<std::chrono::system_clock> timePoint_t;
	//std::chrono::system_clock::time_point timePoint_t;

    Timing(){};
    void start();
    void end();
    double get_elapse();
    int output_timing_vector_to_file(std::string filename, std::vector<double> vec, int append,int pos);


private:
    //std::chrono::duration<double> _timeElapse = std::chrono::duration<double>(0);
    std::chrono::duration<double> _timeElapse;
    //timePoint_t _start;
    //timePoint_t _end;
    std::chrono::time_point<std::chrono::system_clock> _start;
	std::chrono::time_point<std::chrono::system_clock> _end;
};

void Timing::start(){
    this->_start = std::chrono::system_clock::now();//current time
}
void Timing::end(){
    this->_end = std::chrono::system_clock::now();
    this->_timeElapse = (_end - _start);
}

double Timing::get_elapse(){
    return this->_timeElapse.count();
}

int Timing::output_timing_vector_to_file(std::string filename, std::vector<double> vec, int append,int pos = -1){
    std::fstream outputFile;
    std::ostringstream ossVec;

    if(append == 0){
        outputFile.open(filename,std::fstream::out);
    }
    else{
        outputFile.open(filename, std::fstream::out | std::fstream::app);
    }

    if(outputFile.is_open()){

        if(pos != -1){
            vec.insert(vec.begin(),pos);
        }


        std::copy(vec.begin(), vec.end()-1,
        std::ostream_iterator<float>(ossVec, ","));
        ossVec << vec.back();
        outputFile<<ossVec.str()<<"\n";

        outputFile.close();
        return 1;
    }
    return 0;
}

