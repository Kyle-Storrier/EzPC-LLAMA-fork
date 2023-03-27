#ifndef __FUNCTION_CONFIG_HPP__
#define __FUNCTION_CONFIG_HPP__

#include "group_element.h"
#include "utils.h"

#include<vector>

// SIGMOID_12_12

namespace llama_config {

int sin = 12;
int sout = 12;


int ib = 64;
int ob = 64;
int cb = 64;
int degree = 2;
int scoef = 20;
int numPoly = 20;

int input_precision = 12;
int input_bitwidth = 16;
int output_precision = 12;
int output_bitwidth = 16;

std::vector<std::vector<GroupElement>> fxd_polynomials
    = {
        {GroupElement(   -29928, cb),  GroupElement(     1114308608, cb),  GroupElement(   8796093022208, cb)},
        {GroupElement(   -52222, cb),  GroupElement(     1283092480, cb),  GroupElement(   8476621275136, cb)},
        {GroupElement(   -35115, cb),  GroupElement(     1024057344, cb),  GroupElement(   9457182441472, cb)},
        {GroupElement(   -17337, cb),  GroupElement(      620269568, cb),  GroupElement(  11749956780032, cb)},
        {GroupElement(    -7459, cb),  GroupElement(      321150976, cb),  GroupElement(  14014562172928, cb)},
        {GroupElement(    -3077, cb),  GroupElement(      155246592, cb),  GroupElement(  15584590823424, cb)},
        {GroupElement(    -1232, cb),  GroupElement(       71475200, cb),  GroupElement(  16535942856704, cb)},
        {GroupElement(     -493, cb),  GroupElement(       32301056, cb),  GroupElement(  17054962810880, cb)},
        {GroupElement(     -200, cb),  GroupElement(       14553088, cb),  GroupElement(  17323733811200, cb)},
        {GroupElement(        0, cb),  GroupElement(              0, cb),  flt2fxd(1, degree*12 + scoef, cb)},

        // after x = N/2

        {GroupElement(        0, cb),  GroupElement(              0, cb),  GroupElement(               0, cb)},
        {GroupElement(      199, cb),  GroupElement(       14553088, cb),  GroupElement(    268435456000, cb)},
        {GroupElement(      492, cb),  GroupElement(       32301056, cb),  GroupElement(    537206456320, cb)},
        {GroupElement(     1231, cb),  GroupElement(       71475200, cb),  GroupElement(   1056226410496, cb)},
        {GroupElement(     3076, cb),  GroupElement(      155246592, cb),  GroupElement(   2007578443776, cb)},
        {GroupElement(     7458, cb),  GroupElement(      321150976, cb),  GroupElement(   3577607094272, cb)},
        {GroupElement(    17336, cb),  GroupElement(      620269568, cb),  GroupElement(   5842212487168, cb)},
        {GroupElement(    35114, cb),  GroupElement(     1024057344, cb),  GroupElement(   8134986825728, cb)},
        {GroupElement(    52221, cb),  GroupElement(     1283092480, cb),  GroupElement(   9115547992064, cb)},
        {GroupElement(    29927, cb),  GroupElement(     1114308608, cb),  GroupElement(   8796093022208, cb)},
    };

std::vector<GroupElement> fxd_p = 
    { GroupElement(0, ib),    GroupElement(3640, ib),    GroupElement(7281, ib),   GroupElement(10922, ib),   GroupElement(14563, ib),   GroupElement(18204, ib),   GroupElement(21845, ib),   GroupElement(25486, ib),   GroupElement(29127, ib),   GroupElement(32767, ib),
    GroupElement(-32768, ib),  GroupElement(-29128, ib),  GroupElement(-25487, ib),  GroupElement(-21846, ib),  GroupElement(-18205, ib),  GroupElement(-14564, ib),  GroupElement(-10923, ib),   GroupElement(-7282, ib),   GroupElement(-3641, ib)
    };
    
// int sin;
// int sout;


// int ib, ob;
// int cb;
// int degree;
// int scoef;
// int numPoly;

// int input_precision;
// int input_bitwidth;
// int output_precision;
// int output_bitwidth;

// std::vector<std::vector<GroupElement>> fxd_polynomials;

// std::vector<GroupElement> fxd_p;

/*int scaleIn;
int scaleOut;
int scaleCoef;
int degree;
int numPoly;*/
}

#endif // __FUNCTION_CONFIG_HPP__