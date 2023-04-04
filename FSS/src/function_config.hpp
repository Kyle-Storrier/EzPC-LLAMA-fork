#ifndef __FUNCTION_CONFIG_HPP__
#define __FUNCTION_CONFIG_HPP__

#include "group_element.h"
#include "utils.h"

#include "config.h"

#include <vector>


namespace llama_config
{

#if defined(TANH_12_12) // || defined(SIGMOID_TANH_37)
    std::string function_id = "LLAMA_TANH_LLAMA_12_12";
    std::string function_name = "TANH";
    std::string lut_src = "LLAMA";
    int sin = 12;
    int sout = 12;

    int ib = 64;
    int ob = 64;
    int cb = 64;
    int degree = 2;
    int scoef = 20;
    int numPoly = 20;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-29928, cb), GroupElement(1114308608, cb), GroupElement(8796093022208, cb)},
        {GroupElement(-52222, cb), GroupElement(1283092480, cb), GroupElement(8476621275136, cb)},
        {GroupElement(-35115, cb), GroupElement(1024057344, cb), GroupElement(9457182441472, cb)},
        {GroupElement(-17337, cb), GroupElement(620269568, cb), GroupElement(11749956780032, cb)},
        {GroupElement(-7459, cb), GroupElement(321150976, cb), GroupElement(14014562172928, cb)},
        {GroupElement(-3077, cb), GroupElement(155246592, cb), GroupElement(15584590823424, cb)},
        {GroupElement(-1232, cb), GroupElement(71475200, cb), GroupElement(16535942856704, cb)},
        {GroupElement(-493, cb), GroupElement(32301056, cb), GroupElement(17054962810880, cb)},
        {GroupElement(-200, cb), GroupElement(14553088, cb), GroupElement(17323733811200, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), flt2fxd(1, degree * 12 + scoef, cb)},

        // after x = N/2

        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb)},
        {GroupElement(199, cb), GroupElement(14553088, cb), GroupElement(268435456000, cb)},
        {GroupElement(492, cb), GroupElement(32301056, cb), GroupElement(537206456320, cb)},
        {GroupElement(1231, cb), GroupElement(71475200, cb), GroupElement(1056226410496, cb)},
        {GroupElement(3076, cb), GroupElement(155246592, cb), GroupElement(2007578443776, cb)},
        {GroupElement(7458, cb), GroupElement(321150976, cb), GroupElement(3577607094272, cb)},
        {GroupElement(17336, cb), GroupElement(620269568, cb), GroupElement(5842212487168, cb)},
        {GroupElement(35114, cb), GroupElement(1024057344, cb), GroupElement(8134986825728, cb)},
        {GroupElement(52221, cb), GroupElement(1283092480, cb), GroupElement(9115547992064, cb)},
        {GroupElement(29927, cb), GroupElement(1114308608, cb), GroupElement(8796093022208, cb)},
    };

    std::vector<GroupElement> fxd_p =
        {GroupElement(0, ib), GroupElement(3640, ib), GroupElement(7281, ib), GroupElement(10922, ib), GroupElement(14563, ib), GroupElement(18204, ib), GroupElement(21845, ib), GroupElement(25486, ib), GroupElement(29127, ib), GroupElement(32767, ib),
         GroupElement(-32768, ib), GroupElement(-29128, ib), GroupElement(-25487, ib), GroupElement(-21846, ib), GroupElement(-18205, ib), GroupElement(-14564, ib), GroupElement(-10923, ib), GroupElement(-7282, ib), GroupElement(-3641, ib)};
#elif defined(TANH_9_9) // scale is not 12 (input scale 9, output scale 9)
    std::string function_id = "LLAMA_TANH_LLAMA_9_9";
    std::string function_name = "TANH";
    std::string lut_src = "LLAMA";

    int ib = 64, ob = 64, sin = 9, scoef = 18, sout = 9, degree = 2, numPoly = 12;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;
    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-82639, cb), GroupElement(144710656, cb), GroupElement(0, cb)},
        {GroupElement(-85271, cb), GroupElement(146579968, cb), GroupElement(-331874304, cb)},
        {GroupElement(-25412, cb), GroupElement(61579776, cb), GroupElement(29842997248, cb)},
        {GroupElement(-8319, cb), GroupElement(25172480, cb), GroupElement(49230118912, cb)},
        {GroupElement(-1192, cb), GroupElement(4930048, cb), GroupElement(63602163712, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), flt2fxd(1, degree *sin + scoef, cb)},

        // after x=N/2

        {GroupElement(0, cb), GroupElement(0, cb), flt2fxd(-1, degree *sin + scoef, cb)},
        {GroupElement(1191, cb), GroupElement(4930048, cb), GroupElement(-63602425856, cb)},
        {GroupElement(8318, cb), GroupElement(25172480, cb), GroupElement(-49230381056, cb)},
        {GroupElement(25411, cb), GroupElement(61579776, cb), GroupElement(-29843259392, cb)},
        {GroupElement(85270, cb), GroupElement(146579968, cb), GroupElement(331612160, cb)},
        {GroupElement(82638, cb), GroupElement(144710656, cb), GroupElement(0, cb)}

    };

    std::vector<GroupElement> fxd_p{GroupElement(0, ib), GroupElement(355, ib), GroupElement(710, ib), GroupElement(1065, ib), GroupElement(1420, ib), GroupElement(1775, ib), GroupElement(32767, ib),
                                    GroupElement(-1775, ib), GroupElement(-1420, ib), GroupElement(-1065, ib), GroupElement(-710, ib), GroupElement(-355, ib)};
#elif defined(TANH_8_8)
    std::string function_id = "LLAMA_TANH_LLAMA_8_8";
    std::string function_name = "TANH";
    int ib = 64, ob = 64, sin = 8, scoef = 18, sout = 8, degree = 2, numPoly = 10;
    std::string lut_src = "LLAMA";

    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-87883, cb), GroupElement(73233920, cb), GroupElement(-65536, cb)},
        {GroupElement(-74280, cb), GroupElement(67799296, cb), GroupElement(542769152, cb)},
        {GroupElement(-15013, cb), GroupElement(20444928, cb), GroupElement(10001776640, cb)},
        {GroupElement(-6420, cb), GroupElement(10146304, cb), GroupElement(13087539200, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), flt2fxd(1, degree *sin + scoef, cb)},
        // after x=N/2
        {GroupElement(0, cb), GroupElement(0, cb), flt2fxd(-1, degree *sin + scoef, cb)},
        {GroupElement(6419, cb), GroupElement(10146304, cb), GroupElement(-13087604736, cb)},
        {GroupElement(15012, cb), GroupElement(20444928, cb), GroupElement(-10001842176, cb)},
        {GroupElement(74279, cb), GroupElement(67799296, cb), GroupElement(-542834688, cb)},
        {GroupElement(87882, cb), GroupElement(73233920, cb), GroupElement(-65536, cb)}};

    std::vector<GroupElement> fxd_p{
        GroupElement(0, ib),
        GroupElement(199, ib),
        GroupElement(399, ib),
        GroupElement(599, ib),
        GroupElement(799, ib),
        GroupElement(32767, ib),
        // after x=N/2
        GroupElement(-799, ib),
        GroupElement(-600, ib),
        GroupElement(-400, ib),
        GroupElement(-200, ib)};
#elif defined(TANH_11_11)
    std::string function_id = "LLAMA_TANH_LLAMA_11_11";
    std::string function_name = "TANH";
    int ib = 64, ob = 64, sin = 11, scoef = 18, sout = 11, degree = 2, numPoly = 20;
    std::string lut_src = "LLAMA";

    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-59861, cb), GroupElement(557158400, cb), GroupElement(-4194304, cb)},
        {GroupElement(-104442, cb), GroupElement(641546240, cb), GroupElement(-39938162688, cb)},
        {GroupElement(-70219, cb), GroupElement(511983616, cb), GroupElement(82686509056, cb)},
        {GroupElement(-34664, cb), GroupElement(310079488, cb), GroupElement(369325244416, cb)},
        {GroupElement(-14913, cb), GroupElement(160530432, cb), GroupElement(652403015680, cb)},
        {GroupElement(-6150, cb), GroupElement(77594624, cb), GroupElement(848637722624, cb)},
        {GroupElement(-2463, cb), GroupElement(35721216, cb), GroupElement(967529463808, cb)},
        {GroupElement(-985, cb), GroupElement(16142336, cb), GroupElement(1032385986560, cb)},
        {GroupElement(-399, cb), GroupElement(7270400, cb), GroupElement(1065973972992, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), flt2fxd(1, degree *sin + scoef, cb)},
        // after x=N/2
        {GroupElement(0, cb), GroupElement(0, cb), flt2fxd(-1, degree *sin + scoef, cb)},
        {GroupElement(398, cb), GroupElement(7270400, cb), GroupElement(-1065978167296, cb)},
        {GroupElement(984, cb), GroupElement(16142336, cb), GroupElement(-1032390180864, cb)},
        {GroupElement(2462, cb), GroupElement(35721216, cb), GroupElement(-967533658112, cb)},
        {GroupElement(6149, cb), GroupElement(77594624, cb), GroupElement(-848641916928, cb)},
        {GroupElement(14912, cb), GroupElement(160530432, cb), GroupElement(-652407209984, cb)},
        {GroupElement(34663, cb), GroupElement(310079488, cb), GroupElement(-369329438720, cb)},
        {GroupElement(70218, cb), GroupElement(511983616, cb), GroupElement(-82690703360, cb)},
        {GroupElement(104441, cb), GroupElement(641546240, cb), GroupElement(39933968384, cb)},
        {GroupElement(59860, cb), GroupElement(557158400, cb), GroupElement(-4194304, cb)},
    };

    std::vector<GroupElement> fxd_p{
        GroupElement(0, ib),
        GroupElement(946, ib),
        GroupElement(1892, ib),
        GroupElement(2839, ib),
        GroupElement(3785, ib),
        GroupElement(4732, ib),
        GroupElement(5678, ib),
        GroupElement(6625, ib),
        GroupElement(7571, ib),
        GroupElement(8518, ib),
        GroupElement(32767, ib),
        // after x=N/2
        GroupElement(-8518, ib),
        GroupElement(-7572, ib),
        GroupElement(-6626, ib),
        GroupElement(-5679, ib),
        GroupElement(-4733, ib),
        GroupElement(-3786, ib),
        GroupElement(-2840, ib),
        GroupElement(-1893, ib),
        GroupElement(-947, ib)};
#elif defined(TANH_13_13)
    std::string function_id = "LLAMA_TANH_LLAMA_13_13";
    std::string function_name = "TANH";
    int ib = 64, ob = 64, sin = 13, scoef = 18, sout = 13, degree = 2, numPoly = 12;
    std::string lut_src = "LLAMA";

    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-37436, cb), GroupElement(2177662976, cb), GroupElement(-67108864, cb)},
        {GroupElement(-90509, cb), GroupElement(2426101760, cb), GroupElement(-290782707712, cb)},
        {GroupElement(-101496, cb), GroupElement(2528968704, cb), GroupElement(-531569311744, cb)},
        {GroupElement(-84129, cb), GroupElement(2285068288, cb), GroupElement(324739792896, cb)},
        {GroupElement(-59163, cb), GroupElement(1817600000, cb), GroupElement(2513025630208, cb)},
        {GroupElement(-37872, cb), GroupElement(1319256064, cb), GroupElement(5429039988736, cb)},
        {GroupElement(-22968, cb), GroupElement(900677632, cb), GroupElement(8368206905344, cb)},
        {GroupElement(-13507, cb), GroupElement(590635008, cb), GroupElement(10908076081152, cb)},
        {GroupElement(-7805, cb), GroupElement(377094144, cb), GroupElement(12907316248576, cb)},
        {GroupElement(-4465, cb), GroupElement(236412928, cb), GroupElement(14389079965696, cb)},
        {GroupElement(-2540, cb), GroupElement(146276352, cb), GroupElement(15443829981184, cb)},
        {GroupElement(-1442, cb), GroupElement(89726976, cb), GroupElement(16171894046720, cb)},
        {GroupElement(-814, cb), GroupElement(54444032, cb), GroupElement(16667358789632, cb)},
        {GroupElement(-470, cb), GroupElement(33513472, cb), GroupElement(16985790349312, cb)},
        // blank area b/w 2^16-1 and -2^16
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb)},
        {GroupElement(469, cb), GroupElement(33513472, cb), GroupElement(-16985857458176, cb)},
        {GroupElement(813, cb), GroupElement(54444032, cb), GroupElement(-16667425898496, cb)},
        {GroupElement(1441, cb), GroupElement(89726976, cb), GroupElement(-16171961155584, cb)},
        {GroupElement(2539, cb), GroupElement(146276352, cb), GroupElement(-15443897090048, cb)},
        {GroupElement(4464, cb), GroupElement(236412928, cb), GroupElement(-14389147074560, cb)},
        {GroupElement(7804, cb), GroupElement(377094144, cb), GroupElement(-12907383357440, cb)},
        {GroupElement(13506, cb), GroupElement(590635008, cb), GroupElement(-10908143190016, cb)},
        {GroupElement(22967, cb), GroupElement(900677632, cb), GroupElement(-8368274014208, cb)},
        {GroupElement(37871, cb), GroupElement(1319256064, cb), GroupElement(-5429107097600, cb)},
        {GroupElement(59162, cb), GroupElement(1817600000, cb), GroupElement(-2513092739072, cb)},
        {GroupElement(84128, cb), GroupElement(2285068288, cb), GroupElement(-324806901760, cb)},
        {GroupElement(101495, cb), GroupElement(2528968704, cb), GroupElement(531502202880, cb)},
        {GroupElement(90508, cb), GroupElement(2426101760, cb), GroupElement(290715598848, cb)},
        {GroupElement(37435, cb), GroupElement(2177662976, cb), GroupElement(-67108864, cb)},
    };

    std::vector<GroupElement> fxd_p{
        GroupElement(0, ib),
        GroupElement(2340, ib),
        GroupElement(4681, ib),
        GroupElement(7021, ib),
        GroupElement(9362, ib),
        GroupElement(11702, ib),
        GroupElement(14043, ib),
        GroupElement(16384, ib),
        GroupElement(18724, ib),
        GroupElement(21065, ib),
        GroupElement(23405, ib),
        GroupElement(25746, ib),
        GroupElement(28086, ib),
        GroupElement(30427, ib),
        GroupElement(32767, ib),
        // blank area b/w 2^16-1 and -2^16 because ib=64 and actual inp bitlen = 16
        GroupElement(-32768, ib),
        GroupElement(-30428, ib),
        GroupElement(-28087, ib),
        GroupElement(-25747, ib),
        GroupElement(-23406, ib),
        GroupElement(-21066, ib),
        GroupElement(-18725, ib),
        GroupElement(-16384, ib),
        GroupElement(-14044, ib),
        GroupElement(-11703, ib),
        GroupElement(-9363, ib),
        GroupElement(-7022, ib),
        GroupElement(-4682, ib),
        GroupElement(-2341, ib)};
#elif defined(TANH_GROTTO_9_9)
    std::string function_id = "LLAMA_TANH_GROTTO_9_9";
    std::string function_name = "TANH";
    int ib = 64, ob = 64, sin = 9, scoef = 9, sout = 9, degree = 3, numPoly = 20;
    std::string lut_src = "GROTTO";

    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;
    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(65371, cb), GroupElement(0, cb), GroupElement(133955584, cb), GroupElement(0, cb)},
        {GroupElement(65459, cb), GroupElement(33518080, cb), GroupElement(139460608, cb), GroupElement(8795824586752, cb)},
        {GroupElement(17, cb), GroupElement(33435648, cb), GroupElement(163577856, cb), GroupElement(8793542885376, cb)},
        {GroupElement(55, cb), GroupElement(33386496, cb), GroupElement(185073664, cb), GroupElement(8790321659904, cb)},
        {GroupElement(31, cb), GroupElement(33440768, cb), GroupElement(142868480, cb), GroupElement(5234491392, cb)},
        {GroupElement(12, cb), GroupElement(33499136, cb), GroupElement(82837504, cb), GroupElement(25769803776, cb)},
        {GroupElement(3, cb), GroupElement(33536000, cb), GroupElement(33816576, cb), GroupElement(47513075712, cb)},
        {GroupElement(0, cb), GroupElement(33551360, cb), GroupElement(7602176, cb), GroupElement(62545461248, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(262144, cb), GroupElement(68316823552, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(68585259008, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(8727507763200, cb)},
        {GroupElement(0, cb), GroupElement(512, cb), GroupElement(2359296, cb), GroupElement(8729655246848, cb)},
        {GroupElement(1, cb), GroupElement(9216, cb), GroupElement(19136512, cb), GroupElement(8740661100544, cb)},
        {GroupElement(7, cb), GroupElement(35328, cb), GroupElement(57933824, cb), GroupElement(8759988453376, cb)},
        {GroupElement(21, cb), GroupElement(85504, cb), GroupElement(115605504, cb), GroupElement(8782268596224, cb)},
        {GroupElement(45, cb), GroupElement(147456, cb), GroupElement(170917888, cb), GroupElement(2415919104, cb)},
        {GroupElement(52, cb), GroupElement(163840, cb), GroupElement(182976512, cb), GroupElement(5368709120, cb)},
        {GroupElement(65527, cb), GroupElement(90112, cb), GroupElement(153616384, cb), GroupElement(1476395008, cb)},
        {GroupElement(65424, cb), GroupElement(16896, cb), GroupElement(135790592, cb), GroupElement(0, cb)},
        {GroupElement(65371, cb), GroupElement(0, cb), GroupElement(133955584, cb), GroupElement(0, cb)},
    };

    std::vector<GroupElement> fxd_p{
        GroupElement(0, ib),
        GroupElement(133, ib),
        GroupElement(290, ib),
        GroupElement(464, ib),
        GroupElement(776, ib),
        GroupElement(1027, ib),
        GroupElement(1331, ib),
        GroupElement(1739, ib),
        GroupElement(2405, ib),
        GroupElement(4215, ib),
        GroupElement(-32768, ib),
        GroupElement(-2951, ib),
        GroupElement(-1988, ib),
        GroupElement(-1499, ib),
        GroupElement(-1157, ib),
        GroupElement(-886, ib),
        GroupElement(-636, ib),
        GroupElement(-401, ib),
        GroupElement(-238, ib),
        GroupElement(-79, ib)};
#elif defined(SIGMOID_12_12) // || defined(SIGMOID_TANH_37)
    std::string function_id = "LLAMA_SIGMOID_LLAMA_12_12";
    std::string function_name = "SIGMOID";
    std::string lut_src = "LLAMA";
    int ib = 64, ob = 64, sin = 12, scoef = 20, sout = 12, degree = 2, numPoly = 20;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;
    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-28859, cb), GroupElement(1111248896, cb), GroupElement(8796093022208, cb)},
        {GroupElement(-52266, cb), GroupElement(1281687552, cb), GroupElement(8485815189504, cb)},
        {GroupElement(-37164, cb), GroupElement(1061756928, cb), GroupElement(9286558154752, cb)},
        {GroupElement(-19267, cb), GroupElement(670793728, cb), GroupElement(11421727326208, cb)},
        {GroupElement(-8685, cb), GroupElement(362573824, cb), GroupElement(13666132951040, cb)},
        {GroupElement(-3723, cb), GroupElement(181891072, cb), GroupElement(15310753103872, cb)},
        {GroupElement(-1552, cb), GroupElement(87056384, cb), GroupElement(16346595196928, cb)},
        {GroupElement(-643, cb), GroupElement(40681472, cb), GroupElement(16937522298880, cb)},
        {GroupElement(-271, cb), GroupElement(19050496, cb), GroupElement(17252564860928, cb)},

        // marks end of x = 32768
        // add dummy polynomial for the interval 32768 to -32768 (because we are operating in 64 bitlen for breakp bitlen)
        // if we change break points to 16 bitlen need to remove/change this
        // currently breakpoints are represented in 64 bitlen as well (although dcf comparison happens in 16 bits)

        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb)},

        // marks start of x = -32768

        {GroupElement(270, cb), GroupElement(19050496, cb), GroupElement(339604406272, cb)},
        {GroupElement(642, cb), GroupElement(40681472, cb), GroupElement(654646968320, cb)},
        {GroupElement(1551, cb), GroupElement(87056384, cb), GroupElement(1245574070272, cb)},
        {GroupElement(3722, cb), GroupElement(181891072, cb), GroupElement(2281416163328, cb)},
        {GroupElement(8684, cb), GroupElement(362573824, cb), GroupElement(3926036316160, cb)},
        {GroupElement(19266, cb), GroupElement(670793728, cb), GroupElement(6170441940992, cb)},
        {GroupElement(37163, cb), GroupElement(1061756928, cb), GroupElement(8305611112448, cb)},
        {GroupElement(52265, cb), GroupElement(1281687552, cb), GroupElement(9106354077696, cb)},
        {GroupElement(28858, cb), GroupElement(1111248896, cb), GroupElement(8796093022208, cb)},
    };

    std::vector<GroupElement> fxd_p{GroupElement(0, ib), GroupElement(3640, ib), GroupElement(7281, ib), GroupElement(10922, ib), GroupElement(14563, ib), GroupElement(18204, ib), GroupElement(21845, ib), GroupElement(25486, ib), GroupElement(29127, ib), GroupElement(32767, ib),
                                    GroupElement(-32768, ib), GroupElement(-29128, ib), GroupElement(-25487, ib), GroupElement(-21846, ib), GroupElement(-18205, ib), GroupElement(-14564, ib), GroupElement(-10923, ib), GroupElement(-7282, ib), GroupElement(-3641, ib)};

#elif defined(SIGMOID_9_14)
    std::string function_id = "LLAMA_SIGMOID_LLAMA_9_14";
    std::string function_name = "SIGMOID";
    //     assert((scaleIn == 9) && (scaleOut == 14));
    std::string lut_src = "LLAMA";

    int ib = 64, ob = 64, sin = 9, scoef = 20, sout = 14, degree = 2, numPoly = 34;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-28859, cb), GroupElement(1111248896, cb), GroupElement(8796093022208, cb)},
        {GroupElement(-52266, cb), GroupElement(1281687552, cb), GroupElement(8485815189504, cb)},
        {GroupElement(-37164, cb), GroupElement(1061756928, cb), GroupElement(9286558154752, cb)},
        {GroupElement(-19267, cb), GroupElement(670793728, cb), GroupElement(11421727326208, cb)},
        {GroupElement(-8685, cb), GroupElement(362573824, cb), GroupElement(13666132951040, cb)},
        {GroupElement(-3723, cb), GroupElement(181891072, cb), GroupElement(15310753103872, cb)},
        {GroupElement(-1552, cb), GroupElement(87056384, cb), GroupElement(16346595196928, cb)},
        {GroupElement(-643, cb), GroupElement(40681472, cb), GroupElement(16937522298880, cb)},
        {GroupElement(-271, cb), GroupElement(19050496, cb), GroupElement(17252564860928, cb)},

        // marks end of x = 32768
        // add dummy polynomial for the interval 32768 to -32768 (because we are operating in 64 bitlen for breakp bitlen)
        // if we change break points to 16 bitlen need to remove/change this
        // currently breakpoints are represented in 64 bitlen as well (although dcf comparison happens in 16 bits)

        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb)},

        // marks start of x = -32768

        {GroupElement(270, cb), GroupElement(19050496, cb), GroupElement(339604406272, cb)},
        {GroupElement(642, cb), GroupElement(40681472, cb), GroupElement(654646968320, cb)},
        {GroupElement(1551, cb), GroupElement(87056384, cb), GroupElement(1245574070272, cb)},
        {GroupElement(3722, cb), GroupElement(181891072, cb), GroupElement(2281416163328, cb)},
        {GroupElement(8684, cb), GroupElement(362573824, cb), GroupElement(3926036316160, cb)},
        {GroupElement(19266, cb), GroupElement(670793728, cb), GroupElement(6170441940992, cb)},
        {GroupElement(37163, cb), GroupElement(1061756928, cb), GroupElement(8305611112448, cb)},
        {GroupElement(52265, cb), GroupElement(1281687552, cb), GroupElement(9106354077696, cb)},
        {GroupElement(28858, cb), GroupElement(1111248896, cb), GroupElement(8796093022208, cb)},
    };

    std::vector<GroupElement> fxd_p{GroupElement(0, ib), GroupElement(3640, ib), GroupElement(7281, ib), GroupElement(10922, ib), GroupElement(14563, ib), GroupElement(18204, ib), GroupElement(21845, ib), GroupElement(25486, ib), GroupElement(29127, ib), GroupElement(32767, ib),
                                    GroupElement(-32768, ib), GroupElement(-29128, ib), GroupElement(-25487, ib), GroupElement(-21846, ib), GroupElement(-18205, ib), GroupElement(-14564, ib), GroupElement(-10923, ib), GroupElement(-7282, ib), GroupElement(-3641, ib)};
#elif defined(SIGMOID_8_14)
    std::string function_id = "LLAMA_SIGMOID_LLAMA_8_14";
    std::string function_name = "SIGMOID";
    //     assert((scaleIn == 8) && (scaleOut == 14));
    std::string lut_src = "LLAMA";

    int ib = 64, ob = 64, sin = 8, scoef = 20, sout = 14, degree = 2, numPoly = 34;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-19869, cb), GroupElement(68175872, cb), GroupElement(34359738368, cb)},
        {GroupElement(-46756, cb), GroupElement(76527616, cb), GroupElement(33711128576, cb)},
        {GroupElement(-50110, cb), GroupElement(78611456, cb), GroupElement(33387511808, cb)},
        {GroupElement(-39418, cb), GroupElement(68647680, cb), GroupElement(35708731392, cb)},
        {GroupElement(-26312, cb), GroupElement(52363008, cb), GroupElement(40767193088, cb)},
        {GroupElement(-16037, cb), GroupElement(36404992, cb), GroupElement(46963359744, cb)},
        {GroupElement(-9295, cb), GroupElement(23839744, cb), GroupElement(52818018304, cb)},
        {GroupElement(-5240, cb), GroupElement(15022848, cb), GroupElement(57610797056, cb)},
        {GroupElement(-2910, cb), GroupElement(9231616, cb), GroupElement(61208592384, cb)},
        {GroupElement(-1602, cb), GroupElement(5576960, cb), GroupElement(63762923520, cb)},
        {GroupElement(-879, cb), GroupElement(3328512, cb), GroupElement(65508999168, cb)},
        {GroupElement(-481, cb), GroupElement(1968640, cb), GroupElement(66670493696, cb)},
        {GroupElement(-263, cb), GroupElement(1156096, cb), GroupElement(67427762176, cb)},
        {GroupElement(-144, cb), GroupElement(675584, cb), GroupElement(67912859648, cb)},
        {GroupElement(-78, cb), GroupElement(391424, cb), GroupElement(68221730816, cb)},
        {GroupElement(-44, cb), GroupElement(230912, cb), GroupElement(68408836096, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), flt2fxd(1, degree *sin + scoef, cb)},
        // start of x=N/2
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb)},
        {GroupElement(43, cb), GroupElement(230912, cb), GroupElement(310575104, cb)},
        {GroupElement(77, cb), GroupElement(391424, cb), GroupElement(497680384, cb)},
        {GroupElement(143, cb), GroupElement(675584, cb), GroupElement(806551552, cb)},
        {GroupElement(262, cb), GroupElement(1156096, cb), GroupElement(1291649024, cb)},
        {GroupElement(480, cb), GroupElement(1968640, cb), GroupElement(2048917504, cb)},
        {GroupElement(878, cb), GroupElement(3328512, cb), GroupElement(3210412032, cb)},
        {GroupElement(1601, cb), GroupElement(5576960, cb), GroupElement(4956487680, cb)},
        {GroupElement(2909, cb), GroupElement(9231616, cb), GroupElement(7510818816, cb)},
        {GroupElement(5239, cb), GroupElement(15022848, cb), GroupElement(11108614144, cb)},
        {GroupElement(9294, cb), GroupElement(23839744, cb), GroupElement(15901392896, cb)},
        {GroupElement(16036, cb), GroupElement(36404992, cb), GroupElement(21756051456, cb)},
        {GroupElement(26311, cb), GroupElement(52363008, cb), GroupElement(27952218112, cb)},
        {GroupElement(39417, cb), GroupElement(68647680, cb), GroupElement(33010679808, cb)},
        {GroupElement(50109, cb), GroupElement(78611456, cb), GroupElement(35331899392, cb)},
        {GroupElement(46755, cb), GroupElement(76527616, cb), GroupElement(35008282624, cb)},
        {GroupElement(19868, cb), GroupElement(68175872, cb), GroupElement(34359738368, cb)},
    };

    std::vector<GroupElement> fxd_p{GroupElement(0, ib),
                                    GroupElement(155, ib),
                                    GroupElement(310, ib),
                                    GroupElement(465, ib),
                                    GroupElement(621, ib),
                                    GroupElement(776, ib),
                                    GroupElement(931, ib),
                                    GroupElement(1087, ib),
                                    GroupElement(1242, ib),
                                    GroupElement(1397, ib),
                                    GroupElement(1553, ib),
                                    GroupElement(1708, ib),
                                    GroupElement(1863, ib),
                                    GroupElement(2019, ib),
                                    GroupElement(2174, ib),
                                    GroupElement(2329, ib),
                                    GroupElement(2485, ib),
                                    // x=N/2
                                    GroupElement(32767, ib),
                                    GroupElement(-2485, ib),
                                    GroupElement(-2330, ib),
                                    GroupElement(-2175, ib),
                                    GroupElement(-2020, ib),
                                    GroupElement(-1864, ib),
                                    GroupElement(-1709, ib),
                                    GroupElement(-1554, ib),
                                    GroupElement(-1398, ib),
                                    GroupElement(-1243, ib),
                                    GroupElement(-1088, ib),
                                    GroupElement(-932, ib),
                                    GroupElement(-777, ib),
                                    GroupElement(-622, ib),
                                    GroupElement(-466, ib),
                                    GroupElement(-311, ib),
                                    GroupElement(-156, ib)};
#elif defined(SIGMOID_11_14)
    std::string function_id = "LLAMA_SIGMOID_LLAMA_11_14";
    std::string function_name = "SIGMOID";
    //     assert((scaleIn == 11) && (scaleOut == 14));
    std::string lut_src = "LLAMA";

    int ib = 64, ob = 64, sin = 11, scoef = 20, sout = 14, degree = 2, numPoly = 34;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-19863, cb), GroupElement(545402880, cb), GroupElement(2199019061248, cb)},
        {GroupElement(-46749, cb), GroupElement(612192256, cb), GroupElement(2157541588992, cb)},
        {GroupElement(-50115, cb), GroupElement(628916224, cb), GroupElement(2136767201280, cb)},
        {GroupElement(-39432, cb), GroupElement(549302272, cb), GroupElement(2285102956544, cb)},
        {GroupElement(-26328, cb), GroupElement(419088384, cb), GroupElement(2608588652544, cb)},
        {GroupElement(-16051, cb), GroupElement(291432448, cb), GroupElement(3004996517888, cb)},
        {GroupElement(-9305, cb), GroupElement(190885888, cb), GroupElement(3379673694208, cb)},
        {GroupElement(-5247, cb), GroupElement(120313856, cb), GroupElement(3686482837504, cb)},
        {GroupElement(-2914, cb), GroupElement(73947136, cb), GroupElement(3916846596096, cb)},
        {GroupElement(-1605, cb), GroupElement(44681216, cb), GroupElement(4080437035008, cb)},
        {GroupElement(-880, cb), GroupElement(26671104, cb), GroupElement(4192286539776, cb)},
        {GroupElement(-482, cb), GroupElement(15777792, cb), GroupElement(4266706075648, cb)},
        {GroupElement(-263, cb), GroupElement(9267200, cb), GroupElement(4315229978624, cb)},
        {GroupElement(-144, cb), GroupElement(5416960, cb), GroupElement(4346322354176, cb)},
        {GroupElement(-79, cb), GroupElement(3139584, cb), GroupElement(4366123663360, cb)},
        {GroupElement(-44, cb), GroupElement(1851392, cb), GroupElement(4378115178496, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), flt2fxd(1, degree *sin + scoef, cb)},
        // x=N/2
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb)},
        {GroupElement(43, cb), GroupElement(1851392, cb), GroupElement(19927138304, cb)},
        {GroupElement(78, cb), GroupElement(3139584, cb), GroupElement(31918653440, cb)},
        {GroupElement(143, cb), GroupElement(5416960, cb), GroupElement(51719962624, cb)},
        {GroupElement(262, cb), GroupElement(9267200, cb), GroupElement(82812338176, cb)},
        {GroupElement(481, cb), GroupElement(15777792, cb), GroupElement(131336241152, cb)},
        {GroupElement(879, cb), GroupElement(26671104, cb), GroupElement(205755777024, cb)},
        {GroupElement(1604, cb), GroupElement(44681216, cb), GroupElement(317605281792, cb)},
        {GroupElement(2913, cb), GroupElement(73947136, cb), GroupElement(481195720704, cb)},
        {GroupElement(5246, cb), GroupElement(120313856, cb), GroupElement(711559479296, cb)},
        {GroupElement(9304, cb), GroupElement(190885888, cb), GroupElement(1018368622592, cb)},
        {GroupElement(16050, cb), GroupElement(291432448, cb), GroupElement(1393045798912, cb)},
        {GroupElement(26327, cb), GroupElement(419088384, cb), GroupElement(1789453664256, cb)},
        {GroupElement(39431, cb), GroupElement(549302272, cb), GroupElement(2112939360256, cb)},
        {GroupElement(50114, cb), GroupElement(628916224, cb), GroupElement(2261275115520, cb)},
        {GroupElement(46748, cb), GroupElement(612192256, cb), GroupElement(2240500727808, cb)},
        {GroupElement(19862, cb), GroupElement(545402880, cb), GroupElement(2199019061248, cb)},
    };

    std::vector<GroupElement> fxd_p{
        GroupElement(0, ib),
        GroupElement(1242, ib),
        GroupElement(2484, ib),
        GroupElement(3726, ib),
        GroupElement(4968, ib),
        GroupElement(6210, ib),
        GroupElement(7452, ib),
        GroupElement(8694, ib),
        GroupElement(9937, ib),
        GroupElement(11179, ib),
        GroupElement(12421, ib),
        GroupElement(13663, ib),
        GroupElement(14905, ib),
        GroupElement(16147, ib),
        GroupElement(17389, ib),
        GroupElement(18631, ib),
        GroupElement(19874, ib),
        // x=N/2
        GroupElement(32767, ib),
        GroupElement(-19874, ib),
        GroupElement(-18632, ib),
        GroupElement(-17390, ib),
        GroupElement(-16148, ib),
        GroupElement(-14906, ib),
        GroupElement(-13664, ib),
        GroupElement(-12422, ib),
        GroupElement(-11180, ib),
        GroupElement(-9937, ib),
        GroupElement(-8695, ib),
        GroupElement(-7453, ib),
        GroupElement(-6211, ib),
        GroupElement(-4969, ib),
        GroupElement(-3727, ib),
        GroupElement(-2485, ib),
        GroupElement(-1243, ib)};
#elif defined(SIGMOID_13_14)
    std::string function_id = "LLAMA_SIGMOID_LLAMA_13_14";
    std::string function_name = "SIGMOID";
    //     assert((scaleIn == 13) && (scaleOut == 14));
    std::string lut_src = "LLAMA";

    int ib = 64, ob = 64, sin = 9, scoef = 9, sout = 9, degree = 3, numPoly = 85;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(-18720, cb), GroupElement(2177662976, cb), GroupElement(35184304979968, cb)},
        {GroupElement(-45253, cb), GroupElement(2426077184, cb), GroupElement(34602940891136, cb)},
        {GroupElement(-50752, cb), GroupElement(2529050624, cb), GroupElement(34120897921024, cb)},
        {GroupElement(-42055, cb), GroupElement(2284789760, cb), GroupElement(35835999158272, cb)},
        {GroupElement(-29603, cb), GroupElement(1818443776, cb), GroupElement(40202034741248, cb)},
        {GroupElement(-18884, cb), GroupElement(1316691968, cb), GroupElement(46074060341248, cb)},
        {GroupElement(-11677, cb), GroupElement(911826944, cb), GroupElement(51759657517056, cb)},
        // dummy poly b/w 2^16-1 and -2^16, problem arising b/c we have ib=64 and actual bitlen=16}
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb)},
        {GroupElement(11676, cb), GroupElement(911826944, cb), GroupElement(18609019551744, cb)},
        {GroupElement(18883, cb), GroupElement(1316691968, cb), GroupElement(24294616727552, cb)},
        {GroupElement(29602, cb), GroupElement(1818443776, cb), GroupElement(30166642327552, cb)},
        {GroupElement(42054, cb), GroupElement(2284789760, cb), GroupElement(34532677910528, cb)},
        {GroupElement(50751, cb), GroupElement(2529050624, cb), GroupElement(36247779147776, cb)},
        {GroupElement(45252, cb), GroupElement(2426077184, cb), GroupElement(35765736177664, cb)},
        {GroupElement(18719, cb), GroupElement(2177662976, cb), GroupElement(35184304979968, cb)}};

    std::vector<GroupElement> fxd_p{
        GroupElement(0, ib),
        GroupElement(4681, ib),
        GroupElement(9362, ib),
        GroupElement(14043, ib),
        GroupElement(18724, ib),
        GroupElement(23405, ib),
        GroupElement(28086, ib),
        GroupElement(32767, ib),
        // blank area that is due to ib=64 and actual i/p bitlen 16
        GroupElement(-32768, ib),
        GroupElement(-28087, ib),
        GroupElement(-23406, ib),
        GroupElement(-18725, ib),
        GroupElement(-14044, ib),
        GroupElement(-9363, ib),
        GroupElement(-4682, ib)};
#elif defined(SIGMOID_GROTTO_9_9)
    std::string function_id = "LLAMA_SIGMOID_GROTTO_9_9";
    std::string function_name = "SIGMOID";
    //     assert((scaleIn == 9) && (scaleOut == 9));
    std::string lut_src = "GROTTO";

    int ib = 64, ob = 64, sin = 9, scoef = 9, sout = 9, degree = 3, numPoly = 29;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials{
        {GroupElement(65526, cb), GroupElement(0, cb), GroupElement(33554432, cb), GroupElement(34225520640, cb)},
        {GroupElement(65527, cb), GroupElement(0, cb), GroupElement(33554432, cb), GroupElement(34225520640, cb)},
        {GroupElement(65528, cb), GroupElement(33553408, cb), GroupElement(33554432, cb), GroupElement(34225520640, cb)},
        {GroupElement(65530, cb), GroupElement(33551872, cb), GroupElement(34078720, cb), GroupElement(34225520640, cb)},
        {GroupElement(65532, cb), GroupElement(33549824, cb), GroupElement(34865152, cb), GroupElement(34091302912, cb)},
        {GroupElement(65533, cb), GroupElement(33547264, cb), GroupElement(36175872, cb), GroupElement(33957085184, cb)},
        {GroupElement(65535, cb), GroupElement(33544704, cb), GroupElement(37486592, cb), GroupElement(33688649728, cb)},
        {GroupElement(0, cb), GroupElement(33541632, cb), GroupElement(39321600, cb), GroupElement(33285996544, cb)},
        {GroupElement(1, cb), GroupElement(33539072, cb), GroupElement(41418752, cb), GroupElement(32749125632, cb)},
        {GroupElement(2, cb), GroupElement(33536512, cb), GroupElement(43515904, cb), GroupElement(32212254720, cb)},
        {GroupElement(3, cb), GroupElement(33534464, cb), GroupElement(45350912, cb), GroupElement(31675383808, cb)},
        {GroupElement(3, cb), GroupElement(33533440, cb), GroupElement(46661632, cb), GroupElement(31272730624, cb)},
        {GroupElement(3, cb), GroupElement(33533440, cb), GroupElement(46137344, cb), GroupElement(31406948352, cb)},
        {GroupElement(3, cb), GroupElement(33534976, cb), GroupElement(44302336, cb), GroupElement(32212254720, cb)},
        {GroupElement(2, cb), GroupElement(33537024, cb), GroupElement(41418752, cb), GroupElement(33688649728, cb)},
        {GroupElement(2, cb), GroupElement(33539072, cb), GroupElement(38010880, cb), GroupElement(35567697920, cb)},
        {GroupElement(1, cb), GroupElement(33541120, cb), GroupElement(34340864, cb), GroupElement(37715181568, cb)},
        {GroupElement(1, cb), GroupElement(33543168, cb), GroupElement(30408704, cb), GroupElement(40265318400, cb)},
        {GroupElement(1, cb), GroupElement(33545216, cb), GroupElement(26214400, cb), GroupElement(43083890688, cb)},
        {GroupElement(0, cb), GroupElement(33547264, cb), GroupElement(22282240, cb), GroupElement(46036680704, cb)},
        {GroupElement(0, cb), GroupElement(33548800, cb), GroupElement(18350080, cb), GroupElement(48989470720, cb)},
        {GroupElement(0, cb), GroupElement(33549824, cb), GroupElement(14942208, cb), GroupElement(51942260736, cb)},
        {GroupElement(0, cb), GroupElement(33551360, cb), GroupElement(11534336, cb), GroupElement(54895050752, cb)},
        {GroupElement(0, cb), GroupElement(33552384, cb), GroupElement(8912896, cb), GroupElement(57579405312, cb)},
        {GroupElement(0, cb), GroupElement(33552896, cb), GroupElement(6553600, cb), GroupElement(60129542144, cb)},
        {GroupElement(0, cb), GroupElement(33553408, cb), GroupElement(4456448, cb), GroupElement(62277025792, cb)},
        {GroupElement(0, cb), GroupElement(33553920, cb), GroupElement(2883584, cb), GroupElement(64156073984, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(1572864, cb), GroupElement(65766686720, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(786432, cb), GroupElement(66974646272, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(262144, cb), GroupElement(67779952640, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(68316823552, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(68585259008, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(134217728, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(268435456, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(402653184, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(262144, cb), GroupElement(536870912, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(262144, cb), GroupElement(671088640, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(524288, cb), GroupElement(939524096, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(524288, cb), GroupElement(1207959552, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(786432, cb), GroupElement(1476395008, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(1048576, cb), GroupElement(1879048192, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(1310720, cb), GroupElement(2281701376, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(1572864, cb), GroupElement(2818572288, cb)},
        {GroupElement(0, cb), GroupElement(0, cb), GroupElement(2097152, cb), GroupElement(3355443200, cb)},
        {GroupElement(0, cb), GroupElement(512, cb), GroupElement(2621440, cb), GroupElement(3892314112, cb)},
        {GroupElement(0, cb), GroupElement(512, cb), GroupElement(3145728, cb), GroupElement(4697620480, cb)},
        {GroupElement(0, cb), GroupElement(512, cb), GroupElement(3670016, cb), GroupElement(5368709120, cb)},
        {GroupElement(0, cb), GroupElement(1024, cb), GroupElement(4456448, cb), GroupElement(6308233216, cb)},
        {GroupElement(0, cb), GroupElement(1024, cb), GroupElement(5242880, cb), GroupElement(7247757312, cb)},
        {GroupElement(0, cb), GroupElement(1536, cb), GroupElement(6291456, cb), GroupElement(8321499136, cb)},
        {GroupElement(0, cb), GroupElement(1536, cb), GroupElement(7340032, cb), GroupElement(9529458688, cb)},
        {GroupElement(0, cb), GroupElement(2048, cb), GroupElement(8650752, cb), GroupElement(10871635968, cb)},
        {GroupElement(0, cb), GroupElement(2560, cb), GroupElement(9961472, cb), GroupElement(12213813248, cb)},
        {GroupElement(0, cb), GroupElement(3072, cb), GroupElement(11534336, cb), GroupElement(13555990528, cb)},
        {GroupElement(0, cb), GroupElement(3584, cb), GroupElement(13107200, cb), GroupElement(15032385536, cb)},
        {GroupElement(0, cb), GroupElement(4608, cb), GroupElement(14942208, cb), GroupElement(16642998272, cb)},
        {GroupElement(0, cb), GroupElement(5120, cb), GroupElement(16777216, cb), GroupElement(18253611008, cb)},
        {GroupElement(0, cb), GroupElement(6144, cb), GroupElement(18874368, cb), GroupElement(19998441472, cb)},
        {GroupElement(0, cb), GroupElement(7168, cb), GroupElement(21233664, cb), GroupElement(21877489664, cb)},
        {GroupElement(1, cb), GroupElement(8192, cb), GroupElement(23592960, cb), GroupElement(23622320128, cb)},
        {GroupElement(1, cb), GroupElement(9216, cb), GroupElement(25952256, cb), GroupElement(25367150592, cb)},
        {GroupElement(1, cb), GroupElement(10240, cb), GroupElement(28573696, cb), GroupElement(27111981056, cb)},
        {GroupElement(1, cb), GroupElement(11776, cb), GroupElement(31195136, cb), GroupElement(28856811520, cb)},
        {GroupElement(1, cb), GroupElement(13312, cb), GroupElement(33816576, cb), GroupElement(30601641984, cb)},
        {GroupElement(2, cb), GroupElement(14336, cb), GroupElement(36438016, cb), GroupElement(32078036992, cb)},
        {GroupElement(2, cb), GroupElement(15872, cb), GroupElement(38797312, cb), GroupElement(33554432000, cb)},
        {GroupElement(2, cb), GroupElement(17408, cb), GroupElement(41156608, cb), GroupElement(34762391552, cb)},
        {GroupElement(2, cb), GroupElement(18944, cb), GroupElement(43253760, cb), GroupElement(35836133376, cb)},
        {GroupElement(3, cb), GroupElement(19968, cb), GroupElement(45088768, cb), GroupElement(36641439744, cb)},
        {GroupElement(3, cb), GroupElement(20992, cb), GroupElement(46137344, cb), GroupElement(37178310656, cb)},
        {GroupElement(3, cb), GroupElement(20992, cb), GroupElement(46661632, cb), GroupElement(37312528384, cb)},
        {GroupElement(3, cb), GroupElement(20480, cb), GroupElement(45875200, cb), GroupElement(37044092928, cb)},
        {GroupElement(2, cb), GroupElement(18944, cb), GroupElement(44564480, cb), GroupElement(36641439744, cb)},
        {GroupElement(2, cb), GroupElement(16896, cb), GroupElement(42729472, cb), GroupElement(36104568832, cb)},
        {GroupElement(1, cb), GroupElement(14848, cb), GroupElement(41156608, cb), GroupElement(35701915648, cb)},
        {GroupElement(0, cb), GroupElement(12288, cb), GroupElement(39321600, cb), GroupElement(35299262464, cb)},
        {GroupElement(65535, cb), GroupElement(10240, cb), GroupElement(37748736, cb), GroupElement(34896609280, cb)},
        {GroupElement(65534, cb), GroupElement(7680, cb), GroupElement(36438016, cb), GroupElement(34762391552, cb)},
        {GroupElement(65532, cb), GroupElement(5632, cb), GroupElement(35389440, cb), GroupElement(34493956096, cb)},
        {GroupElement(65531, cb), GroupElement(3584, cb), GroupElement(34340864, cb), GroupElement(34359738368, cb)},
        {GroupElement(65529, cb), GroupElement(2048, cb), GroupElement(33816576, cb), GroupElement(34359738368, cb)},
        {GroupElement(65528, cb), GroupElement(512, cb), GroupElement(33554432, cb), GroupElement(34359738368, cb)},
        {GroupElement(65527, cb), GroupElement(0, cb), GroupElement(33554432, cb), GroupElement(34359738368, cb)},
        {GroupElement(65526, cb), GroupElement(0, cb), GroupElement(33554432, cb), GroupElement(34225520640, cb)},
    };

    std::vector<GroupElement> fxd_p{
        GroupElement(0, ib),
        GroupElement(125, ib),
        GroupElement(218, ib),
        GroupElement(304, ib),
        GroupElement(387, ib),
        GroupElement(469, ib),
        GroupElement(553, ib),
        GroupElement(638, ib),
        GroupElement(726, ib),
        GroupElement(820, ib),
        GroupElement(923, ib),
        GroupElement(1042, ib),
        GroupElement(1216, ib),
        GroupElement(1369, ib),
        GroupElement(1505, ib),
        GroupElement(1638, ib),
        GroupElement(1772, ib),
        GroupElement(1909, ib),
        GroupElement(2051, ib),
        GroupElement(2199, ib),
        GroupElement(2356, ib),
        GroupElement(2524, ib),
        GroupElement(2704, ib),
        GroupElement(2899, ib),
        GroupElement(3114, ib),
        GroupElement(3352, ib),
        GroupElement(3619, ib),
        GroupElement(3925, ib),
        GroupElement(4283, ib),
        GroupElement(4712, ib),
        GroupElement(5251, ib),
        GroupElement(5972, ib),
        GroupElement(-32768, ib),
        GroupElement(-6344, ib),
        GroupElement(-5692, ib),
        GroupElement(-5502, ib),
        GroupElement(-5313, ib),
        GroupElement(-5127, ib),
        GroupElement(-4954, ib),
        GroupElement(-4783, ib),
        GroupElement(-4625, ib),
        GroupElement(-4477, ib),
        GroupElement(-4327, ib),
        GroupElement(-4187, ib),
        GroupElement(-4045, ib),
        GroupElement(-3913, ib),
        GroupElement(-3787, ib),
        GroupElement(-3659, ib),
        GroupElement(-3538, ib),
        GroupElement(-3414, ib),
        GroupElement(-3297, ib),
        GroupElement(-3185, ib),
        GroupElement(-3071, ib),
        GroupElement(-2962, ib),
        GroupElement(-2858, ib),
        GroupElement(-2759, ib),
        GroupElement(-2656, ib),
        GroupElement(-2557, ib),
        GroupElement(-2462, ib),
        GroupElement(-2363, ib),
        GroupElement(-2268, ib),
        GroupElement(-2176, ib),
        GroupElement(-2086, ib),
        GroupElement(-1992, ib),
        GroupElement(-1901, ib),
        GroupElement(-1812, ib),
        GroupElement(-1724, ib),
        GroupElement(-1629, ib),
        GroupElement(-1535, ib),
        GroupElement(-1439, ib),
        GroupElement(-1340, ib),
        GroupElement(-1219, ib),
        GroupElement(-1070, ib),
        GroupElement(-970, ib),
        GroupElement(-878, ib),
        GroupElement(-794, ib),
        GroupElement(-716, ib),
        GroupElement(-641, ib),
        GroupElement(-569, ib),
        GroupElement(-499, ib),
        GroupElement(-423, ib),
        GroupElement(-347, ib),
        GroupElement(-270, ib),
        GroupElement(-189, ib),
        GroupElement(-101, ib)};
//     scaleCoef = 9;

#elif defined(INVSQRT_10_9)
    std::string function_id = "LLAMA_INVSQRT_LLAMA_10_9";
    std::string function_name = "INVSQRT";
    //     assert((scaleIn == 9) && (scaleOut == 9));
    std::string lut_src = "LLAMA";

    int ib = 64, ob = 64, sin = 10, scoef = 13, sout = 9, degree = 2, numPoly = 10;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        // for x=0 to fxd2flt(epsilon=0.1), technically input shouldn't fall here because we have condition x >= epsilon,
        // but worst case, just to be safe, use same poly as first interval of octave spline
        { GroupElement(116573, cb),  GroupElement(-99579904, cb),    GroupElement(35036069888, cb)},
        // octave spline from epsilon to end (2^16)
        { GroupElement(116573, cb),  GroupElement(-99579904, cb),    GroupElement(35036069888, cb)},
        { GroupElement( 15140, cb),  GroupElement(-27114496, cb),    GroupElement(22093496320, cb)},
        { GroupElement(  5288, cb),  GroupElement(-15048704, cb),    GroupElement(18398314496, cb)},
        { GroupElement(   957, cb),  GroupElement( -5320704, cb),    GroupElement(12937330688, cb)},
        { GroupElement(   190, cb),  GroupElement( -2032640, cb),    GroupElement( 9413066752, cb)},
        { GroupElement(    34, cb),  GroupElement(  -731136, cb),    GroupElement( 6689914880, cb)},
        { GroupElement(     6, cb),  GroupElement(  -260096, cb),    GroupElement( 4739563520, cb)},
        { GroupElement(     1, cb),  GroupElement(   -97280, cb),    GroupElement( 3407872000, cb)},
        // for negative x (input doesn't fall here, so use some dummy)
        { GroupElement(0, cb),  GroupElement(0, cb),    GroupElement(0, cb)}
    };

    std::vector<GroupElement> fxd_p{/* dummy knot x=0 for consistency */  GroupElement(0, ib),   /* actual spline starts here */  GroupElement(102, ib),     GroupElement(357, ib),     GroupElement(612, ib),    GroupElement(1122, ib),    GroupElement(2143, ib),    GroupElement(4185, ib),    GroupElement(8268, ib),   GroupElement(16435, ib),   GroupElement(32767, ib)  /* actual spline ends here  */
    };

#elif defined(INVSQRT_12_11)
    std::string function_id = "LLAMA_INVSQRT_LLAMA_12_11";
    std::string function_name = "INVSQRT";
    //     assert((scaleIn == 9) && (scaleOut == 9));
    std::string lut_src = "LLAMA";

    int ib = 64, ob = 64, sin = 12, scoef = 13, sout = 11, degree = 2, numPoly = 10;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        // for x=0 to fxd2flt(epsilon=0.1), technically input shouldn't fall here because we have condition x >= epsilon,
        // but worst case, just to be safe, use same poly as first interval of octave spline
        { GroupElement( 454375, cb),   GroupElement(-850591744, cb),   GroupElement(705582596096, cb)},
        // octave spline from epsilon to end (2^16)
        { GroupElement( 454375, cb),   GroupElement(-850591744, cb),   GroupElement(705582596096, cb)},
        { GroupElement( 191957, cb),   GroupElement(-503250944, cb),   GroupElement(590641889280, cb)},
        { GroupElement(  75342, cb),   GroupElement(-289939456, cb),   GroupElement(493099155456, cb)},
        { GroupElement(  21226, cb),   GroupElement(-136224768, cb),   GroupElement(383946588160, cb)},
        { GroupElement(   4919, cb),   GroupElement( -56926208, cb),   GroupElement(287527927808, cb)},
        { GroupElement(   1016, cb),   GroupElement( -22155264, cb),   GroupElement(210101075968, cb)},
        { GroupElement(    189, cb),   GroupElement(  -8101888, cb),   GroupElement(150374187008, cb)},
        { GroupElement(     38, cb),   GroupElement(  -3104768, cb),   GroupElement(108917686272, cb)},
        // for negative x (input doesn't fall here, so use some dummy)
        { GroupElement(0, cb),  GroupElement(0, cb),    GroupElement(0, cb)}
    };

    std::vector<GroupElement> fxd_p{/* dummy knot x=0 for consistency */  GroupElement(0, ib),   /* actual spline starts here */   GroupElement(409, ib),     GroupElement(661, ib),     GroupElement(914, ib),    GroupElement(1420, ib),    GroupElement(2431, ib),    GroupElement(4453, ib),    GroupElement(8498, ib),   GroupElement(16588, ib),   GroupElement(32768, ib) /* actual spline ends here  */
    };
#elif defined(INVSQRT_GROTTO_9_9)
    std::string function_id = "LLAMA_INVSQRT_GROTTO_9_9";
    std::string function_name = "INVSQRT";
    //     assert((scaleIn == 9) && (scaleOut == 9));
    std::string lut_src = "GROTTO";

    int ib = 64, ob = 64, sin = 9, scoef = 9, sout = 9, degree = 3, numPoly = 40;
    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials 
    {
        { GroupElement(59372, cb), GroupElement(26828800, cb), GroupElement(9170845696, cb), GroupElement(3292360867840, cb) },
        { GroupElement(43958, cb), GroupElement(30216704, cb), GroupElement(14810349568, cb), GroupElement(2336327991296, cb) },
        { GroupElement(28693, cb), GroupElement(13525504, cb), GroupElement(1156841472, cb), GroupElement(1943741136896, cb) },
        { GroupElement(15571, cb), GroupElement(15899136, cb), GroupElement(12407799808, cb), GroupElement(1690203848704, cb) },
        { GroupElement(23791, cb), GroupElement(3635200, cb), GroupElement(14416347136, cb), GroupElement(1562025918464, cb) },
        { GroupElement(1781, cb), GroupElement(1638400, cb), GroupElement(16973561856, cb), GroupElement(1336808570880, cb) },
        { GroupElement(48090, cb), GroupElement(2639872, cb), GroupElement(9288286208, cb), GroupElement(1234534662144, cb) },
        { GroupElement(6606, cb), GroupElement(16313344, cb), GroupElement(10689970176, cb), GroupElement(1136421502976, cb) },
        { GroupElement(49892, cb), GroupElement(14036480, cb), GroupElement(5082447872, cb), GroupElement(1043005964288, cb) },
        { GroupElement(2698, cb), GroupElement(16413184, cb), GroupElement(10809245696, cb), GroupElement(954422263808, cb) },
        { GroupElement(15258, cb), GroupElement(4158464, cb), GroupElement(11466440704, cb), GroupElement(872012578816, cb) },
        { GroupElement(20367, cb), GroupElement(6629888, cb), GroupElement(8200650752, cb), GroupElement(794703167488, cb) },
        { GroupElement(35108, cb), GroupElement(6558720, cb), GroupElement(1801453568, cb), GroupElement(722359812096, cb) },
        { GroupElement(29465, cb), GroupElement(3270656, cb), GroupElement(10049028096, cb), GroupElement(655385165824, cb) },
        { GroupElement(55229, cb), GroupElement(26493440, cb), GroupElement(16320299008, cb), GroupElement(593376575488, cb) },
        { GroupElement(20216, cb), GroupElement(29394944, cb), GroupElement(3898343424, cb), GroupElement(535797170176, cb) },
        { GroupElement(34444, cb), GroupElement(15900160, cb), GroupElement(7489978368, cb), GroupElement(482378514432, cb) },
        { GroupElement(1119, cb), GroupElement(872448, cb), GroupElement(10167779328, cb), GroupElement(432986390528, cb) },
        { GroupElement(10624, cb), GroupElement(5576704, cb), GroupElement(12145393664, cb), GroupElement(387755016192, cb) },
        { GroupElement(10966, cb), GroupElement(22231552, cb), GroupElement(13593739264, cb), GroupElement(346281738240, cb) },
        { GroupElement(41292, cb), GroupElement(12455424, cb), GroupElement(14646509568, cb), GroupElement(308432338944, cb) },
        { GroupElement(54990, cb), GroupElement(6873600, cb), GroupElement(15406465024, cb), GroupElement(273804165120, cb) },
        { GroupElement(61046, cb), GroupElement(3735552, cb), GroupElement(15949627392, cb), GroupElement(242397216768, cb) },
        { GroupElement(63663, cb), GroupElement(2000384, cb), GroupElement(16334192640, cb), GroupElement(213943058432, cb) },
        { GroupElement(64771, cb), GroupElement(1055744, cb), GroupElement(16603414528, cb), GroupElement(188307472384, cb) },
        { GroupElement(65230, cb), GroupElement(548864, cb), GroupElement(16790585344, cb), GroupElement(165222023168, cb) },
        { GroupElement(65419, cb), GroupElement(276992, cb), GroupElement(16921395200, cb), GroupElement(144149839872, cb) },
        { GroupElement(65494, cb), GroupElement(133120, cb), GroupElement(17013145600, cb), GroupElement(124554051584, cb) },
        { GroupElement(65522, cb), GroupElement(62464, cb), GroupElement(17073700864, cb), GroupElement(107105746944, cb) },
        { GroupElement(65532, cb), GroupElement(28672, cb), GroupElement(17113546752, cb), GroupElement(91670708224, cb) },
        { GroupElement(65535, cb), GroupElement(12800, cb), GroupElement(17138974720, cb), GroupElement(77980499968, cb) },
        { GroupElement(0, cb), GroupElement(5120, cb), GroupElement(17154965504, cb), GroupElement(66035122176, cb) },
        { GroupElement(0, cb), GroupElement(2048, cb), GroupElement(17165189120, cb), GroupElement(55700357120, cb) },
        { GroupElement(0, cb), GroupElement(512, cb), GroupElement(17171218432, cb), GroupElement(46573551616, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(17174888448, cb), GroupElement(38788923392, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(17177247744, cb), GroupElement(31809601536, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(17178558464, cb), GroupElement(25769803776, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(17179344896, cb), GroupElement(20803747840, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(8727507763200, cb) },
        { GroupElement(16796, cb), GroupElement(2476544, cb), GroupElement(11361583104, cb), GroupElement(1537329856512, cb) },
    };

    std::vector<GroupElement> fxd_p 
    {
        GroupElement(0, ib),
        GroupElement(1, ib),
        GroupElement(2, ib),
        GroupElement(3, ib),
        GroupElement(4, ib),
        GroupElement(5, ib),
        GroupElement(6, ib),
        GroupElement(8, ib),
        GroupElement(9, ib),
        GroupElement(11, ib),
        GroupElement(13, ib),
        GroupElement(16, ib),
        GroupElement(20, ib),
        GroupElement(24, ib),
        GroupElement(29, ib),
        GroupElement(36, ib),
        GroupElement(44, ib),
        GroupElement(55, ib),
        GroupElement(68, ib),
        GroupElement(86, ib),
        GroupElement(108, ib),
        GroupElement(136, ib),
        GroupElement(173, ib),
        GroupElement(222, ib),
        GroupElement(286, ib),
        GroupElement(371, ib),
        GroupElement(483, ib),
        GroupElement(643, ib),
        GroupElement(864, ib),
        GroupElement(1173, ib),
        GroupElement(1610, ib),
        GroupElement(2235, ib),
        GroupElement(3133, ib),
        GroupElement(4440, ib),
        GroupElement(6362, ib),
        GroupElement(9315, ib),
        GroupElement(14005, ib),
        GroupElement(21414, ib),
        GroupElement(-32768, ib),
        GroupElement(-15, ib)
    };
// #elif defined(STRUCT_NAME)
//     std::string function_name = "SIGMOID";
//     //     assert((scaleIn == 9) && (scaleOut == 9));
//     std::string lut_src = "LLAMA";

//     int ib = 64, ob = 64, sin = 9, scoef = 9, sout = 9, degree = 3, numPoly = 29;
//     int cb = 64;

//     int input_precision = sin;
//     int input_bitwidth = 16;
//     int output_precision = sout;
//     int output_bitwidth = 16;

//     std::vector<std::vector<GroupElement>> fxd_polynomials{
//     };

//     std::vector<GroupElement> fxd_p{};
#elif defined(LOG10_GROTTO_9_9)
    std::string function_id = "LLAMA_LOG10_GROTTO_9_9";
    std::string function_name = "LOG10";
    int ib = 64, ob = 64, sin = 9, scoef = 9, sout = 9, degree = 3, numPoly = 32;
    std::string lut_src = "GROTTO";

    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;
    std::vector<std::vector<GroupElement>> fxd_polynomials 
    {
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(134217728, cb) },
        { GroupElement(57730, cb), GroupElement(10974208, cb), GroupElement(6453985280, cb), GroupElement(1387811307520, cb) },
        { GroupElement(8842, cb), GroupElement(6444544, cb), GroupElement(12588941312, cb), GroupElement(8558124990464, cb) },
        { GroupElement(6926, cb), GroupElement(3035648, cb), GroupElement(6230114304, cb), GroupElement(8578794520576, cb) },
        { GroupElement(12123, cb), GroupElement(10639872, cb), GroupElement(11648106496, cb), GroupElement(8588995067904, cb) },
        { GroupElement(34116, cb), GroupElement(33403904, cb), GroupElement(3374317568, cb), GroupElement(8599061397504, cb) },
        { GroupElement(18766, cb), GroupElement(32083968, cb), GroupElement(14547943424, cb), GroupElement(8609396162560, cb) },
        { GroupElement(26152, cb), GroupElement(23590912, cb), GroupElement(10220208128, cb), GroupElement(8619999363072, cb) },
        { GroupElement(44413, cb), GroupElement(14178304, cb), GroupElement(7196639232, cb), GroupElement(8630468345856, cb) },
        { GroupElement(25193, cb), GroupElement(24159744, cb), GroupElement(5083234304, cb), GroupElement(8640803110912, cb) },
        { GroupElement(48910, cb), GroupElement(28548608, cb), GroupElement(3601858560, cb), GroupElement(8651137875968, cb) },
        { GroupElement(33196, cb), GroupElement(30635008, cb), GroupElement(2561409024, cb), GroupElement(8661204205568, cb) },
        { GroupElement(17857, cb), GroupElement(14995968, cb), GroupElement(1827143680, cb), GroupElement(8671404752896, cb) },
        { GroupElement(28918, cb), GroupElement(24390144, cb), GroupElement(1284243456, cb), GroupElement(8681873735680, cb) },
        { GroupElement(10066, cb), GroupElement(29020160, cb), GroupElement(903348224, cb), GroupElement(8692342718464, cb) },
        { GroupElement(3527, cb), GroupElement(31301120, cb), GroupElement(636747776, cb), GroupElement(8702811701248, cb) },
        { GroupElement(1247, cb), GroupElement(32427520, cb), GroupElement(450101248, cb), GroupElement(8713146466304, cb) },
        { GroupElement(445, cb), GroupElement(32987136, cb), GroupElement(319291392, cb), GroupElement(8723347013632, cb) },
        { GroupElement(161, cb), GroupElement(33266688, cb), GroupElement(227278848, cb), GroupElement(8733547560960, cb) },
        { GroupElement(57, cb), GroupElement(33409536, cb), GroupElement(161742848, cb), GroupElement(8743748108288, cb) },
        { GroupElement(19, cb), GroupElement(33483264, cb), GroupElement(112984064, cb), GroupElement(8754351308800, cb) },
        { GroupElement(6, cb), GroupElement(33519616, cb), GroupElement(79691776, cb), GroupElement(8764820291584, cb) },
        { GroupElement(2, cb), GroupElement(33537024, cb), GroupElement(56098816, cb), GroupElement(8775289274368, cb) },
        { GroupElement(0, cb), GroupElement(33545728, cb), GroupElement(39583744, cb), GroupElement(8785489821696, cb) },
        { GroupElement(0, cb), GroupElement(33550336, cb), GroupElement(28049408, cb), GroupElement(8795690369024, cb) },
        { GroupElement(0, cb), GroupElement(33552384, cb), GroupElement(20185088, cb), GroupElement(9529458688, cb) },
        { GroupElement(0, cb), GroupElement(33553408, cb), GroupElement(14155776, cb), GroupElement(19864223744, cb) },
        { GroupElement(0, cb), GroupElement(33553920, cb), GroupElement(9961472, cb), GroupElement(30333206528, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(7077888, cb), GroupElement(40802189312, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(4980736, cb), GroupElement(51271172096, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(3407872, cb), GroupElement(61471719424, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(2359296, cb), GroupElement(71672266752, cb) },
    };

    std::vector<GroupElement> fxd_p 
    {
        GroupElement(0, ib),
        GroupElement(-16, ib),
        GroupElement(0, ib),
        GroupElement(1, ib),
        GroupElement(2, ib),
        GroupElement(3, ib),
        GroupElement(5, ib),
        GroupElement(7, ib),
        GroupElement(10, ib),
        GroupElement(14, ib),
        GroupElement(20, ib),
        GroupElement(29, ib),
        GroupElement(41, ib),
        GroupElement(58, ib),
        GroupElement(82, ib),
        GroupElement(117, ib),
        GroupElement(166, ib),
        GroupElement(235, ib),
        GroupElement(331, ib),
        GroupElement(464, ib),
        GroupElement(661, ib),
        GroupElement(940, ib),
        GroupElement(1334, ib),
        GroupElement(1887, ib),
        GroupElement(2661, ib),
        GroupElement(3739, ib),
        GroupElement(5232, ib),
        GroupElement(7448, ib),
        GroupElement(10587, ib),
        GroupElement(15008, ib),
        GroupElement(21218, ib),
        GroupElement(29892, ib)
    };


#elif defined(SQRT_GROTTO_9_9)
    std::string function_id = "LLAMA_SQRT_GROTTO_9_9";
    std::string function_name = "SQRT";
    int ib = 64, ob = 64, sin = 9, scoef = 9, sout = 9, degree = 3, numPoly = 26;
    std::string lut_src = "GROTTO";

    int cb = 64;

    int input_precision = sin;
    int input_bitwidth = 16;
    int output_precision = sout;
    int output_bitwidth = 16;

    std::vector<std::vector<GroupElement>> fxd_polynomials 
    {
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(134217728, cb) },
        { GroupElement(18688, cb), GroupElement(24580608, cb), GroupElement(11994398720, cb), GroupElement(3297461141504, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb), GroupElement(0, cb) },
        { GroupElement(56534, cb), GroupElement(13641216, cb), GroupElement(2193358848, cb), GroupElement(1207959552, cb) },
        { GroupElement(33757, cb), GroupElement(19710976, cb), GroupElement(1416626176, cb), GroupElement(1879048192, cb) },
        { GroupElement(56692, cb), GroupElement(31485952, cb), GroupElement(958398464, cb), GroupElement(2684354560, cb) },
        { GroupElement(4554, cb), GroupElement(21100032, cb), GroupElement(674496512, cb), GroupElement(3892314112, cb) },
        { GroupElement(27889, cb), GroupElement(28744704, cb), GroupElement(490733568, cb), GroupElement(5368709120, cb) },
        { GroupElement(6567, cb), GroupElement(31537664, cb), GroupElement(367001600, cb), GroupElement(7247757312, cb) },
        { GroupElement(1659, cb), GroupElement(32672256, cb), GroupElement(278659072, cb), GroupElement(9663676416, cb) },
        { GroupElement(466, cb), GroupElement(33142784, cb), GroupElement(216006656, cb), GroupElement(12348030976, cb) },
        { GroupElement(143, cb), GroupElement(33352192, cb), GroupElement(170393600, cb), GroupElement(15703474176, cb) },
        { GroupElement(47, cb), GroupElement(33451008, cb), GroupElement(136314880, cb), GroupElement(19730006016, cb) },
        { GroupElement(16, cb), GroupElement(33499648, cb), GroupElement(110362624, cb), GroupElement(24427626496, cb) },
        { GroupElement(6, cb), GroupElement(33524224, cb), GroupElement(90177536, cb), GroupElement(29796335616, cb) },
        { GroupElement(2, cb), GroupElement(33537536, cb), GroupElement(74448896, cb), GroupElement(36104568832, cb) },
        { GroupElement(0, cb), GroupElement(33544704, cb), GroupElement(61865984, cb), GroupElement(43486543872, cb) },
        { GroupElement(0, cb), GroupElement(33548800, cb), GroupElement(51904512, cb), GroupElement(51942260736, cb) },
        { GroupElement(0, cb), GroupElement(33551360, cb), GroupElement(43778048, cb), GroupElement(61471719424, cb) },
        { GroupElement(0, cb), GroupElement(33552384, cb), GroupElement(36962304, cb), GroupElement(72880226304, cb) },
        { GroupElement(0, cb), GroupElement(33553408, cb), GroupElement(31457280, cb), GroupElement(85496692736, cb) },
        { GroupElement(0, cb), GroupElement(33553920, cb), GroupElement(27000832, cb), GroupElement(99589554176, cb) },
        { GroupElement(0, cb), GroupElement(33553920, cb), GroupElement(23330816, cb), GroupElement(115293028352, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(20185088, cb), GroupElement(132741332992, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(17563648, cb), GroupElement(151934468096, cb) },
        { GroupElement(0, cb), GroupElement(0, cb), GroupElement(15466496, cb), GroupElement(173140869120, cb) },
    };

    std::vector<GroupElement> fxd_p 
    {
        GroupElement(0, ib),
        GroupElement(-15, ib),
        GroupElement(0, ib),
        GroupElement(1, ib),
        GroupElement(2, ib),
        GroupElement(5, ib),
        GroupElement(12, ib),
        GroupElement(24, ib),
        GroupElement(45, ib),
        GroupElement(79, ib),
        GroupElement(134, ib),
        GroupElement(220, ib),
        GroupElement(349, ib),
        GroupElement(537, ib),
        GroupElement(812, ib),
        GroupElement(1204, ib),
        GroupElement(1751, ib),
        GroupElement(2520, ib),
        GroupElement(3551, ib),
        GroupElement(4999, ib),
        GroupElement(6943, ib),
        GroupElement(9493, ib),
        GroupElement(12796, ib),
        GroupElement(17032, ib),
        GroupElement(22448, ib),
        GroupElement(29252, ib)
    };

#else
    throw std::invalid_argument("no scales selected");
#endif
}

#endif // __FUNCTION_CONFIG_HPP__