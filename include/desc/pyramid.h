#ifndef GaussianPyramid_hpp
#define GaussianPyramid_hpp

#define N_OCTAVE 4
#define N_SCALE 4
#define SIGMA_INIT 0.5
#define SIGMA_BASE sqrt(2)
#define SCALE_INTERVAL sqrt(2)

#include "3rdparty/CImg.h"
#include <stdio.h>
#include <vector>

using namespace std;
class SIFT;
struct Gaussian {
    Image image;
    float scale;
    int octave;
};

struct DoG {
    Image image;
    float scale;
    int octave;
};

class GaussianPyramid {
    int octaves_ = N_OCTAVE;
    int s_ = N_SCALE;
    double sigma_ = SIGMA_BASE;
    double scale_interval_ = SCALE_INTERVAL;
    double init_sigma_ = SIGMA_INIT;
    bool dbl = false;
    
    // whether upsample the original image
    bool upsample_ = false;
    
    
    
public:
    vector<vector<DoG> > dogs_;
    vector<vector<Gaussian> > gaussians_;
    friend class SIFT;
    void build(const Image& img);
};

#endif /* GaussianPyramid_hpp */