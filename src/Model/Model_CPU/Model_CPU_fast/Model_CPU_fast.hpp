#ifdef GALAX_MODEL_CPU_FAST

#ifndef MODEL_CPU_FAST_HPP_
#define MODEL_CPU_FAST_HPP_

#include "../Model_CPU.hpp"

#include <xsimd/xsimd.hpp>
namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

class Model_CPU_fast : public Model_CPU
{
public:
    Model_CPU_fast(const Initstate& initstate, Particles& particles);

    virtual ~Model_CPU_fast() = default;

    virtual void step();

    void computeAcceleration(int start, int end);

    void computeAccelerationVectorized(b_type rposx_i, b_type rposy_i,
     b_type rposz_i, b_type raccx_i, b_type raccy_i, b_type raccz_i);
};
#endif // MODEL_CPU_FAST_HPP_

#endif // GALAX_MODEL_CPU_FAST
