#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;
b_type one_vec = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
b_type zero_vec = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
b_type pone_vec = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
b_type two_vec = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);


    int vec_size = n_particles - n_particles % b_type::size;
    // OMP + xsimd version

    for (int i = 0; i < vec_size; i += b_type::size)
    {
        // load registers body i
        b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
        b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
        b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
        b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);


        for (int j = 0; j < n_particles; j++)
        {
    
            b_type diffx =  b_type::load_unaligned(&particles.x[j]) - rposx_i; 
            b_type diffy =  b_type::load_unaligned(&particles.y[j]) - rposy_i; 
            b_type diffz =  b_type::load_unaligned(&particles.z[j]) - rposz_i; 

            b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;
            

            b_type ten_vec = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
            b_type comp = dij -  one_vec;
            xs::batch_bool<float, xs::avx2> dij_1 = comp < zero_vec;

            dij = xs::sqrt(dij);
            b_type res_acc = xs::select(dij_1, ten_vec, ten_vec / (dij * dij * dij));
            //dij = dij_1 * ten_vec +  ten_vec / (dij_2 * dij_2);


            raccx_i += diffx * res_acc * b_type::load_unaligned(&initstate.masses[j]);
            raccy_i += diffy * res_acc * b_type::load_unaligned(&initstate.masses[j]);
            raccz_i += diffz * res_acc * b_type::load_unaligned(&initstate.masses[j]);
        }
        
        rvelx_i = rvelx_i + raccx_i * two_vec;
        rvely_i = rvely_i + raccy_i * two_vec;
        rvelz_i = rvelz_i + raccz_i * two_vec;

        rposx_i = rposx_i + rvelx_i * pone_vec;
        rposy_i = rposy_i + rvely_i * pone_vec;
        rposz_i = rposz_i + rvelz_i * pone_vec;
        
        rposx_i.store_unaligned(&particles.x[i]);
        rposy_i.store_unaligned(&particles.y[i]);
        rposz_i.store_unaligned(&particles.z[i]);
        
        

    
    }
}

void Model_CPU_fast::computeAcceleration(int start, int end){

    for (int i = start; i < end+1; i++)
	{
        for (int j = 0; j < n_particles; j++)
            {
                if(i != j)
                {
                    const float diffx = particles.x[j] - particles.x[i];
                    const float diffy = particles.y[j] - particles.y[i];
                    const float diffz = particles.z[j] - particles.z[i];

                    float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                    if (dij < 1.0)
                    {
                        dij = 10.0;
                    }
                    else
                    {
                        dij = std::sqrt(dij);
                        dij = 10.0 / (dij * dij * dij);
                    }

                    accelerationsx[i] += diffx * dij * initstate.masses[j];
                    accelerationsy[i] += diffy * dij * initstate.masses[j];
                    accelerationsz[i] += diffz * dij * initstate.masses[j];
                }
            }
	}
    
}

void Model_CPU_fast::computeAccelerationVectorized(b_type rposx_i, b_type rposy_i,
     b_type rposz_i, b_type raccx_i, b_type raccy_i, b_type raccz_i){
    
        for (int j = 0; j < n_particles; j++)
        {
    
            b_type diffx =  b_type::load_unaligned(&particles.x[j]) - rposx_i; 
            b_type diffy =  b_type::load_unaligned(&particles.y[j]) - rposy_i; 
            b_type diffz =  b_type::load_unaligned(&particles.z[j]) - rposz_i; 

            b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;
            

            b_type ten_vec = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
            b_type comp = dij -  one_vec;
            xs::batch_bool<float, xs::avx2> dij_1 = comp < zero_vec;

            dij = xs::sqrt(dij);
            b_type res_acc = xs::select(dij_1, ten_vec, ten_vec / (dij * dij * dij));
            //dij = dij_1 * ten_vec +  ten_vec / (dij_2 * dij_2);

            
            

            raccx_i += diffx * res_acc * b_type::load_unaligned(&initstate.masses[j]);
            raccy_i += diffy * res_acc * b_type::load_unaligned(&initstate.masses[j]);
            raccz_i += diffz * res_acc * b_type::load_unaligned(&initstate.masses[j]);
        }
  
}

#endif // GALAX_MODEL_CPU_FAST
