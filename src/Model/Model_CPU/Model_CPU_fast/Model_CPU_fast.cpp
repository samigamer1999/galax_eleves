#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <omp.h>

#include "Model_CPU_fast.hpp"




Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
   
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);
    
    
 
    
    
}

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);
    
    
    //float min = *min_element(particles.x.begin(), particles.x.end());
    //std::cout << min << "   mminnnnnnnnnnnnnn" << std::endl; 
    // Get Root Quadrant boundraries
    float rect[6] = {-50.0, 50.0, -50.0, 50.0, -50.0, 50.0};
    Quadrant * root = newQuadrant(rect);
    
    
        omp_set_dynamic(0); 
    omp_set_num_threads(4);

        for (int i = 0; i < n_particles; i++){
            insertIntoNode(i, root);
        }
        
        updateCoM(root);
        
       
       #pragma omp parallel for 
       for(int i = 0; i < n_particles; i++){
           calculateAcceleration(i, root, 0.0);
       }
        
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i++){
          velocitiesx[i] += accelerationsx[i] * 2.0f;
		  velocitiesy[i] += accelerationsy[i] * 2.0f;
		  velocitiesz[i] += accelerationsz[i] * 2.0f;
		  particles.x[i] += velocitiesx   [i] * 0.1f;
		  particles.y[i] += velocitiesy   [i] * 0.1f;
		  particles.z[i] += velocitiesz   [i] * 0.1f;

          //std::cout << accelerationsx[i] << "   " << std::endl; 
        }
    
}


Quadrant * Model_CPU_fast
::newQuadrant(float *rect)
{
    Quadrant * quad = new Quadrant{-1, {0, 0, 0}, 0, {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}, {0, 0, 0, 0, 0, 0}, 'l'};
    memcpy(quad->rect, rect, 6 * sizeof(float));
    return quad;
}

void Model_CPU_fast
::insertIntoNode(int bodyId, Quadrant *quad)
{
    if (!isBodyInQuad(bodyId, quad)){
        return;
    }

    if(quad->bodyId == -1){
    	quad->bodyId = bodyId;
        return;
    }

    if (quad->type == 'i'){
       for (int i = 0; i < 8; i++){
            insertIntoNode(bodyId, quad->children[i]);
       }

    }
    
    if (quad->type == 'l' && quad->bodyId != -1){
    	subdivide(quad);
        for (int i = 0; i < 8; i++){
            insertIntoNode(bodyId, quad->children[i]);
        }
        for (int i = 0; i < 8; i++){
            insertIntoNode(quad->bodyId, quad->children[i]);
       }
   
    }
}

bool Model_CPU_fast
::isBodyInQuad(int bodyId, Quadrant *quad)
{
    float x = particles.x[bodyId], y = particles.y[bodyId], z = particles.z[bodyId];
    float *rect = quad->rect;
    if (x >= rect[0] && x <= rect[1] && y >= rect[2] && y <= rect[3] && z >= rect[4] && z <= rect[5]){
            return true;
    }
    return false;
}

void Model_CPU_fast
::updateCoM(Quadrant *quad)
{

    int id = quad->bodyId;
    
   if (quad->type == 'l'){
		if (quad->bodyId != -1){
			quad->centerOfMass[0] = particles.x[id]; 
            quad->centerOfMass[1] = particles.y[id];
            quad->centerOfMass[2] =  particles.z[id];
			quad->totalMass = initstate.masses[id];
		}
	}else{
			for(int i = 0 ; i < 8 ; i++){
				updateCoM(quad->children[i]);
				quad->totalMass += quad->children[i]->totalMass;
                //std::cout << quad->totalMass << "  bbb " << std::endl;
                quad->centerOfMass[0] +=  quad->children[i]->centerOfMass[0] * quad->children[i]->totalMass;
                quad->centerOfMass[1] +=  quad->children[i]->centerOfMass[1] * quad->children[i]->totalMass;
                quad->centerOfMass[2] +=  quad->children[i]->centerOfMass[2] * quad->children[i]->totalMass;
			}
			quad->centerOfMass[0] *= 1 / quad->totalMass; 
            quad->centerOfMass[1] *= 1 / quad->totalMass;
            quad->centerOfMass[2] *=  1 / quad->totalMass;
	}
   
}

void Model_CPU_fast
::subdivide(Quadrant *quad)
{
    quad->type = 'i';
    float *rect = quad->rect;
    float x0 = rect[0], y0 = rect[2], z0 = rect[4];
    float x1 = rect[1], y1 = rect[3], z1 = rect[5];
    float dx = (rect[1] - rect[0]) / 2.0;
    float dy = (rect[3] - rect[2]) / 2.0;
    float dz = (rect[5] - rect[4]) / 2.0;

    float rects[8][6] = {{x0, x0 + dx, y0, y0 + dy, z0, z0 + dz},
                         {x0, x0 + dx, y0 + dy, y1, z0, z0 + dz},
                         {x0 + dx, x1, y0, y0 + dy, z0, z0 + dz},
                         {x0 + dx, x1, y0 + dy, y1, z0, z0 + dz},
                         {x0, x0 + dx, y0, y0 + dy, z0 + dz, z1},
                         {x0, x0 + dx, y0 + dy, y1, z0 + dz, z1},
                         {x0 + dx, x1, y0, y0 + dy, z0 + dz, z1},
                         {x0 + dx, x1, y0 + dy, y1, z0 + dz, z1}};


    for(int i = 0 ; i < 8 ; i++){
        
        Quadrant * child = newQuadrant(rects[i]);
        quad->children[i] = child;
        
    }
 
   
    
    
}

void Model_CPU_fast::calculateAcceleration(int bodyId, Quadrant * quad, float theta){
    if (quad->type == 'l'){
    	 if (quad->bodyId == -1){
    	 	return;
    	 }else if( quad->bodyId != bodyId){
            b2bAcc(bodyId, quad->bodyId);
        }
    }else{
    	// Compare s/d with theta, s : width of the region, d : distance between body and center of mass of the node
        float s = quad->rect[1]  - quad->rect[0];

        float dx = pow(quad->centerOfMass[0] - particles.x[bodyId], 2);
        float dy = pow(quad->centerOfMass[1] - particles.y[bodyId], 2);
        float dz = pow(quad->centerOfMass[2] - particles.z[bodyId], 2);

        float d = sqrt(dx + dy +dz);
        
        if (s / d < theta){
            
            b2nAcc(bodyId, quad);
            
        }else{
            for (int i = 0; i < 8; i++){
                calculateAcceleration(bodyId, quad->children[i], theta);
            }
        }
    }
}

void Model_CPU_fast::b2bAcc(int i, int j){
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

void Model_CPU_fast::b2nAcc(int i, Quadrant * quad){

    const float diffx = quad->centerOfMass[0] - particles.x[i];
    const float diffy = quad->centerOfMass[1] - particles.y[i];
    const float diffz = quad->centerOfMass[2] - particles.z[i];

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

    accelerationsx[i] += diffx * dij * quad->totalMass;
    accelerationsy[i] += diffy * dij * quad->totalMass;
    accelerationsz[i] += diffz * dij * quad->totalMass;
    //std::cout <<  quad->centerOfMass[0] << "    ";
}




#endif // GALAX_MODEL_CPU_FAST
