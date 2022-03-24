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
    float xmin = *min_element(particles.x.begin(), particles.x.end());
    float ymin = *min_element(particles.y.begin(), particles.y.end());
    float zmin = *min_element(particles.z.begin(), particles.z.end());
    float xmax = *max_element(particles.x.begin(), particles.x.end());
    float ymax = *max_element(particles.y.begin(), particles.y.end());
    float zmax = *max_element(particles.z.begin(), particles.z.end());

    float rect[6] = {xmin, xmax, ymin, ymax, zmin, zmax};
    this->root = newQuadrant(rect);
    
    
        omp_set_dynamic(0); 
    omp_set_num_threads(12);
        
        for (int i = 0; i < n_particles; i++){
            insertIntoNode(i, root);
        }
        
        updateCoM(root);

        for(int i = 0; i < n_particles; i++){
            bodies.push_back(i);
        }
    
 
    
    
}

void Model_CPU_fast
::step()
{
    
    
    
    //float min = *min_element(particles.x.begin(), particles.x.end());
    //std::cout << min << "   mminnnnnnnnnnnnnn" << std::endl; 
    // Get Root Quadrant boundraries
    
        
    /*float xmin = *min_element(particles.x.begin(), particles.x.end()) - 0.05;
    float ymin = *min_element(particles.y.begin(), particles.y.end()) - 0.05;
    float zmin = *min_element(particles.z.begin(), particles.z.end()) - 0.05;
    float xmax = *max_element(particles.x.begin(), particles.x.end()) + 0.05;
    float ymax = *max_element(particles.y.begin(), particles.y.end()) + 0.05;
    float zmax = *max_element(particles.z.begin(), particles.z.end()) + 0.05;*/

    float xmin = 0;
    float ymin = 0;
    float zmin = 0;
    float xmax = 0;
    float ymax = 0;
    float zmax = 0;

    for(int i = 0; i < bodies.size(); i++){
        int j = bodies[i];
        if (xmin > particles.x[j]){
            xmin = particles.x[j];
        }
        if (ymin > particles.y[j]){
            ymin = particles.y[j];
        }
        if (zmin > particles.z[j]){
            zmin = particles.z[j];
        }

        if (xmax < particles.x[j]){
            xmax = particles.x[j];
        }
        if (ymax < particles.y[j]){
            ymax = particles.y[j];
        }
        if (zmax < particles.z[j]){
            zmax = particles.z[j];
        }
}

    xmin -= 0.05;
    ymin -= 0.05;
    zmin -= 0.05;
    xmax += 0.05;
    ymax += 0.05;
    zmax += 0.05;

    float rect[6] = {xmin, xmax, ymin, ymax, zmin, zmax};
    delete this->root;
    this->root = newQuadrant(rect);
    
    
        omp_set_dynamic(0); 
         omp_set_num_threads(12);
        
        for (int i = 0; i < bodies.size(); i++){
            insertIntoNode(bodies[i], root);
        }
        
        updateCoM(root);

    
       
       #pragma omp parallel for 
       for(int i = 0; i < bodies.size(); i++){
           accelerationsx[bodies[i]] = 0;
           accelerationsy[bodies[i]] = 0;
           accelerationsz[bodies[i]] = 0;
           calculateAcceleration(bodies[i], root, 0.5);
       }
        //std::cout << accelerationsx[0] << "  ,  " << accelerationsy[0] << std::endl;
        
        
        // SIMD THIS
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

        for(int i = 0; i < bodies.size(); i++){
            if(isOutlier(bodies[i], root, 2.0)){
                auto it = find(bodies.begin(), bodies.end(), bodies[i]);
                bodies.erase(it);
            }
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
        if(quad->rect[1] - quad->rect[0] < 0.1){
            quad->totalMass += initstate.masses[bodyId];
        }else{
    	subdivide(quad);
        // Parallelize these insertions
        
        for (int i = 0; i < 8; i++){
            insertIntoNode(bodyId, quad->children[i]);
        }
        for (int i = 0; i < 8; i++){
            insertIntoNode(quad->bodyId, quad->children[i]);
       }
        }
    }
}

bool Model_CPU_fast
::isBodyInQuad(int bodyId, Quadrant *quad)
{
    float x = particles.x[bodyId], y = particles.y[bodyId], z = particles.z[bodyId];
    float *rect = quad->rect;
    if (x >= rect[0] && x < rect[1] && y >= rect[2] && y < rect[3] && z >= rect[4] && z < rect[5]){
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
			quad->totalMass += initstate.masses[id];
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
        updateCounter++;
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

        float dx = quad->centerOfMass[0] - particles.x[bodyId];
        float dy = quad->centerOfMass[1] - particles.y[bodyId];
        float dz = quad->centerOfMass[2] - particles.z[bodyId];

        float d = dx * dx + dy * dy + dz * dz;
        
        if (std::sqrt(2) * s / std::sqrt(d) < theta){
            //std::cout << "Node" << std::endl;
            b2nAcc(bodyId, quad, d, dx, dy, dz);
            
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

void Model_CPU_fast::b2nAcc(int i, Quadrant * quad, float dij, float diffx, float diffy, float diffz){

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

int Model_CPU_fast::countroots(Quadrant * quad){
    int result = 0;
    
    if(quad->type == 'i'){
        for(int i = 0; i < 8 ; i++){
            result += countroots(quad->children[i]);
        }
        
    }else if(quad->bodyId != -1){
        //std::cout << quad->bodyId << std::endl;
        return 1;
    }
    return result;
}

bool Model_CPU_fast::isOutlier(int bodyId, Quadrant * quad, float theta){
    bool outlier = false;
    float s = quad->rect[1]  - quad->rect[0];
    float dx =  quad->centerOfMass[0] - particles.x[bodyId];
    float dy = quad->centerOfMass[1] - particles.y[bodyId];
    float dz = quad->centerOfMass[2] - particles.z[bodyId];
    float d = std::sqrt(dx * dx + dy * dy + dz * dz);

    if(d / s > theta){
        // Check escape velocity
        float velx = velocitiesx[bodyId];
        float vely = velocitiesx[bodyId];
        float velz = velocitiesx[bodyId];
        float espace_vel = std::sqrt(20 * quad->totalMass / d);
        float body_vel = std::sqrt(velx * velx + vely * vely + velz * velz);

        if(espace_vel < body_vel){
            return true;
        }

    }
    return false;
}




#endif // GALAX_MODEL_CPU_FAST
