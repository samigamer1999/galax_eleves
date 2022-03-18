#ifdef GALAX_MODEL_CPU_FAST

#ifndef MODEL_CPU_FAST_HPP_
#define MODEL_CPU_FAST_HPP_

#include "../Model_CPU.hpp"

struct Quadrant
{
    int bodyId;
    float centerOfMass[3]; 
    float totalMass;
    struct Quadrant* children[8];
    float rect[6];
    char type;
};



class Model_CPU_fast : public Model_CPU
{
public:
    Model_CPU_fast(const Initstate& initstate, Particles& particles);

    virtual ~Model_CPU_fast() = default;

    virtual void step();

    Quadrant * newQuadrant(float rect[6]);

    void insertIntoNode(int bodyId, Quadrant *quad);

    bool isBodyInQuad(int bodyId, Quadrant *quad);

    void updateCoM(Quadrant *quad); // Update quadrant center of mass

    void subdivide(Quadrant *quad);

    void calculateAcceleration(int bodyId, Quadrant * quad, float theta);

    void b2bAcc(int i, int j);

    void b2nAcc(int i, Quadrant * quad);

};
#endif // MODEL_CPU_FAST_HPP_

#endif // GALAX_MODEL_CPU_FAST
