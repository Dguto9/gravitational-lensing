
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <SDL2/SDL.h>

#ifdef __CUDACC__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#ifdef __TINYC__
    #undef main
#endif

float* cellVals;
float fLen = 50;
float viewDist = 10;
float rayRes = 0.05;

int simW = 150;
int simH = 200;
int width = 800;
int height = 600;

typedef struct vec3_t {
    float x;
    float y;
    float z;
} vec3_t;

typedef struct box_t {
    vec3_t pos;
    vec3_t scale;
    vec3_t col;
} box_t;

typedef struct sphere_t {
    vec3_t pos;
    vec3_t vel;
    float radius;
    vec3_t col;
    float mass;
} sphere_t;

#ifdef __CUDACC__
float* cellValsGPU;
sphere_t* spheresGPU;
#endif

sphere_t* spheres;
int sphereCount = 1;
box_t* boxes;
int boxCount = 3;

box_t table = {{0, 0, -3.2}, {2.2, 1, 0.4}, {0.6, 0.6, 0.6}};
box_t p1 = {{-2.4, 0, -2.5}, {0.4, 0.4, 0.4}, {0.7, 0.7, 0.7}};
box_t p2 = {{2.4, 0, -2.5}, {0.4, 0.4, 0.4}, {0.7, 0.7, 0.7}};
sphere_t ball = {{0, 0, -1}, {0, 0, 0}, 0.1, {1, 1, 1}, 0.1};

box_t bounds = {{0,0,0}, {4,4,4}, {1,1,1}};

int keyA = 0;
int keyD = 0;
int keyW = 0;
int keyS = 0;
int keyJ = 0;
int keyL = 0;
int keyI = 0;
int keyK = 0;

float serve = 0.3;

int gravToggle = 1;
vec3_t camPos = {0,-3,-1.5};
vec3_t camRot = {0,0,0};
int tick = 0;
int testsum = 0;

float G = 0;

#ifdef __CUDACC__
__global__ void computeRaymarch(float* cellVals, int simW, int simH, sphere_t* spheres, int sphereCount, int gravToggle, float G, vec3_t camPos, vec3_t camRot, float fLen, float viewDist, float rayRes);
#else
void computeRaymarch(float* cellVals, int simW, int simH, sphere_t* spheres, int sphereCount, int gravToggle, float G, vec3_t camPos, vec3_t camRot, float fLen, float viewDist, float rayRes);
#endif
#ifdef __CUDACC__ 
__host__ __device__
#endif 
void normalize(vec3_t* v);
#ifdef __CUDACC__ 
__host__ __device__
#endif 
float distance(vec3_t* v);
#ifdef __CUDACC__ 
__host__ __device__
#endif 
float distance_squared(vec3_t* v);
#ifdef __CUDACC__ 
__host__ __device__
#endif 
void rotate(vec3_t* v, float yaw, float pitch);
void subtract(vec3_t* left, vec3_t* right);
void vPrint(vec3_t v);
#ifdef __CUDACC__ 
__host__ __device__
#endif
int sphereFunc(vec3_t pos, sphere_t* sphere);
int boxFunc(vec3_t pos, box_t* box);

int main(int argc, char **argv) {
    SDL_Init(SDL_INIT_VIDEO);

    cellVals = (float*)malloc(3*simW*simH*sizeof(float));
    spheres = (sphere_t*)malloc(sphereCount * sizeof(sphere_t));
    boxes = (box_t*)malloc(boxCount*sizeof(box_t));

    //for (int i = 0; i<sphereCount; i++){
    sphere_t currentsphere = {{0,0,-2}, {0,0,0}, 1,{1,0,0},1};//{{8*(rand()/(float)RAND_MAX)-4, 8*(rand()/(float)RAND_MAX)-4, 8*(rand()/(float)RAND_MAX)-4},0.5,{rand()/(float)RAND_MAX, rand()/(float)RAND_MAX, rand()/(float)RAND_MAX},0.1f*rand()/(float)RAND_MAX};
    spheres[0] = currentsphere;
    //spheres[1] = ball;
    // sphere_t currentsphere2 = {{2,0,2},1,{1,0,0},1};
    // spheres[1] = currentsphere2;
     //}
    boxes[0] = table;
    boxes[1] = p1;
    boxes[2] = p2;

#ifdef __CUDACC__
    cudaMalloc(&cellValsGPU, 3 * simW * simH * sizeof(float));
    cudaMalloc(&spheresGPU, sphereCount * sizeof(sphere_t));
    cudaMemcpy(spheresGPU, spheres, sphereCount*sizeof(sphere_t), cudaMemcpyHostToDevice);
#endif

    SDL_Window *window = SDL_CreateWindow("Ray Marching", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);    
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, simW, simH);
    //SDL_SetWindowFullscreen(window,SDL_WINDOW_FULLSCREEN_DESKTOP);
    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 50);
    SDL_SetRelativeMouseMode((SDL_bool)1);
    bool running = true;
    SDL_Event event;
    while(running) {
        while(SDL_PollEvent(&event)) {
            switch(event.type){
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_KEYDOWN:
                    switch(event.key.keysym.sym){
                        case SDLK_a:
                            keyA = 1;
                            break;
                        case SDLK_d:
                            keyD = 1;
                            break;
                        case SDLK_w:
                            keyW = 1;
                            break;
                        case SDLK_s:
                            keyS = 1;
                            break;
                        case SDLK_j:
                            keyJ = 1;
                            break;
                        case SDLK_l:
                            keyL = 1;
                            break;
                        case SDLK_i:
                            keyI = 1;
                            break;
                        case SDLK_k:
                            keyK = 1;
                            break;
                        case SDLK_LSHIFT:
                            gravToggle = 0;
                            break;
                        default:
                            break;           
                     }
                    break; 
                case SDL_KEYUP:
                    switch(event.key.keysym.sym){
                        case SDLK_a:
                            keyA = 0;
                            break;
                        case SDLK_d:
                            keyD = 0;
                            break;
                        case SDLK_w:
                            keyW = 0;
                            break;
                        case SDLK_s:
                            keyS = 0;
                            break;
                        case SDLK_j:
                            keyJ = 0;
                            break;
                        case SDLK_l:
                            keyL = 0;
                            break;
                        case SDLK_i:
                            keyI = 0;
                            break;
                        case SDLK_k:
                            keyK = 0;
                            break;
                        case SDLK_LSHIFT:
                            gravToggle = 1;
                            break;
                        default:
                            break;           
                     }
                    break;
                case SDL_MOUSEMOTION:
                    //camRot.z -= event.motion.xrel/(float)simW;
                    camRot.x -= event.motion.yrel/(float)simH;
                    if (camRot.x > M_PI / 2) camRot.x = M_PI / 2;
                    if (camRot.x < -M_PI / 2) camRot.x = -M_PI / 2;
                    break;
                case SDL_MOUSEWHEEL:
                    G += event.wheel.preciseY/50;
                    printf("G: %f\n", G);
                    break;
            }
        }

#ifdef __CUDACC__
        dim3 threadsPerBlock(16, 16);
        dim3 blockCount(simW / threadsPerBlock.x, simH / threadsPerBlock.y);
        computeRaymarch<<<blockCount, threadsPerBlock>>>(cellValsGPU, simW, simH, spheresGPU, sphereCount, gravToggle, G, camPos, camRot, fLen, viewDist, rayRes);
        cudaDeviceSynchronize();
        cudaMemcpy(cellVals, cellValsGPU, 3 * simW * simH * sizeof(float), cudaMemcpyDeviceToHost);
#else
        computeRaymarch(cellVals, simW, simH, spheres, sphereCount, gravToggle, G, camPos, camRot, fLen, viewDist, rayRes);
#endif
        vec3_t camForward = {0,1,0};
        vec3_t camRight = {1,0,0};
        rotate(&camForward, camRot.z, camRot.x);
        rotate(&camRight, camRot.z, camRot.x);
        camPos.x = -0.2*ball.pos.x;
        camRot.z = -0.05*ball.pos.x;

        //camPos.x += 0.1*((arrowU-arrowD)*camForward.x + (arrowR-arrowL)*camRight.x);
        //camPos.y += 0.1*((arrowU-arrowD)*camForward.y + (arrowR-arrowL)*camRight.y);
        //camPos.z += 0.1*((arrowU-arrowD)*camForward.z + (arrowR-arrowL)*camRight.z);
        
        p1.pos.y += 0.4*(keyD-keyA);
        p1.pos.z += 0.4*(keyW-keyS);
        p2.pos.y += 0.4*(keyJ-keyL);
        p2.pos.z += 0.4*(keyI-keyK);
        boxes[1] = p1;
        boxes[2] = p2;
        
        vec3_t dirTo = { spheres[0].pos.x - ball.pos.x, spheres[0].pos.y - ball.pos.y, spheres[0].pos.z - ball.pos.z };
        float dist = distance_squared(&dirTo);
        normalize(&dirTo);                    
        ball.vel.x += ((gravToggle) ? G : 0) * (dirTo.x * spheres[0].mass / (dist + 0.01));
        ball.vel.y += ((gravToggle) ? G : 0) * (dirTo.y * spheres[0].mass / (dist + 0.01));
        ball.vel.z += ((gravToggle) ? G : 0) * (dirTo.z * spheres[0].mass / (dist + 0.01)) - 0.05;
        ball.pos.x += ball.vel.x;
        ball.pos.y += ball.vel.y;
        ball.pos.z += ball.vel.z;
        //spheres[1] = ball;

        if(!boxFunc(ball.pos, &bounds)){
            ball.pos.x = ball.pos.y = ball.vel.y = ball.vel.z = 0;
            ball.pos.z = -1;
            ball.vel.x = serve;
            serve*=-1;
            spheres[0].pos.x = 4*(float)rand()/RAND_MAX - 2;
            spheres[0].pos.z = 2*(float)rand()/RAND_MAX - 1;
            
        }
        if(boxFunc(ball.pos, &table)){
            ball.vel.z *= -1;
            ball.pos.z += table.scale.z;
        }
        if(boxFunc(ball.pos, &p1)){
            ball.vel.x *= -1;
            ball.pos.x += p1.scale.x;
        }
        if(boxFunc(ball.pos, &p2)){
            ball.vel.x *= -1;
            ball.pos.x += p2.scale.x;
        }
        
        tick++;

        int* pixels;
        int pitch;
        SDL_LockTexture(texture, NULL, (void**)&pixels, &pitch);
        for (int i = 0; i < simW*simH*3; i+=3){
            uint8_t r = cellVals[i]*255;
            uint8_t g = cellVals[i+1]*255;
            uint8_t b = cellVals[i+2]*255;
            pixels[i/3] = (r << 24 | g << 16 | b << 8 | 0xFF);
        }
        SDL_UnlockTexture(texture);

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

#ifdef __CUDACC__
__global__ void computeRaymarch(float* cellVals, int simW, int simH, sphere_t* spheres, int sphereCount, int gravToggle, float G, vec3_t camPos, vec3_t camRot, float fLen, float viewDist, float rayRes) {
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < simH; i += blockDim.y * gridDim.y) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < simW; j += blockDim.x * gridDim.x) {
            vec3_t ray = { j - (simW / 2), fLen, -(i - (simH / 2)) };
            normalize(&ray);
            rotate(&ray, camRot.z, camRot.x);
            vec3_t raymarch = camPos;
            for (int k = 0; k < (int)(viewDist/rayRes); k++) {
                raymarch.x += ray.x * rayRes;
                raymarch.y += ray.y * rayRes;
                raymarch.z += ray.z * rayRes;
                for (int l = 0; l < sphereCount; l++) {
                    vec3_t dirTo = { spheres[l].pos.x - raymarch.x, spheres[l].pos.y - raymarch.y, spheres[l].pos.z - raymarch.z };
                    float dist = distance_squared(&dirTo);
                    normalize(&dirTo);                    
                    ray.x += ((gravToggle) ? G : 0) * (dirTo.x * spheres[l].mass / (dist + 0.01)) * rayRes;
                    ray.y += ((gravToggle) ? G : 0) * (dirTo.y * spheres[l].mass / (dist + 0.01)) * rayRes;
                    ray.z += ((gravToggle) ? G : 0) * (dirTo.z * spheres[l].mass / (dist + 0.01)) * rayRes;
                    if (sphereFunc(raymarch, &spheres[l])) {
                        cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * spheres[l].col.x;
                        cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * spheres[l].col.y;
                        cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * spheres[l].col.z;
                        k = (int)(viewDist / rayRes);
                        break;
                    }
                    else {
                        for (int m = 0; m < 3; m++) {
                            cellVals[(3 * j) + m + (3 * i * simW)] = 0;
                        }
                    }
                }
                if (raymarch.x > 4 || raymarch.y > 4 || raymarch.z < -4) {
                    for (int m = 0; m < 3; m++) {
                        cellVals[(3 * j) + m + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.9) * (((int)raymarch.y % 2) ? 1 : -0.9) * (((int)raymarch.z % 2) ? 1 : -0.9) + 1) * 0.5;
                    }
                    break;
                }
            }
        }
    }
}
#else
void computeRaymarch(float* cellVals, int simW, int simH, sphere_t* spheres, int sphereCount, int gravToggle, float G, vec3_t camPos, vec3_t camRot, float fLen, float viewDist, float rayRes) {
    for (int i = 0; i < simH; i++) {
        for (int j = 0; j < simW; j++) {
            vec3_t ray = { j - (simW / 2.0f), fLen, -(i - (simH / 2.0f)) };
            normalize(&ray);
            rotate(&ray, camRot.z, camRot.x);
            vec3_t raymarch = camPos;
            for (int k = 0; k < (int)(viewDist/rayRes); k++) {
                raymarch.x += ray.x * rayRes;
                raymarch.y += ray.y * rayRes;
                raymarch.z += ray.z * rayRes;
                for (int l = 0; l < sphereCount; l++) {
                    vec3_t dirTo = { spheres[l].pos.x - raymarch.x, spheres[l].pos.y - raymarch.y, spheres[l].pos.z - raymarch.z };
                    float dist = distance_squared(&dirTo);
                    normalize(&dirTo);
                    ray.x += ((gravToggle) ? G : 0) * (dirTo.x * spheres[l].mass / (dist+0.001)) * 0.1;
                    ray.y += ((gravToggle) ? G : 0) * (dirTo.y * spheres[l].mass / (dist+0.001)) * 0.1;
                    ray.z += ((gravToggle) ? G : 0) * (dirTo.z * spheres[l].mass / (dist+0.001)) * 0.1;
                    // if (sphereFunc(raymarch, &spheres[l])) {
                    //     cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / 150)) * spheres[l].col.x;
                    //     cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / 150)) * spheres[l].col.y;
                    //     cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / 150)) * spheres[l].col.z;
                    //     k = 150;
                    //     break;
                    // }
                    // else {
                    // }
                }
                
                for (int m = 0; m < 3; m++) {
                    cellVals[(3 * j) + m + (3 * i * simW)] = 0;
                }
                for (int m = 0; m < boxCount; m++){
                    if (boxFunc(raymarch, &boxes[m])){
                        cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * boxes[m].col.x;
                        cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * boxes[m].col.y;
                        cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * boxes[m].col.z;
                        k = (int)(viewDist/rayRes);
                        break;
                    }
                }
                if (sphereFunc(raymarch, &ball)){
                    cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ball.col.x;
                    cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ball.col.y;
                    cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ball.col.z;
                    break;
                }
                if (raymarch.x > 4) {
                    cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))); 
                    cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes)));
                    cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.5) * (((int)raymarch.y % 2) ? 1 : -0.5) * (((int)raymarch.z % 2) ? 1 : -0.5) + 1) * 0.5;
                    break;
                }
                else if (raymarch.y > 4) {
                    cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))); 
                    cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes)));
                    cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.5) * (((int)raymarch.y % 2) ? 1 : -0.5) * (((int)raymarch.z % 2) ? 1 : -0.5) + 1) * 0.5;
                    break;
                }
                else if (raymarch.z > 4) {
                    cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))); 
                    cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes)));
                    cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.5) * (((int)raymarch.y % 2) ? 1 : -0.5) * (((int)raymarch.z % 2) ? 1 : -0.5) + 1) * 0.5;
                    break;
                }
                else if (raymarch.x < -4) {
                    cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))); 
                    cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.5) * (((int)raymarch.y % 2) ? 1 : -0.5) * (((int)raymarch.z % 2) ? 1 : -0.5) + 1) * 0.5;
                    cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.5) * (((int)raymarch.y % 2) ? 1 : -0.5) * (((int)raymarch.z % 2) ? 1 : -0.5) + 1) * 0.5;
                    break;
                }
                else if (raymarch.y < -4) {
                    cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))); 
                    cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.5) * (((int)raymarch.y % 2) ? 1 : -0.5) * (((int)raymarch.z % 2) ? 1 : -0.5) + 1) * 0.5;
                    cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.5) * (((int)raymarch.y % 2) ? 1 : -0.5) * (((int)raymarch.z % 2) ? 1 : -0.5) + 1) * 0.5;
                    break;
                }
                else if (raymarch.z < -4) {
                    cellVals[(3 * j) + 1 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))); 
                    cellVals[(3 * j) + 2 + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.5) * (((int)raymarch.y % 2) ? 1 : -0.5) * (((int)raymarch.z % 2) ? 1 : -0.5) + 1) * 0.5;
                    cellVals[(3 * j) + (3 * i * simW)] = (1 - ((float)k / (viewDist / rayRes))) * ((((int)raymarch.x % 2) ? 1 : -0.5) * (((int)raymarch.y % 2) ? 1 : -0.5) * (((int)raymarch.z % 2) ? 1 : -0.5) + 1) * 0.5;
                    break;
                }
            }
        }
    }
}
#endif

#ifdef __CUDACC__ 
__host__ __device__ 
#endif 
void normalize(vec3_t* v){
    float magn = sqrt((v->x*v->x) + (v->y*v->y) + (v->z*v->z));
    v->x /= magn;
    v->y /= magn;
    v->z /= magn;
}

#ifdef __CUDACC__ 
__host__ __device__
#endif 
float distance(vec3_t* v) {
    return sqrt((v->x * v->x) + (v->y * v->y) + (v->z * v->z));
}

#ifdef __CUDACC__ 
__host__ __device__
#endif 
float distance_squared(vec3_t* v) {
    return (v->x * v->x) + (v->y * v->y) + (v->z * v->z);
}

#ifdef __CUDACC__ 
__host__ __device__
#endif 
void rotate(vec3_t* v, float yaw, float pitch){
    vec3_t rot;
    rot.x = v->x*cos(yaw) - v->y*sin(yaw)*cos(pitch) + v->z*sin(yaw)*sin(pitch);
    rot.y = v->x*sin(yaw) + v->y*cos(yaw)*cos(pitch) - v->z*sin(pitch)*cos(yaw);
    rot.z =                 v->y*sin(pitch)          + v->z*cos(pitch);
    *v = rot;
}

void subtract(vec3_t* left, vec3_t* right){
    left->x -= right->x;
    left->y -= right->y;
    left->z -= right->z; 
}

void vPrint(vec3_t v){
    printf("{%f,%f,%f}\n",v.x,v.y,v.z);   
}

int boxFunc(vec3_t pos, box_t* box){
    subtract(&pos, &box->pos);
    return (abs(pos.x)<box->scale.x) && (abs(pos.y)<box->scale.y) && (abs(pos.z)<box->scale.z);
}

#ifdef __CUDACC__ 
__host__ __device__
#endif
int sphereFunc(vec3_t pos, sphere_t* sphere){
    return (sphere->pos.x-pos.x)*(sphere->pos.x-pos.x)+(sphere->pos.y-pos.y)*(sphere->pos.y-pos.y)+(sphere->pos.z-pos.z)*(sphere->pos.z-pos.z) < (sphere->radius*sphere->radius);
}
