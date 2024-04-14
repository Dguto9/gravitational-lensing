
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
float fLen = 100;

int simW = 160;
int simH = 120;
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
} box_t;

typedef struct sphere_t {
    vec3_t pos;
    float radius;
    vec3_t col;
    float mass;
} sphere_t;

sphere_t* spheres;
int sphereCount = 1;
int arrowL = 0;
int arrowR = 0;
int arrowU = 0;
int arrowD = 0;
int gravToggle = 1;
vec3_t camPos = {0,0,0};
vec3_t camRot = {0,0,0};
int tick = 0;

float G = 0;

void normalize(vec3_t* v);
float distance(vec3_t* v);
void rotate(vec3_t* v, float yaw, float pitch);
void vPrint(vec3_t v);
int sphereFunc(vec3_t pos, sphere_t* sphere);

int main(int argc, char **argv) {
    SDL_Init(SDL_INIT_VIDEO);

    cellVals = (float*)malloc(3*simW*simH*sizeof(float));
    spheres = (sphere_t*)malloc(sphereCount*sizeof(sphere_t));

    for (int i = 0; i<sphereCount; i++){
        spheres[i] = sphere_t{{8*(rand()/(float)RAND_MAX)-4, 8*(rand()/(float)RAND_MAX)-4, 8*(rand()/(float)RAND_MAX)-4},0.5,{rand()/(float)RAND_MAX, rand()/(float)RAND_MAX, rand()/(float)RAND_MAX},0.1f*rand()/(float)RAND_MAX};
    }

    SDL_Window *window = SDL_CreateWindow("Ray Marching", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);    
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, simW, simH);
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
                            arrowL = 1;
                            break;
                        case SDLK_d:
                            arrowR = 1;
                            break;
                        case SDLK_w:
                            arrowU = 1;
                            break;
                        case SDLK_s:
                            arrowD = 1;
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
                            arrowL = 0;
                            break;
                        case SDLK_d:
                            arrowR = 0;
                            break;
                        case SDLK_w:
                            arrowU = 0;
                            break;
                        case SDLK_s:
                            arrowD = 0;
                            break;
                        case SDLK_LSHIFT:
                            gravToggle = 1;
                            break;
                        default:
                            break;           
                     }
                    break;
                case SDL_MOUSEMOTION:
                    camRot.z -= event.motion.xrel/(float)simW;
                    camRot.x -= event.motion.yrel/(float)simH;
                    if (camRot.x > M_PI / 2) camRot.x = M_PI / 2;
                    if (camRot.x < -M_PI / 2) camRot.x = -M_PI / 2;
                    break;
                case SDL_MOUSEWHEEL:
                    G += event.wheel.preciseY/10;
                    printf("G: %f\n", G);
                    break;
            }
        }

        for (int i = 0; i < simH; i++){
            for (int j = 0; j < simW; j++){
                vec3_t ray = {j-(simW/2), fLen, -(i-(simH/2))};                
                normalize(&ray);
                rotate(&ray, camRot.z, camRot.x);
                normalize(&ray);
                vec3_t raymarch = camPos;
                for (int k = 0; k < 75; k++){
                    raymarch.x += ray.x*0.1;
                    raymarch.y += ray.y*0.1;
                    raymarch.z += ray.z*0.1;
                    for (int l = 0; l < sphereCount; l++){
                        vec3_t dirTo = {spheres[l].pos.x - raymarch.x, spheres[l].pos.y - raymarch.y, spheres[l].pos.z - raymarch.z };
                        normalize(&dirTo);
                        float dist = distance(&dirTo);
                        ray.x += ((gravToggle) ? G : 0) *(dirTo.x * spheres[l].mass / (dist * dist))*0.1;
                        ray.y += ((gravToggle) ? G : 0) *(dirTo.y * spheres[l].mass / (dist * dist))*0.1;
                        ray.z += ((gravToggle) ? G : 0) *(dirTo.z * spheres[l].mass / (dist * dist))*0.1;
                        if (sphereFunc(raymarch, &spheres[l])){
                            cellVals[(3*j) + (3*i*simW)] = (1-((float)k/75))*spheres[l].col.x;
                            cellVals[(3*j)+1 + (3*i*simW)] = (1-((float)k/75))*spheres[l].col.y;
                            cellVals[(3*j)+2 + (3*i*simW)] = (1-((float)k/75))*spheres[l].col.z;
                            k = 75;
                            break;
                        }
                        else{
                            for (int m = 0; m < 3; m++){
                                cellVals[(3*j)+m+(3*i*simW)] = 0;
                            }
                        }
                    }
                    if (raymarch.x > 4 || raymarch.y > 4 || raymarch.z < -4) {
                        for (int m = 0; m < 3; m++) {
                            cellVals[(3 * j) + m + (3 * i * simW)] = (1 - ((float)k / 75)) * ((((int)raymarch.x % 2) ? 1 : -1) * (((int)raymarch.y % 2) ? 1 : -1) * (((int)raymarch.z % 2) ? 1 : -1) + 1) * 0.5;
                        }
                        break;
                    }
                }
            }
        }
        vec3_t camForward = {0,1,0};
        vec3_t camRight = {1,0,0};
        rotate(&camForward, camRot.z, camRot.x);
        rotate(&camRight, camRot.z, camRot.x);

        camPos.x += 0.1*((arrowU-arrowD)*camForward.x + (arrowR-arrowL)*camRight.x);
        camPos.y += 0.1*((arrowU-arrowD)*camForward.y + (arrowR-arrowL)*camRight.y);
        camPos.z += 0.1*((arrowU-arrowD)*camForward.z + (arrowR-arrowL)*camRight.z);

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

void normalize(vec3_t* v){
    float magn = sqrt((v->x*v->x) + (v->y*v->y) + (v->z*v->z));
    v->x /= magn;
    v->y /= magn;
    v->z /= magn;
}

float distance(vec3_t* v) {
    return sqrt((v->x * v->x) + (v->y * v->y) + (v->z * v->z));
}

void rotate(vec3_t* v, float yaw, float pitch){
    vec3_t rot;
    rot.x = v->x*cos(yaw) - v->y*sin(yaw)*cos(pitch) + v->z*sin(yaw)*sin(pitch);
    rot.y = v->x*sin(yaw) + v->y*cos(yaw)*cos(pitch) - v->z*sin(pitch)*cos(yaw);
    rot.z =                 v->y*sin(pitch)          + v->z*cos(pitch);
    *v = rot;
}

void vPrint(vec3_t v){
    printf("{%f,%f,%f}\n",v.x,v.y,v.z);   
}

int boxFunc(vec3_t pos, box_t box){
    return (pos.x < box.pos.x+box.scale.x) && (pos.x > box.pos.x) && (pos.y < box.pos.y+box.scale.y) && (pos.y > box.pos.y) && (pos.z < box.pos.z+box.scale.z) && (pos.z > box.pos.z);
}

int sphereFunc(vec3_t pos, sphere_t* sphere){
    return (sphere->pos.x-pos.x)*(sphere->pos.x-pos.x)+(sphere->pos.y-pos.y)*(sphere->pos.y-pos.y)+(sphere->pos.z-pos.z)*(sphere->pos.z-pos.z) < (sphere->radius*sphere->radius);
}