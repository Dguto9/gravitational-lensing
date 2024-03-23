
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <SDL2/SDL.h>
// Normally SDL2 will redefine the main entry point of the program for Windows applications
// this doesn't seem to play nice with TCC, so we just undefine the redefinition
#ifdef __TINYC__
    #undef main
#endif

float* cellVals;
float fLen = 100;

int simW = 200;
int simH = 150;
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
} sphere_t;

box_t box = {{-1,4,-1},{2,2,2}};
sphere_t sphere = {{-2,3,-1},0.5};
int arrowL = 0;
int arrowR = 0;
int arrowU = 0;
int arrowD = 0;
vec3_t camPos = {0,0,0};
vec3_t camRot = {0,0,0};
int tick = 0;

void normalize(vec3_t* v);
void rotate(vec3_t* v, float yaw, float pitch);
void vPrint(vec3_t v);

int main(int argc, char **argv) {
  SDL_Init(SDL_INIT_VIDEO);

	cellVals = malloc(simW*simH*sizeof(float));

    SDL_Window *window = SDL_CreateWindow("Ray Marching", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);    
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, simW, simH);
    SDL_SetRelativeMouseMode(1);
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
                        case SDLK_LEFT:
                            arrowL = 1;
                            break;
                        case SDLK_RIGHT:
                            arrowR = 1;
                            break;
                        case SDLK_UP:
                            arrowU = 1;
                            break;
                        case SDLK_DOWN:
                            arrowD = 1;
                            break;
                        default:
                            break;           
                     }
                    break; 
                case SDL_KEYUP:
                    switch(event.key.keysym.sym){
                        case SDLK_LEFT:
                            arrowL = 0;
                            break;
                        case SDLK_RIGHT:
                            arrowR = 0;
                            break;
                        case SDLK_UP:
                            arrowU = 0;
                            break;
                        case SDLK_DOWN:
                            arrowD = 0;
                            break;
                        default:
                            break;           
                     }
                    break;
                case SDL_MOUSEMOTION:
                    camRot.z += event.motion.xrel/(float)simW;
                    camRot.x += event.motion.yrel/(float)simH;
                    printf("%i\n",event.motion.yrel);
                    //printf("Early: {%f,%f}\n",camRot.z,camRot.x);
                    break;        
            }
        }

        for (int i = 0; i < simH; i++){
            for (int j = 0; j < simW; j++){
                vec3_t ray = {j-(simW/2), fLen, -(i-(simH/2))};                
                normalize(&ray);
                rotate(&ray, camRot.z, camRot.x);
                vec3_t raymarch = camPos;
                for (int k = 0; k < 75; k++){
                    raymarch.x += ray.x*0.1;
                    raymarch.y += ray.y*0.1;
                    raymarch.z += ray.z*0.1;
                    if ((raymarch.x < box.pos.x+box.scale.x && raymarch.x > box.pos.x && raymarch.y < box.pos.y+box.scale.y && raymarch.y > box.pos.y && raymarch.z < box.pos.z+box.scale.z && raymarch.z > box.pos.z) || ((sphere.pos.x-raymarch.x)*(sphere.pos.x-raymarch.x)+(sphere.pos.y-raymarch.y)*(sphere.pos.y-raymarch.y)+(sphere.pos.z-raymarch.z)*(sphere.pos.z-raymarch.z)) < (sphere.radius*sphere.radius)){
                        cellVals[j + (i*simW)] = (1-((float)k/75));//*(1-((float)k/100));
                        break;
                    }
                    else{
                        cellVals[j+(i*simW)] = 0;
                    }
                }
            }
        }
        vec3_t camForward = {0,1,0};
        vec3_t camRight = {1,0,0};
        rotate(&camForward, camRot.z, camRot.x);
        rotate(&camRight, camRot.z, camRot.x);
        //printf("Late:  {%f,%f}\n",camRot.z,camRot.x);        
        //vPrint(camForward);
        box.pos.x = -1+2*sin((float)tick/20);
        //camPos.x += 0.1*(float)(-arrowL+arrowR);
        //camPos.z += 0.1*(float)(arrowD-arrowU);
        camPos.x += 0.1*((arrowU-arrowD)*camForward.x + (arrowR-arrowL)*camRight.x);
        camPos.y += 0.1*((arrowU-arrowD)*camForward.y + (arrowR-arrowL)*camRight.y);
        camPos.z += 0.1*((arrowU-arrowD)*camForward.z + (arrowR-arrowL)*camRight.z);
        
        box.pos.z = -1+2*cos((float)tick/20);
        tick++;
        int* pixels;
        int pitch;
        SDL_LockTexture(texture, NULL, &pixels, &pitch);
        for (int i = 0; i < simW*simH; i++){
            uint8_t col = abs(cellVals[i]*255);
            pixels[i] = (col << 24 | col << 16 | col << 8 | 0xFF);
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

void rotate(vec3_t* v, float yaw, float pitch){
    v->x = v->x*cos(yaw) - v->y*sin(yaw)*cos(pitch) + v->z*sin(yaw)*sin(pitch);
    v->y = v->x*sin(yaw) + v->y*cos(yaw)*cos(pitch) - v->z*sin(pitch)*cos(yaw);
    v->z =                 v->y*sin(pitch)          + v->z*cos(pitch);
}

void vPrint(vec3_t v){
    printf("{%f,%f,%f}\n",v.x,v.y,v.z);   
}