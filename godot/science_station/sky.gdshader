shader_type sky;
void sky(){
 float h=clamp(EYEDIR.y,0.,1.);
 // Clear azure at the zenith, a pale blue band only close to the horizon.
 COLOR=mix(vec3(.12,.37,.64),vec3(.022,.17,.46),pow(h,.32));
}
