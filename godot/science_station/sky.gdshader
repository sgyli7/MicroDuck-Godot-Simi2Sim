shader_type sky;
void sky(){
 float h=clamp(EYEDIR.y,0.,1.);
 COLOR=mix(vec3(.12,.32,.58),vec3(.012,.105,.35),pow(h,.22));
}
