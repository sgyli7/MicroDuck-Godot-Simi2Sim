shader_type spatial;
render_mode specular_disabled;
varying vec3 world;
void vertex(){ world=(MODEL_MATRIX*vec4(VERTEX,1.)).xyz; }
float box(vec2 p,vec2 c,vec2 h){vec2 d=abs(p-c)-h;return 1.-smoothstep(-.008,.008,max(d.x,d.y));}
float line(float d,float width){return 1.-smoothstep(width,width+max(fwidth(d),.002),abs(d));}
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float segment(vec2 p,vec2 a,vec2 b){vec2 d=b-a;return length(p-a-d*clamp(dot(p-a,d)/dot(d,d),0.,1.));}
void fragment(){
 vec2 p=world.xz;
 vec3 sand=vec3(.56,.39,.25), slab=vec3(.79,.80,.74), blue=vec3(.09,.31,.46);
 float pads=max(box(p,vec2(0.,7.),vec2(3.,2.6)),box(p,vec2(-6.,.1),vec2(3.6,3.3)));
 // The operations court belongs to an engineered station, with sand around it.
 float court=box(p,vec2(-3.5,-16.4),vec2(9.8,7.4));
 pads=max(pads,court);
 float apron=1.-smoothstep(3.8,4.0,length((p-vec2(-7.5,-.5))*vec2(1.,1.1)));
 pads=max(pads,apron);
 float berth=box(p,vec2(13.,-2.),vec2(10.,6.));
 float lane=line(p.x+0.7*sin(p.y*.19),1.8);
 float branch=min(segment(p,vec2(0.,7.),vec2(-2.,3.8)),segment(p,vec2(-2.,3.8),vec2(-5.,1.)));
 branch=min(branch,segment(p,vec2(-5.,1.),vec2(5.,5.)));
 branch=min(branch,segment(p,vec2(5.,5.),vec2(0.,7.)));
 float footpath=1.-smoothstep(.46,.52,branch);
 float trail=line(p.y+3.+.15*sin(p.x*.45),.55)*smoothstep(-23.,-21.8,p.x)*(1.-smoothstep(-12.,-10.,p.x));
 vec3 color=mix(sand,slab,max(max(max(pads,berth*.8),lane*.7),max(footpath*.50,trail*.24)));
 float seam=max(line(mod(p.x+1.,2.)-1.,.009),line(mod(p.y+1.,2.)-1.,.009));
 color=mix(color,vec3(.47,.48,.44),seam*max(pads,berth)*.38);
 float guide=max(line(p.x+.7*sin(p.y*.19)-.94,.045),line(p.x+.7*sin(p.y*.19)+.94,.045));
 guide*=smoothstep(-25.,-23.,p.y)*(1.-smoothstep(10.,12.,p.y));
 float dotted=(1.-smoothstep(.012,.012+max(fwidth(branch),.003),branch))*step(.57,fract((p.x+p.y*.47)*2.7));
 float rim=box(p,vec2(13.,-2.),vec2(9.6,5.6))-box(p,vec2(13.,-2.),vec2(9.54,5.54));
 color=mix(color,blue,max(max(guide,rim),dotted*.70));
 // Flush service trenches and inspection covers; no collision-height mismatch.
 float trench=line(p.x+11.35,.12)*box(p,vec2(-11.35,-12.),vec2(.3,8.));
 float slots=step(.45,fract(p.y*5.));
 color=mix(color,vec3(.21,.25,.28),trench*(.55+.45*slots));
 float cover=box(p,vec2(-3.,-12.),vec2(.7,.43));
 color=mix(color,vec3(.34,.40,.41),cover);
 float edge=box(p,vec2(-3.,-12.),vec2(.73,.46))-cover;
 color=mix(color,blue,edge);
 // Sparse wind-scoured strokes; the large sand and concrete fields stay calm.
 vec2 cell=floor(p*.65), local=fract(p*.65)-.5;
 float seed=hash(cell);
 float stroke=line(local.y+.07*sin(local.x*9.+seed*4.),.0015);
 stroke*=1.-smoothstep(.08+seed*.25,.11+seed*.25,abs(local.x));
 stroke*=step(.63,seed)*(1.-max(max(pads,berth),lane));
 color=mix(color,vec3(.23,.24,.23),stroke*.32);
 // Short chips on selected slab joints, rather than a repeated grunge texture.
 float chip=line(mod(p.x+1.,2.)-1.+.025*sin(p.y*15.),.016);
 chip*=step(.97,hash(floor(p*vec2(.5,7.))))*max(pads,berth);
 color=mix(color,vec3(.42,.41,.34),chip*.45);
 float grain=fract(sin(dot(floor(p*65.),vec2(12.9898,78.233)))*43758.5453);
 float fade=1.-smoothstep(.1,.5,length(fwidth(p*65.)));
 color*=1.-.035*step(.94,grain)*fade;
 ALBEDO=color; ROUGHNESS=1.;
}
void light(){
 float n=dot(NORMAL,LIGHT);
 float band=.28+.58*smoothstep(.10,.18,n);
 DIFFUSE_LIGHT+=LIGHT_COLOR*band*ATTENUATION/3.14159265;
}
