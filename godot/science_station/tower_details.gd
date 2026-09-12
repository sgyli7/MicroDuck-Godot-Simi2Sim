extends RefCounted
## Purposeful asymmetry: weather intake A and optical instrument B.
func build(w:Node3D,p:Vector3,h:float,r:float,id:String) -> void:
	# Suspended plumbing stays above the usable ground-level passage.
	for x in [-.31,.31]:
		for z in [-.24,.24]:
			w._cylinder(p+Vector3(x,1.25,z),.095,.53,"metal")
			w._cylinder(p+Vector3(x,1.03,z),.112,.06,"yellow")
			w._line(p+Vector3(x,1.0,z),p+Vector3(x*.45,.88,z),.026,"graphite")
	w._cylinder(p+Vector3(0,1.36,0),.30,.39,"graphite")
	# Cable trays follow one rear leg, away from the two cross passages.
	var b:=p+Vector3(-r*.68,1.90,-r*.68)
	w._box(b,Vector3(.16,.22,.14),"metal",.02)
	for shift in [-.035,.035]:
		var last:=b+Vector3(shift,0,0)
		for q in [b+Vector3(shift,-.30,-.16),b+Vector3(shift-.2,-.66,-.34),b+Vector3(shift-.33,-1.35,-.42)]:
			w._line(last,q,.017,"purple");last=q
	# Hatches and seams form a few clusters with large calm areas between them.
	for spec in [[-.42,2.7,.43,.72],[.55,3.3,.25,.48],[-.74,h*.69,.32,.82]]:
		var a:float=spec[0];var y:float=spec[1]
		var rr:float=lerpf(r,r*.64,clampf((y-1.8)/(h-2.45),0.,1.))+.018
		var q:=p+Vector3(sin(a)*rr,y,cos(a)*rr)
		w._box(q,Vector3(spec[2],spec[3],.035),"metal",.022,a)
		w._box(q+Vector3(sin(a)*.022,0,cos(a)*.022),Vector3(spec[2]-.045,spec[3]-.055,.012),"paper",.016,a)
		for side in [-1,1]:
			w._box(q+Vector3(cos(a)*spec[2]*.29*side,spec[3]*.30,.042),Vector3(.025,.030,.012),"graphite",.003,a)
	# A deep round optical window is readable from a distance.
	var y:float=h*.67
	var rr:float=lerpf(r,r*.64,(y-1.8)/(h-2.45))
	var eye:=p+Vector3(0,y,rr+.035)
	w._cylinder(eye,.34,.075,"metal",Vector3.FORWARD)
	w._cylinder(eye+Vector3(0,0,.048),.29,.035,"blue",Vector3.FORWARD)
	w._cylinder(eye+Vector3(0,0,.068),.20,.012,"graphite",Vector3.FORWARD)
	w._box(eye+Vector3(-.062,.10,.078),Vector3(.15,.035,.012),"paper",.007)
	if id=="02":
		# Wide scanning collar and the recessed drum make the tall tower distinct.
		var crown:=p+Vector3(0,h*.82,0)
		w._frustum(crown,r*.78,r*.93,.18,"paper")
		w._cylinder(crown+Vector3(0,.14,0),r*.76,.20,"graphite")
		w._frustum(crown+Vector3(0,.29,0),r*.94,r*.74,.10,"paper")
		for i in range(20):
			var a:float=i*TAU/20.
			var q:=crown+Vector3(sin(a)*r*.765,.145,cos(a)*r*.765)
			w._box(q,Vector3(.25,.105,.018),"blue" if i%3 else "metal",.005,a)
		# Cantilevered feed arm and an offset mast break axial symmetry.
		w._beam(p+Vector3(r*.61,h-.3,0),p+Vector3(r*1.18,h+.18,0),.055,.09,"metal")
		w._line(p+Vector3(r*1.18,h+.14,0),p+Vector3(r*1.18,h+1.16,0),.015,"graphite")
		w._cylinder(p+Vector3(r*1.18,h+.76,0),.048,.32,"paper")
	else:
		# Weather tower: a rear intake manifold with three protective louvers.
		var q:=p+Vector3(-r*.79,h*.54,r*.47)
		w._box(q,Vector3(.55,.87,.35),"paper",.07,-.7)
		for j in range(4):w._box(q+Vector3(-.15,-.28+j*.17,.17),Vector3(.41,.045,.16),"metal",.009,-.7)
		w._beam(p+Vector3(-r*.63,h-.5,0),p+Vector3(-r*1.3,h-.15,0),.05,.09,"metal")
		for j in range(4):
			var q2:=p+Vector3(-r*1.3,h+.04+j*.14,0)
			w._cylinder(q2,.15,.025,"paper")
		w._line(p+Vector3(-r*1.3,h-.15,0),p+Vector3(-r*1.3,h+.65,0),.020,"graphite")
