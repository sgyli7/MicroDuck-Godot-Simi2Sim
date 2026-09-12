extends RefCounted
## Close and middle-distance architectural fittings, built in the same batches.

func sample_room(w:Node3D,p:Vector3) -> void:
	# Dome panel meridians follow the actual cap, with a small paint offset.
	for i in range(12):
		var az:float=i*TAU/12.
		for j in range(10):
			var a:float=.11+j*(PI*.5-.11)/10.;var b:float=.11+(j+1)*(PI*.5-.11)/10.
			w._line(p+Vector3(sin(az)*sin(a)*1.66,1.1+cos(a)*.788,cos(az)*sin(a)*1.66),
				p+Vector3(sin(az)*sin(b)*1.66,1.1+cos(b)*.788,cos(az)*sin(b)*1.66),.0025,"strata")
	for x in [-.62,.62]:w._box(p+Vector3(x,.53,1.40),Vector3(.07,1.06,.17),"blue",.012)
	w._box(p+Vector3(0,1.10,1.40),Vector3(1.33,.09,.18),"blue",.012)
	w._label("SAMPLE / 04",p+Vector3(0,1.10,1.499),40,.0013,"paper")
	# Rear-wall storage remains visible through the open entrance.
	w._box(p+Vector3(0,.29,-1.15),Vector3(1.20,.58,.35),"graphite",.025)
	for x in [-.31,.31]:
		for y in [.16,.39]:
			w._box(p+Vector3(x,y,-.955),Vector3(.55,.19,.045),"purple",.012)
			w._box(p+Vector3(x,y+.015,-.924),Vector3(.16,.021,.02),"paper",.004)
	w._box(p+Vector3(0,.60,-1.15),Vector3(1.30,.04,.44),"metal",.015)
	for i in range(3):
		w._cylinder(p+Vector3(-.35+i*.24,.685,-1.13),.058,.13,"paper")
		w._cylinder(p+Vector3(-.35+i*.24,.76,-1.13),.06,.024,"blue")
	# A small illuminated measuring station under the awning.
	w._box(p+Vector3(-.69,.57,1.68),Vector3(.25,.24,.18),"paper",.018)
	w._box(p+Vector3(-.69,.61,1.779),Vector3(.18,.10,.025),"graphite",.010)
	for x in [-.75,-.65]:w._cylinder(p+Vector3(x,.51,1.79),.012,.02,"yellow",Vector3.FORWARD)
	w._line(p+Vector3(-1.25,.45,1.54),p+Vector3(-1.35,.95,1.54),.012,"metal")
	w._line(p+Vector3(-1.35,.95,1.54),p+Vector3(-1.02,.99,1.77),.012,"metal")
	w._cylinder(p+Vector3(-1.02,.98,1.77),.065,.035,"light")
	# Roof vents and a paired pressure conduit finish the side elevation.
	for y in [.48,.70]:
		w._line(p+Vector3(-1.57,y,-.25),p+Vector3(-1.61,y,.44),.025,"metal")
		w._line(p+Vector3(-1.61,y,.44),p+Vector3(-1.45,y,.71),.025,"metal")
	w._box(p+Vector3(-1.58,.68,-.10),Vector3(.09,.50,.36),"paper",.025)
	for i in range(5):w._box(p+Vector3(-1.637,.52+i*.08,-.1),Vector3(.018,.022,.27),"graphite",.003)

	# Side-wall hatches and two visor windows break up the large enclosure.
	# The door-facing three panels remain open, as in the collider layout.
	for i in [3,5,9,14,18]:
		var a:float=i*TAU/24.
		var q:=p+Vector3(sin(a)*1.587,.65,cos(a)*1.587)
		w._box(q,Vector3(.27,.46,.035),"graphite",.042,a)
		w._box(q+Vector3(sin(a)*.023,0,cos(a)*.023),Vector3(.22,.40,.016),"blue" if i in [3,5] else "paper",.028,a)
		w._box(q+Vector3(0,.255,0),Vector3(.34,.035,.18),"paper",.009,a)
		if i not in [3,5]:
			w._box(q+Vector3(sin(a)*.037,-.12,cos(a)*.037),Vector3(.09,.018,.016),"metal",.003,a)
	for i in range(2,23):
		var a:float=i*TAU/24.;var b:float=(i+1)*TAU/24.
		w._line(p+Vector3(sin(a)*1.585,.26,cos(a)*1.585),p+Vector3(sin(b)*1.585,.26,cos(b)*1.585),.009,"metal")

func relay(w:Node3D,p:Vector3) -> void:
	# Same small building, but all visible elevations describe a useful instrument.
	for x in [-.45,.1]:
		w._box(p+Vector3(x,.88,.611),Vector3(.40,.35,.04),"graphite",.035)
		w._box(p+Vector3(x,.90,.638),Vector3(.32,.25,.018),"blue",.022)
		w._box(p+Vector3(x,1.08,.68),Vector3(.48,.035,.22),"paper",.008)
	w._box(p+Vector3(.55,.60,.61),Vector3(.19,.65,.04),"paper",.012)
	w._box(p+Vector3(.56,.60,.638),Vector3(.025,.11,.016),"graphite",.003)
	for y in [.20,.28,.36]:w._box(p+Vector3(-.45,y,.62),Vector3(.31,.027,.03),"metal",.005)
	for z in [-.33,.30]:
		w._cylinder(p+Vector3(.80,.63,z),.06,.82,"metal")
		w._cylinder(p+Vector3(.80,.91,z),.09,.08,"paper")
	w._box(p+Vector3(-.76,.48,0),Vector3(.20,.58,.65),"blue",.025)
	for i in range(4):w._box(p+Vector3(-.87,.3+i*.11,0),Vector3(.035,.04,.46),"graphite",.008)
	w._line(p+Vector3(.42,1.36,-.3),p+Vector3(.42,2.50,-.3),.012,"graphite")
	w._label("FIELD RELAY",p+Vector3(-.1,.49,.63),36,.0012,"blue")
