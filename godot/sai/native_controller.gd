extends RefCounted
## Sai locomotion controller executed entirely inside Godot.
##
## The numerical contract is a direct port of sai_agent.control and
## sai_agent.godot_controller.  Inputs stay in Sai's public right-handed
## X-forward/Y-left/Z-up frame; state is sampled from Jolt by robot.gd.

const CONTROL_DT := 0.02
const STAND_HEIGHT := 0.2192
const CROUCH_DROP := 0.035
const SIDES := [1.0,-1.0,1.0,-1.0]
const FRONTS := [1.0,1.0,-1.0,-1.0]
const LEG_INDICES := [0,1,2,4,5,6,8,9,10,12,13,14]
const WHEEL_INDICES := [3,7,11,15]
const DEFAULT_STAIRS := {
	"speed":0.12,"lift_height":0.055,"leg_scale":0.18,"min_crouch":0.0,
	"route_center_y":null,"motion_phase_start_seconds":null,
	"yaw_correction_limit":0.4,
}

var flat_policy
var stair_policy
var flat_policy_id := "sai-flat-motion-v1"
var flat_policy_sha256 := "094adb4484b3d812dfb8a056491beda34a23fd0fa6f463b7784854ebc354d53d"
var stair_profile_id := "stairs-dev40"
var stair_settings := DEFAULT_STAIRS.duplicate(true)
var experimental_profile := false
var previous := PackedFloat32Array()
var crouch := 0.0
var desired_heading: Variant = null
var motion_origin: Variant = null
var last_error := ""
var stance_impedance
var terrain_suspension

func _init(profile_path := "") -> void:
	previous.resize(16)
	flat_policy = _load_policy("res://sai_policy/flat-motion-v1.onnx",flat_policy_sha256)
	if flat_policy==null:return
	var suspension_profile:Dictionary=JSON.parse_string(FileAccess.get_file_as_string("res://sai_policy/suspension-v2.json"))
	if suspension_profile.get("flat_policy_sha256","")!=flat_policy_sha256:
		_fail("Sai suspension profile does not match flat policy")
		return
	stance_impedance=preload("res://sai/stance_impedance.gd").new(suspension_profile)
	terrain_suspension=preload("res://sai/terrain_suspension.gd").new(suspension_profile)
	var stairs_path := "res://sai_policy/stairs-dev40.onnx"
	var stairs_hash := "0eca6930e7ccd3201023ce9dd0ce4b9cb0476a9490020da89802cac598b2e38d"
	if not str(profile_path).is_empty():
		var profile: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(profile_path))
		if profile.get("schema_version") != 1:
			_fail("Unsupported experimental stair profile schema")
		for key in profile.get("control",{}):
			if not stair_settings.has(key): _fail("Unknown experimental stair control setting: "+str(key))
		stair_settings.merge(profile.get("control",{}),true)
		stairs_path = str(profile_path).get_base_dir().path_join(str(profile.actor))
		stairs_hash = str(profile.onnx_sha256)
		stair_profile_id = str(profile.id)
		experimental_profile = true
	_validate_settings()
	if not last_error.is_empty():return
	stair_policy = _load_policy(stairs_path,stairs_hash)

func _fail(message: String) -> void:
	last_error = message
	push_error(message)

func _load_policy(path: String, expected_hash: String):
	if not ClassDB.class_exists("MicroDuckPolicy"):
		_fail("Native ONNX policy extension is unavailable")
		return null
	if not FileAccess.file_exists(path):
		_fail("Sai policy is missing: "+path)
		return null
	var raw := FileAccess.get_file_as_bytes(path)
	var hash := HashingContext.new()
	hash.start(HashingContext.HASH_SHA256)
	hash.update(raw)
	if hash.finish().hex_encode() != expected_hash:
		_fail("Sai policy hash mismatch: "+path)
		return null
	var policy = ClassDB.instantiate("MicroDuckPolicy")
	if not policy.load_model(raw):
		_fail("Sai policy load failed: "+str(policy.get_last_error()))
		return null
	return policy

func _validate_settings() -> void:
	for item in [["speed",0.02,0.16],["lift_height",0.0,0.1],["leg_scale",0.0,0.6],["min_crouch",0.0,1.0]]:
		var value := float(stair_settings[item[0]])
		if value < item[1] or value > item[2]: _fail("Experimental stair setting out of range: "+item[0])
	var route = stair_settings.route_center_y
	if route != null and not is_finite(float(route)): _fail("Non-finite route center")
	var start = stair_settings.motion_phase_start_seconds
	if start != null and (float(start)<0.0 or float(start)>=3.2): _fail("Invalid motion phase start")
	var correction := float(stair_settings.yaw_correction_limit)
	if correction <= 0.0 or correction > 1.2: _fail("Invalid heading correction limit")

func reset() -> void:
	previous.fill(0.0)
	crouch = 0.0
	desired_heading = null
	motion_origin = null
	if stance_impedance!=null:stance_impedance.reset()
	if terrain_suspension!=null:terrain_suspension.reset()

func command(state: Dictionary) -> Dictionary:
	if state.get("robot_id") != "Sai_Agent_001" or state.get("physics_owner") != "Godot/Jolt":
		_fail("Unexpected robot or physics owner")
		return {}
	if state.q.size()<16 or state.v.size()<16 or state.command.size()!=3 or state.terrain_heights.size()!=24:
		_fail("Invalid articulated Sai state")
		return {}
	var requested: Array = [float(state.command[0]),float(state.command[1]),float(state.command[2])]
	var relevant := _step_in_wheel_path(state,requested[2]>.5)
	var descending: bool = bool(state.get("contact_following_descent",false)) and _descending_in_path(state) and requested[0]>.015
	if descending: requested[0]=minf(.16,requested[0])
	var crouch_blocked: bool = requested[2]>.5 and requested[0]>.015 and _span(state.terrain_heights)>.008 and _step_in_wheel_path(state,true,false)
	if crouch_blocked: requested[0]=0.0
	var stairs: bool = not descending and relevant and stair_policy!=null and _span(state.terrain_heights)>.004 and requested[0]>.015
	if stairs: requested[0]=minf(requested[0],float(stair_settings.speed))
	if requested[0]<=.015: motion_origin=null
	elif motion_origin==null: motion_origin=float(state.time)
	var phase_time := float(state.time)
	if stairs and stair_settings.motion_phase_start_seconds!=null:
		phase_time=phase_time-float(motion_origin)+float(stair_settings.motion_phase_start_seconds)
	var requested_crouch := maxf(requested[2],float(stair_settings.min_crouch))
	crouch += clampf(requested_crouch-crouch,-.04,.04)
	var observation := observation(state,requested,crouch,previous,phase_time*TAU/(3.2 if stairs else 2.4))
	var selected = stair_policy if stairs else flat_policy
	var inferred: PackedFloat32Array=selected.infer(observation)
	if inferred.size()!=16:
		_fail("Native Sai inference failed: "+str(selected.get_last_error()))
		return {}
	var action: PackedFloat32Array = filter_action(inferred,requested)
	var target := targets_stairs(action,requested,crouch,phase_time/3.2,state.terrain_heights,
		float(stair_settings.lift_height),float(stair_settings.leg_scale)) if stairs else targets(action,requested,crouch)
	var rotation: Array = state.base_rotation_columns
	var yaw := atan2(float(rotation[0][1]),float(rotation[0][0]))
	var yaw_rate := _dot3(rotation[2],state.base_angular_world)
	if stairs and stair_settings.route_center_y!=null and absf(requested[1])<1e-5:
		desired_heading=clampf(atan2(float(stair_settings.route_center_y)-float(state.base_position[1]),.5),-.4,.4)
	target=_heading(target,requested,yaw,yaw_rate)
	previous=action.duplicate()
	var stage := "descending" if descending else "crouch_blocked" if crouch_blocked else "stairs" if stairs else "crouched" if crouch>.5 else "rolling"
	var result:={"mode":"transport","stage":stage,"target_leg":target,"wheel_speed":[target[3],target[7],target[11],target[15]],
		"target_arm":[0.0,0.0,0.0,0.0,0.0,0.0],"arm_bias":state.get("arm_gravity_bias",[0.0,0.0,0.0,0.0,0.0,0.0]),
		"grip_cap":1.4,"cargo_target_rad":0.0,"policy_action":Array(action),"policy_observation":Array(observation),
		"physics_advanced_by_controller":false,"effective_crouch":crouch,"stair_profile":stair_profile_id,
		"flat_policy_id":flat_policy_id,"flat_policy_sha256":flat_policy_sha256,
		"terrain_step_relevant":relevant,"controller_backend":"godot-native-onnxruntime",
		"contract_id":"sai-experimental-stairs-v1" if experimental_profile else "sai-flat-v1+heading-v1"}
	if terrain_suspension!=null:result=terrain_suspension.apply(result,state)
	return stance_impedance.apply(result,state) if stance_impedance!=null else result

static func observation(state: Dictionary, input_command: Array, effective_crouch: float,
		last_action: PackedFloat32Array, phase: float) -> PackedFloat32Array:
	var result := PackedFloat32Array()
	result.resize(82)
	var columns: Array = state.base_rotation_columns
	# Rotation row 2, then world linear velocity expressed in the body frame.
	for i in range(3): result[i]=float(columns[i][2])
	for i in range(3): result[3+i]=_dot3(columns[i],state.base_linear_world)
	for i in range(3): result[6+i]=_dot3(columns[i],state.base_angular_world)
	result[9]=input_command[0];result[10]=input_command[1];result[11]=effective_crouch
	for i in range(12):
		result[12+i]=float(state.q[LEG_INDICES[i]])
		result[24+i]=float(state.v[LEG_INDICES[i]])*.1
	for i in range(4): result[36+i]=float(state.v[WHEEL_INDICES[i]])*SIDES[i]*.1
	for i in range(16): result[40+i]=last_action[i]
	result[56]=sin(phase);result[57]=cos(phase)
	var ground := float(state.base_position[2])-STAND_HEIGHT
	for i in range(24): result[58+i]=clampf((float(state.terrain_heights[i])-ground)*5.0,-2.0,2.0)
	return result

static func filter_action(action: PackedFloat32Array, input_command: Array) -> PackedFloat32Array:
	var result := PackedFloat32Array()
	result.resize(16)
	var moving: bool = sqrt(float(input_command[0])*float(input_command[0])+float(input_command[1])*float(input_command[1]))>1e-5
	for i in range(16): result[i]=clampf(action[i],-1.0,1.0) if moving else 0.0
	return result

static func targets(action: PackedFloat32Array, input_command: Array, effective_crouch: float) -> Array:
	var filtered := filter_action(action,input_command)
	var down := .172812737-CROUCH_DROP*effective_crouch
	var result: Array=[]
	result.resize(16)
	var theta0 := atan2(.05,.074833147)
	var beta0 := atan2(.05,.09797959)+theta0
	for leg in range(4):
		var beta: float = -FRONTS[leg]*acos(clampf((down*down-.09*.09-.11*.11)/(2.0*.09*.11),-1.0,1.0))
		var theta := -atan2(.11*sin(beta),.09+.11*cos(beta))
		var base := leg*4
		result[base]=.18*filtered[base]
		result[base+1]=.18*filtered[base+1]+SIDES[leg]*(FRONTS[leg]*theta0-theta)
		result[base+2]=.18*filtered[base+2]+SIDES[leg]*(-FRONTS[leg]*beta0-beta)
		result[base+3]=SIDES[leg]*((input_command[0]-input_command[1]*SIDES[leg]*.146)/.048+6.0*filtered[base+3])
	return result

static func targets_stairs(action: PackedFloat32Array,input_command: Array,effective_crouch: float,
		phase_cycles: float,scan_heights: Array,lift_height:=.055,leg_scale:=.18,stride:=.05) -> Array:
	var filtered := filter_action(action,input_command)
	var result := targets(filtered,input_command,effective_crouch)
	var active: bool = _span(scan_heights)>.004 and input_command[0]>.015
	var theta0 := atan2(.05,.074833147)
	var beta0 := atan2(.05,.09797959)+theta0
	var offsets := [0.0,.5,.75,.25]
	for leg in range(4):
		var phase := fposmod(phase_cycles-offsets[leg],1.0)
		var lift := lift_height*pow(sin(PI*phase/.25),2.0) if active and phase<.25 else 0.0
		var dx := (-stride*.5*cos(PI*phase/.25) if phase<.25 else stride*(.5-(phase-.25)/.75)) if active else 0.0
		var down := .172812737-CROUCH_DROP*effective_crouch-lift
		var beta: float = -FRONTS[leg]*acos(clampf((down*down+dx*dx-.09*.09-.11*.11)/(2.0*.09*.11),-1.0,1.0))
		var theta := atan2(dx,down)-atan2(.11*sin(beta),.09+.11*cos(beta))
		var base := leg*4
		result[base]=clampf(leg_scale*filtered[base],-.45,.45)
		result[base+1]=clampf(SIDES[leg]*(FRONTS[leg]*theta0-theta)+leg_scale*filtered[base+1],-.7,.7)
		result[base+2]=clampf(SIDES[leg]*(-FRONTS[leg]*beta0-beta)+leg_scale*filtered[base+2],-1.2,1.2)
	return result

func _heading(target: Array,input_command: Array,yaw: float,yaw_rate: float) -> Array:
	var parked: bool = sqrt(float(input_command[0])*float(input_command[0])+float(input_command[1])*float(input_command[1]))<1e-5
	if desired_heading==null or parked: desired_heading=yaw
	if parked:return target
	desired_heading=float(desired_heading)+float(input_command[1])*CONTROL_DT
	var error := atan2(sin(float(desired_heading)-yaw),cos(float(desired_heading)-yaw))
	var correction := clampf(1.5*error-.25*(yaw_rate-float(input_command[1])),
		-float(stair_settings.yaw_correction_limit),float(stair_settings.yaw_correction_limit))
	var result := target.duplicate()
	for leg in range(4): result[leg*4+3]-=correction*.146/.048
	return result

static func _step_in_wheel_path(state: Dictionary,crouched:=false,honor_course:=true) -> bool:
	if honor_course and (bool(state.get("stair_course",false)) or bool(state.get("experimental_profile",false))): return true
	var dense = state.get("terrain_edge_heights")
	if dense!=null and dense.size()==138:
		for value in _signed_edges(dense,crouched):
			if absf(value)>.008:return true
		return false
	var path = state.get("terrain_path_heights")
	if path==null:return true
	if path.size()!=15:return false
	var rows := 4 if crouched else 5
	for column in range(3):
		var delta: Array=[]
		for row in range(rows-1):delta.append(float(path[(row+1)*3+column])-float(path[row*3+column]))
		var sorted := delta.duplicate();sorted.sort()
		var median: float = float(sorted[sorted.size()/2]) if sorted.size()%2==1 else .5*(float(sorted[sorted.size()/2-1])+float(sorted[sorted.size()/2]))
		for value in delta:
			if absf(value-median)>.008:return true
	return false

static func _descending_in_path(state: Dictionary) -> bool:
	var dense=state.get("terrain_edge_heights")
	if dense==null or dense.size()!=138:return false
	var negative:=false
	for edge in _signed_edges(dense,false):
		if edge> .008:return false
		if edge<-.045:return false
		if edge<-.008:negative=true
	return negative

static func _signed_edges(dense: Array,crouched: bool) -> Array:
	var result: Array=[]
	for i in range(45):
		var x := -.30+.02*i
		if x<-.20 or x>=(.36 if crouched else .54):continue
		for column in range(3):
			var delta := float(dense[(i+1)*3+column])-float(dense[i*3+column])
			var neighbours: Array=[]
			for offset in [-2,-1,1,2]:
				var j:int=clampi(i+offset,0,44)
				neighbours.append(float(dense[(j+1)*3+column])-float(dense[j*3+column]))
			neighbours.sort()
			var background: float = .5*(float(neighbours[1])+float(neighbours[2]))
			result.append(delta-background)
	return result

static func _span(values: Array) -> float:
	var low:=INF;var high:=-INF
	for value in values:low=minf(low,float(value));high=maxf(high,float(value))
	return high-low

static func _dot3(a,b) -> float:
	return float(a[0])*float(b[0])+float(a[1])*float(b[1])+float(a[2])*float(b[2])
