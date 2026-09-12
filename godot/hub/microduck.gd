extends "res://standalone/driver.gd"
var hub: Node3D

func _setup_play_ui() -> void:
	pass

func _follow_camera(_delta: float) -> void:
	pass

func _decide(held: Array, taps: Array, order: Array, elapsed: float) -> bool:
	if taps.has("switch_robot"):
		hub.select_robot("roller" if session.mode == "walk" else "sai")
		return false
	return super._decide(held,taps,order,elapsed)

func _physics_process(delta: float) -> void:
	if _headless: _sample_held()
	super._physics_process(delta)

func _handle(command: Variant) -> void:
	if command is Dictionary and command.get("cmd","") == "step" and command.has("place_ball"):
		if hub.atelier.loose_props.selected > 0:
			var point: Array = command.place_ball
			hub.atelier.loose_props.place_target(_m2g(Vector3(point[0],point[1],point[2])))
			command = command.duplicate()
			command.erase("place_ball")
	super._handle(command)
