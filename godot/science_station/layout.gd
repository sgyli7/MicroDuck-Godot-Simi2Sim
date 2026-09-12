extends RefCounted
## Metres in Godot's Y-up frame. These anchors also define safe reset pads.
const TITLE := "风口科学站"
const CHECKPOINTS := {
	"service": {"title":"服务小院", "position":Vector3(0,0,7)},
	"samples": {"title":"样本处理站", "position":Vector3(-5,0,1)},
	"berth": {"title":"设备泊位", "position":Vector3(5,0,5)}
}
const PROPS := [Vector2(.30,7),Vector2(.66,7.08),Vector2(-4.70,1),Vector2(-4.34,1.08),Vector2(5.30,5),Vector2(5.66,5.08)]
const BERTH := Rect2(3,-8,20,12)
const VIEWS := {
	"arrival":[Vector3(-1.0,.55,8.3),Vector3(5,2,-11),57.],
	"overview":[Vector3(-29,23,32),Vector3(0,1,-5),52.],
	"towers":[Vector3(10.5,2.,-7.5),Vector3(10.8,3.5,-18.5),64.],
	"samples":[Vector3(-9,1.3,5),Vector3(-6,.65,-1.7),53.],
	"berth":[Vector3(23,6,10),Vector3(8,2,-12),55.],
	"hills":[Vector3(-16,1,-1),Vector3(-34,3,-22),62.]
}

static func height_at(x: float,z: float) -> float:
	# Barycentric interpolation of the exact half-metre physics triangles.
	var x0:float=floorf(x*2.)*.5;var z0:float=floorf(z*2.)*.5
	var u:float=(x-x0)*2.;var v:float=(z-z0)*2.
	var b:float=_vertex_height(x0+.5,z0);var c:float=_vertex_height(x0,z0+.5)
	if u+v<=1.:return _vertex_height(x0,z0)*(1.-u-v)+b*u+c*v
	return b*(1.-v)+c*(1.-u)+_vertex_height(x0+.5,z0+.5)*(u+v-1.)

static func _vertex_height(x:float,z:float) -> float:
	# A 12 m wide rise keeps the accepted walking actor within its downhill
	# balance margin (about 2 degrees maximum grade), retaining the 14 cm crest.
	# All three interaction pads stay level.
	var dx: float = (x+18.)/6.
	var dz: float = (z+3.)/7.
	if absf(dx)>=1. or absf(dz)>=1.: return 0.
	return .14*pow(1.-dx*dx,2.)*pow(1.-dz*dz,2.)
