from pxr import UsdGeom, Gf, Vt, Sdf
import omni.usd

INSTANCER_ROOT = "/World/NBodySim"
INSTANCER_PATH = f"{INSTANCER_ROOT}/Instancer"
PROTO_PATH     = f"{INSTANCER_PATH}/Protos/Body"

NEURAL_INSTANCER_ROOT = "/World/NeuralParticles"
NEURAL_INSTANCER_PATH = f"{NEURAL_INSTANCER_ROOT}/Instancer"
NEURAL_PROTO_PATH     = f"{NEURAL_INSTANCER_PATH}/Protos/Body"


def create_instancer(n: int) -> UsdGeom.PointInstancer:
    stage = omni.usd.get_context().get_stage()

    UsdGeom.Xform.Define(stage, INSTANCER_ROOT)
    instancer = UsdGeom.PointInstancer.Define(stage, INSTANCER_PATH)

    stage.DefinePrim(f"{INSTANCER_PATH}/Protos", "Scope")
    UsdGeom.Sphere.Define(stage, PROTO_PATH).GetRadiusAttr().Set(1.0)

    instancer.GetPrototypesRel().AddTarget(PROTO_PATH)
    instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * n))
    instancer.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0)] * n))
    instancer.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1, 1, 1)] * n))

    color_pv = UsdGeom.PrimvarsAPI(instancer.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.varying,
    )
    color_pv.Set(Vt.Vec3fArray([Gf.Vec3f(0.1, 0.1, 1.0)] * n))

    return instancer


def create_neural_instancer(n: int, prim_path: str = NEURAL_INSTANCER_ROOT) -> UsdGeom.PointInstancer:
    stage = omni.usd.get_context().get_stage()

    instancer_path = f"{prim_path}/Instancer"
    proto_path = f"{instancer_path}/Protos/Body"

    UsdGeom.Xform.Define(stage, prim_path)
    instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)

    stage.DefinePrim(f"{instancer_path}/Protos", "Scope")
    UsdGeom.Sphere.Define(stage, proto_path).GetRadiusAttr().Set(1.0)

    instancer.GetPrototypesRel().AddTarget(proto_path)
    instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * n))
    instancer.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0)] * n))
    instancer.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1, 1, 1)] * n))

    color_pv = UsdGeom.PrimvarsAPI(instancer.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.varying,
    )
    color_pv.Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.5, 0.1)] * n))

    return instancer


def destroy_instancer() -> None:
    stage = omni.usd.get_context().get_stage()
    prim  = stage.GetPrimAtPath(INSTANCER_ROOT)
    if prim.IsValid():
        stage.RemovePrim(INSTANCER_ROOT)


def destroy_neural_instancer() -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(NEURAL_INSTANCER_ROOT)
    if prim.IsValid():
        stage.RemovePrim(NEURAL_INSTANCER_ROOT)
