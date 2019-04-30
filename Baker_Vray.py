"""
place script, numpy and cv2.pyd into documents/maya/scripts.
run python code: import Baker_Vray;reload(Baker_Vray);Baker_Vray.ui();
or simply execute via script editor

edit next values according your needs
material_dir - root folder containing all material like $material_$channel.png. no recursive search
textures_dir - directory where all textures will be stored (ID, AO, shadows, material)
size - size of output textures, can be changed via gui
materials - dictionary like {$material: (R, G, B)} all colors must be within [0, 255] range.
not registered materials will be skipped.
multiply_channels = list of $channel to be multiplied with AO and shadows. resulting texture are calculated AO * shadows * %channel
"""


import os
import functools
import collections
import numpy as np
import cv2
import pymel.core as pm
import pymel.core.nodetypes as nt


material_dir = "C:/textures/materials"   # location with material textures
textures_dir = "C:/textures/textures"    # location with AO, ID etc textures
size = 1024
materials = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
multiply_channels = "BaseColor",       # channels to multiply with AO and shadows


# not for edit
if not pm.pluginInfo("vrayformaya.mll", q=True, loaded=True):
    pm.loadPlugin("vrayformaya.mll")
material_node = collections.namedtuple("material", "file name channel")


def loadMaterials():
    mt_nodes = []
    for file_name in os.listdir(material_dir):
        name, channel = os.path.splitext(file_name)[0].rsplit("_", 1)
        file_name = os.path.join(material_dir, file_name)
        mt_nodes.append(material_node(file_name, name, channel))

    return mt_nodes


def settings():
    """setup vray before baking"""
    global size
    size = pm.intSliderGrp("baker_size", q=True, v=True)

    pm.optionVar(intValue=("vrayBakeType", 2))
    pm.optionVar(intValue=("vraySkipNodesWithoutBakeOptions", 0))
    pm.optionVar(intValue=("vrayAssignBakedTextures", 0))
    pm.optionVar(stringValue=("vrayBakeOutputPath", textures_dir))
    pm.optionVar(intValue=("vrayBakeType", 2))
    try:
        options = nt.VRayBakeOptions("vrayDefaultBakeOptions")
    except pm.MayaNodeError:
        options = pm.createNode("VRayBakeOptions", n="vrayDefaultBakeOptions")
    options.setAttr("resolutionX", size)
    options.setAttr("outputTexturePath", textures_dir, type="string")
    options.setAttr("filenamePrefix", "")

    # ao material
    if not pm.objExists("ao_material"):
        material = pm.createNode("VRayMtl", n="ao_material")
        texture = pm.createNode("VRayDirt", n="ao_texture")
        texture.connectAttr("outColor", material.attr("color"))
        texture.connectAttr("outColor", material.attr("illumColor"))
    else:
        texture = nt.DependNode("ao_texture")
    texture.setAttr("radius", pm.floatSliderGrp("baker_radius", q=True, v=True))
    texture.setAttr("falloff", pm.floatSliderGrp("baker_falloff", q=True, v=True))
    texture.setAttr("subdivs", pm.intSliderGrp("baker_sub", q=True, v=True))

    # shadows catch material
    if not pm.objExists("shadow_material"):
        material = pm.createNode("VRayMtl", n="shadow_material")
        material.setAttr("color", (1, 1, 1))


def bakeID(mesh, obj_name):
    print mesh
    """bake id maps to 8bit"""
    mesh_2 = pm.duplicate(mesh)[0]
    pm.surfaceSampler(
            fileFormat="png",
            filename=obj_name,
            filterSize=0,
            filterType=2,
            flipU=0,
            flipV=0,
            ignoreMirroredFaces=0,
            mapHeight=size,
            mapWidth=size,
            mapMaterials=1,
            mapOutput="diffuseRGB",
            mapSpace="tangent",
            max=1,
            maxSearchDistance=0,
            overscan=3,
            searchCage="",
            searchMethod=0,
            searchOffset=0,
            shadows=0,
            source=mesh.name(),
            sourceUVSpace="map1",
            superSampling=0,
            target=mesh_2.name(),
            targetUVSpace="map1",
            useGeometryNormals=0,
            uvSet="map1"
        )
    pm.delete(mesh_2)


def bake(mesh, mesh_name):
    """bake 0 - id, 1 - ao, 2 - shadows"""
    old_name = "-%s.png" % mesh.listRelatives(type="mesh")[0].name()

    pm.select(mesh)
    pm.mel.eval("vrayStartBake();")
    image = om2.MImage()
    image.readFromFile(os.path.join(textures_dir, old_name))
    image.writeToFile(os.path.join(textures_dir, mesh_name), "png")
    try:
        os.remove(old_name)
    except:
        pass


def bakeMaterials(mesh_name):
    """create material maps based on id"""
    mat_list = loadMaterials()
    mask_map = cv2.imread(os.path.join(textures_dir, mesh_name + "_id.png"))
    if mask_map is None:
        pm.warning(mesh_name + " id map doesn't exists")
        return
    mask_map = cv2.resize(mask_map, (size, size))

    for channel in {ch.channel for ch in mat_list}:
        channel_map = np.zeros((size, size, 3), dtype=np.uint8)  # create empty channel

        for color in materials:
            found = [mt for mt in mat_list if mt.name == color and mt.channel == channel]
            if not found:
                pm.warning("%s_%s doesn't exists" % (color, channel))
                continue
            image_file = cv2.resize(cv2.imread(found[0].file), (size, size))
            color_mask = findMask(mask_map, materials[color])
            channel_map |= cv2.bitwise_or(image_file, image_file, mask=color_mask)

        # multiply
        if channel in multiply_channels:
            ao_mask = cv2.resize(cv2.imread(mesh_name + "_ao.png"), (size, size))
            shadow_mask = cv2.resize(cv2.imread(mesh_name + "_shadow.png"), (size, size))
            if ao_mask is not None and shadow_mask is not None:
                ao_mask = cv2.cvtColor(cv2.cvtColor(ao_mask, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) / 255.
                shadow_mask = cv2.cvtColor(cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) / 255.
                channel_map = (channel_map * ao_mask * shadow_mask).astype(np.uint8)

        cv2.imwrite(os.path.join(textures_dir, mesh_name + "_mat_" + channel + ".png"), channel_map)


def findMask(image, rgb):
    """convert texture to mask. args (np.array, (r, g, b))"""
    def minMax(color):
        return min(255, max(color, 0))
    top = np.array([minMax(rgb[2] + 4), minMax(rgb[1] + 4), minMax(rgb[0] + 4)], dtype=np.uint8)
    bottom = np.array([minMax(rgb[2] - 4), minMax(rgb[1] - 4), minMax(rgb[0] - 4)], dtype=np.uint8)
    return cv2.inRange(image, bottom, top)


def render(*args):
    global size, material_dir, textures_dir
    size = pm.intSliderGrp("baker_size", q=True, v=True)
    material_dir = pm.textField("baker_mat_dir", q=True, tx=True)
    textures_dir = pm.textField("baker_out_dir", tx=True, q=True)
    redshift_dir = os.path.join(pm.workspace(fn=True), "images")

    os.chdir(textures_dir)
    settings()

    selected = pm.selected(type="transform") or pm.ls(type="transform")
    selected = [sl for sl in selected if sl.listRelatives(type="mesh")]
    names = [mesh.name().split("|")[-1] for mesh in selected]
    pm.showHidden(selected)

    if pm.checkBox("baker_id", q=True, v=True):
        for mesh, mesh_name in zip(selected, names):
            bakeID(mesh, mesh_name + "_id")

    if pm.checkBox("baker_ao", q=True, v=True):
        pm.select(pm.ls(type="mesh"))
        pm.hyperShade(a="ao_material")
        for mesh, mesh_name in zip(selected, names):
            bake(mesh, mesh_name + "_ao.png")

    if pm.checkBox("baker_shadow", q=True, v=True):
        pm.select(pm.ls(type="mesh"))
        pm.hyperShade(a="shadow_material")
        for mesh, mesh_name in zip(selected, names):
            bake(mesh, mesh_name + "_shadow.png")

    if pm.checkBox("baker_mat", q=True, v=True):
        for mesh_name in names:
            bakeMaterials(mesh_name)
    pm.warning("finished")


def createFinal():
    """create final version of material by multiplying all maps"""


def applyMaterial(color, *args):
    """apply id material to selected"""
    try:
        material = nt.Lambert(color + "_id")
    except pm.MayaNodeError:
        selected = pm.selected()
        material = pm.createNode("lambert", n=color + "_id_material")  # vray node
        color = [c / 255. for c in materials[color]]
        material.setAttr("color", color)
        pm.select(selected)
    pm.hyperShade(a=material)


def applyUV(*args):
    pm.polyAutoProjection(lm=0, pb=0, ibd=1, cm=0, l=2, sc=1, o=1, p=6, ps=.2, ws=0)


def ui():
    if pm.window("Baker", ex=True):
        pm.deleteUI("Baker")

    win = pm.window("Baker", wh=(200, 400), tlb=True, t="Vray baker")
    pm.columnLayout()

    pm.text("material directory", w=200)
    pm.textField("baker_mat_dir", tx=material_dir, w=200)

    pm.text("output directory", w=200)
    pm.textField("baker_out_dir", tx=textures_dir, w=200)

    pm.intSliderGrp("baker_size", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="size", field=True, v=size,
                    max=8192)
    pm.button("baker_uv", l="uv", w=200, c=applyUV)
    pm.text(l="AO", w=200)
    pm.floatSliderGrp("baker_radius", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="radius", field=True, v=10)
    pm.floatSliderGrp("baker_falloff", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="falloff", field=True, v=0)
    pm.intSliderGrp("baker_sub", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="subdivs", field=True, v=3)
    pm.text(l="bake", w=200)

    pm.checkBox("baker_id", l="bake id", v=True)
    pm.checkBox("baker_ao", l="bake ao", v=True)
    pm.checkBox("baker_shadow", l="bake shadow", v=True)
    pm.checkBox("baker_mat", l="bake materials", v=True)
    pm.text(h=30, l="")
    pm.button("baker_run", l="bake", w=200, c=render)
    pm.text(l="materials", w=200)
    for color in materials:
        pm.button(l=color, w=200, c=functools.partial(applyMaterial, color))

    try:
        options = nt.VRayBakeOptions("vrayDefaultBakeOptions")
    except pm.MayaNodeError:
        options = pm.createNode("VRayBakeOptions", n="vrayDefaultBakeOptions")
    options.setAttr("resolutionX", size)
    options.setAttr("outputTexturePath", textures_dir, type="string")
    options.setAttr("filenamePrefix", "")

    win.show()

ui()