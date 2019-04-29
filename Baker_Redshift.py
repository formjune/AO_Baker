"""
place script, numpy and cv2.pyd into documents/maya/scripts.
run python code: import Baker_Redshift;BakerRedshift.ui();
or simply execute via script editor

edit next values according your needs
material_dir - root folder containing all material like $material_$channel.png. no recursive search
redshift_dir - scan directory where redshift saves files. Check in redshift -> baking -> bake
Will be cleared after every bake. by default it's project/images

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
import maya.api.OpenMaya as om2


material_dir = "D:/textures/materials"   # location with material textures
textures_dir = "D:/textures/textures"    # location with AO, ID etc textures
redshift_dir = os.path.join(pm.workspace(fn=True), "images")
size = 1024
materials = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)
             }
multiply_channels = "BaseColor",


# not for edit
pm.loadPlugin("redshift4maya.mll")
material_node = collections.namedtuple("material", "file name channel")


def loadMaterials():
    mt_nodes = []
    for file_name in os.listdir(material_dir):
        name, channel = os.path.splitext(file_name)[0].rsplit("_", 1)
        file_name = os.path.join(material_dir, file_name)
        mt_nodes.append(material_node(file_name, name, channel))

    return mt_nodes


def settings(dirname):
    """setup vray before baking"""

    pm.optionVar(intValue=("redshiftBakeDefaultsHeight", size))
    pm.optionVar(intValue=("redshiftBakeDefaultsWidth", size))
    pm.optionVar(intValue=("redshiftBakeDefaultsTileMode", 1))
    pm.optionVar(stringValue=("redshiftBakeDefaultsUvSet", "map1"))
    pm.optionVar(intValue=("redshiftBakeMode", 2))

    # ao material
    if not pm.objExists("ao_material_rs"):
        material = pm.createNode("RedshiftMaterial", n="ao_material_rs")
        material.setAttr("emission_weight", 1)
        texture = pm.createNode("RedshiftAmbientOcclusion", n="ao_texture_rs")
        texture.connectAttr("outColor", material.attr("diffuse_color"))
        texture.connectAttr("outColor", material.attr("emission_color"))

    else:
        texture = nt.DependNode("ao_texture_rs")
    texture.setAttr("spread", pm.floatSliderGrp("baker_radius", q=True, v=True))
    texture.setAttr("fallOff", pm.floatSliderGrp("baker_falloff", q=True, v=True))
    texture.setAttr("maxDistance", pm.floatSliderGrp("baker_sub", q=True, v=True))

    # shadows catch material
    if not pm.objExists("shadow_material_rs"):
        material = pm.createNode("RedshiftMaterial", n="shadow_material_rs")
        material.setAttr("diffuse_color", (1, 1, 1))


def bakeID(mesh, obj_name):
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
    pm.select(mesh)
    for rs_bake in os.listdir(redshift_dir):
        os.remove(os.path.join(redshift_dir, rs_bake))

    pm.rsRender(bake=True)
    old_name = os.path.join(redshift_dir, os.listdir(redshift_dir)[0])
    image = om2.MImage()
    image.readFromFile(old_name)
    image.writeToFile(os.path.join(textures_dir, mesh_name), "png")


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

    # change build preferences
    global size, material_dir, textures_dir, redshift_dir
    size = pm.intSliderGrp("baker_size", q=True, v=True)
    material_dir = pm.textField("baker_mat_dir", q=True, tx=True)
    textures_dir = pm.textField("baker_out_dir", tx=True, q=True)
    redshift_dir = pm.textField("baker_red_dir", tx=True, q=True)

    # setup directory and render
    if not os.path.exists(textures_dir):
        os.makedirs(textures_dir)
    os.chdir(textures_dir)
    settings(textures_dir)

    selected = pm.selected(type="transform")
    names = [mesh.name() for mesh in selected]
    pm.showHidden(selected)

    if pm.checkBox("baker_id", q=True, v=True):
        for mesh, mesh_name in zip(selected, names):
            bakeID(mesh, mesh_name + "_id")

    if pm.checkBox("baker_ao", q=True, v=True):
        pm.select(pm.ls(type="transform"))
        pm.hyperShade(a="ao_material_rs")
        for mesh, mesh_name in zip(selected, names):
            bake(mesh, mesh_name + "_ao.png")

    if pm.checkBox("baker_shadow", q=True, v=True):
        pm.select(pm.ls(type="transform"))
        pm.hyperShade(a="shadow_material_rs")
        for mesh, mesh_name in zip(selected, names):
            bake(mesh, mesh_name + "_shadow.png")

    if pm.checkBox("baker_mat", q=True, v=True):
        for mesh_name in names:
            bakeMaterials(mesh_name)


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

    win = pm.window("Baker", wh=(200, 400), tlb=True, t="redshift baker")
    pm.columnLayout()
    pm.text("material directory", w=200)
    pm.textField("baker_mat_dir", tx=material_dir, w=200)

    pm.text("output directory", w=200)
    pm.textField("baker_out_dir", tx=textures_dir, w=200)

    pm.text("redshift directory (look in bake prefs)", w=200)
    pm.textField("baker_red_dir", tx=redshift_dir, w=200)

    pm.intSliderGrp("baker_size", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="size", field=True, v=size,
                    max=8192)
    pm.button("baker_uv", l="uv", w=200, c=applyUV)
    pm.text(l="AO", w=200)
    pm.floatSliderGrp("baker_radius", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="spread", field=True, v=.8)
    pm.floatSliderGrp("baker_falloff", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="falloff", field=True, v=1)
    pm.floatSliderGrp("baker_sub", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="max dist", field=True, v=0)
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

    win.show()

ui()