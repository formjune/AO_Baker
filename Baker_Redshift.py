"""
place script, numpy and cv2.pyd into documents/maya/scripts.
run python code: import Baker_Redshift;reload(Baker_Redshift);Baker_Redshift.ui();
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
import shutil
import string
import maya.cmds as cmds


material_dir = "C:/textures/materials"   # location with material textures
textures_dir = "C:/textures/textures"    # location with AO, ID etc textures
redshift_dir = os.path.join(pm.workspace(fn=True), "images")
name_pattern = "%material_%channel"
size = 1024
materials = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
allowed_symbols = set(string.ascii_letters + string.digits + "_")
multiply_channels = "BaseColor",


# not for edit
if not pm.pluginInfo("redshift4maya.mll", q=True, loaded=True):
    pm.loadPlugin("redshift4maya.mll")
material_node = collections.namedtuple("material", "file name channel")


def findNames():
    """convert names into FileNode[]"""
    def makeNode(full_path):
        name = os.path.split(full_path)[1]
        if not os.path.isfile(full_path):
            return None
        keys = os.path.splitext(name)[0].split("_")
        print keys
        try:
            return material_node(full_path, keys[name_id] if name_id != -1 else "",
                                 keys[channel_id] if channel_id != -1 else "")
        except IndexError:
            return None

    sample = name_pattern
    path = material_dir
    split_sample = os.path.splitext(sample)[0].split("_")
    name_id = split_sample.index("%material") if "%material" in split_sample else -1
    channel_id = split_sample.index("%channel") if "%channel" in split_sample else -1

    if os.path.isdir(path):
        nodes = [makeNode(os.path.join(path, file_name)) for file_name in os.listdir(path)]
        return [node for node in nodes if node is not None]
    else:
        return []


def settings():
    """setup vray before baking"""

    if not pm.objExists("redshiftOptions"):
        pm.createNode("RedshiftOptions", n="redshiftOptions")
    pm.setAttr("redshiftOptions.imageFormat", 2)
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

    if not pm.objExists("shadow_material_rs"):
        material = pm.createNode("RedshiftMaterial", n="shadow_material_rs")
        material.setAttr("diffuse_color", (1, 1, 1))

    # else:
    #    texture = nt.DependNode("ao_texture_rs")
    # texture.setAttr("spread", pm.floatSliderGrp("baker_radius", q=True, v=True))
    # texture.setAttr("fallOff", pm.floatSliderGrp("baker_falloff", q=True, v=True))
    # texture.setAttr("maxDistance", pm.floatSliderGrp("baker_sub", q=True, v=True))

    # shadows catch material


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
    # clean and create new dir
    shutil.rmtree(redshift_dir, ignore_errors=True)
    os.makedirs(redshift_dir)

    pm.select(mesh)
    pm.rsRender(bake=True)
    old_name = os.path.join(redshift_dir, os.listdir(redshift_dir)[0])
    open(mesh_name, "wb").write(open(old_name, "rb").read())


def bakeMaterials(mat_list, mesh_name):
    """create material maps based on id"""

    mask_map = cv2.imread(mesh_name + "_id.png")
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
            ao_mask = cv2.imread(mesh_name + "_ao.png")
            shadow_mask = cv2.imread(mesh_name + "_shadow.png")
            if ao_mask is not None and shadow_mask is not None:
                ao_mask = cv2.resize(ao_mask, (size, size))
                shadow_mask = cv2.resize(shadow_mask, (size, size))
                ao_mask = cv2.cvtColor(cv2.cvtColor(ao_mask, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) / 255.
                shadow_mask = cv2.cvtColor(cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) / 255.
                channel_map = (channel_map * ao_mask * shadow_mask).astype(np.uint8)

        cv2.imwrite("%s_mat_%s.png" % (mesh_name, channel), channel_map)


def findMask(image, rgb):
    """convert texture to mask. args (np.array, (r, g, b))"""
    def minMax(color):
        return min(255, max(color, 0))
    top = np.array([minMax(rgb[2] + 4), minMax(rgb[1] + 4), minMax(rgb[0] + 4)], dtype=np.uint8)
    bottom = np.array([minMax(rgb[2] - 4), minMax(rgb[1] - 4), minMax(rgb[0] - 4)], dtype=np.uint8)
    return cv2.inRange(image, bottom, top)


def render(*args):
    global size, material_dir, textures_dir, redshift_dir, name_pattern
    size = pm.intSliderGrp("baker_size", q=True, v=True)
    material_dir = pm.textField("baker_mat_dir", q=True, tx=True)
    textures_dir = pm.textField("baker_out_dir", tx=True, q=True)
    redshift_dir = os.path.join(pm.workspace(fn=True), "images")
    name_pattern = pm.textField("baker_pattern", tx=True, q=True)

    settings()

    selected = pm.selected(type="transform") or getMeshes()
    all_meshes = getMeshes()
    copy_meshes = [pm.duplicate(x, n=x.name() + "_copy")[0] for x in all_meshes]
    pm.showHidden(copy_meshes)
    pm.hide(all_meshes)
    mat_list = findNames()

    for mesh in selected:
        mesh_name = mesh.name()
        mesh_copy = nt.DependNode(mesh_name + "_copy")
        mesh_full_name = os.path.join(textures_dir, mesh_name, mesh_name)
        mesh_full_dir = os.path.join(textures_dir, mesh_name)

        if not os.path.exists(mesh_full_dir):
            os.makedirs(mesh_full_dir)

        if pm.checkBox("baker_id", q=True, v=True):
            bakeID(mesh, mesh_full_name + "_id")

        if pm.checkBox("baker_ao", v=True, q=True):
            pm.select(copy_meshes)
            pm.hyperShade(a="ao_material_rs")
            bake(mesh_copy, mesh_full_name + "_ao.png")

        if pm.checkBox("baker_shadow", v=True, q=True):
            pm.select(copy_meshes)
            pm.hyperShade(a="shadow_material_rs")
            bake(mesh_copy, mesh_full_name + "_shadow.png")

        if pm.checkBox("baker_mat", q=True, v=True):
            bakeMaterials(mat_list, mesh_full_name)

        if pm.checkBox("baker_mesh", q=True, v=True):
            pm.select(mesh_copy)
            cmds.file(mesh_full_name + ".obj", force=True, type='OBJexport', es=True,
                      options='groups=1;ptgroups=1;materials=0;smoothing=1;normals=1')

    pm.showHidden(all_meshes)
    pm.delete(copy_meshes)


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


def multiImport(*args):
    for f in pm.fileDialog2(ff="*.obj", fm=4):
        cmds.file(f, i=True)


def getMeshes():
    return [x for x in pm.ls(type="transform") if x.listRelatives(type="mesh")]


def checkNames(*args):

    for mesh in getMeshes():
        for letter in mesh.name():
            try:
                assert letter in allowed_symbols
            except AssertionError:
                pm.warning("incorrect name: %s, character: '%s'" % (mesh.name(), letter))


def preferences(*args):
    pm.select(args)
    pm.mel.eval("AttributeEditor;")


def ui():

    settings()

    if pm.window("Baker", ex=True):
        pm.deleteUI("Baker")

    win = pm.window("Baker", wh=(200, 400), tlb=True, t="redshift baker")
    pm.columnLayout()

    pm.text("material name pattern", w=200)
    pm.textField("baker_pattern", tx=name_pattern, w=200)

    pm.text("material directory", w=200)
    pm.textField("baker_mat_dir", tx=material_dir, w=200)

    pm.text("output directory", w=200)
    pm.textField("baker_out_dir", tx=textures_dir, w=200)

    pm.intSliderGrp("baker_size", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="size", field=True, v=size,
                    max=8192)
    pm.button(l="import meshes", c=multiImport, w=200)
    pm.button(l="check names", c=checkNames, w=200)
    pm.button("baker_uv", l="uv", w=200, c=applyUV)
    pm.button("render settings", w=200, c=lambda *args: preferences("redshiftOptions"))
    pm.button("ao settings", w=200, c=lambda *args: preferences("ao_texture_rs", "ao_material_rs"))
    pm.button("shadow settings", w=200, c=lambda *args: preferences("shadow_material_rs"))

    # pm.text(l="AO", w=200)
    # pm.floatSliderGrp("baker_radius", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="spread", field=True, v=.8)
    # pm.floatSliderGrp("baker_falloff", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="falloff", field=True, v=1)
    # pm.floatSliderGrp("baker_sub", cw3=[50, 50, 100], ct3=["left", "left", "lfet"], l="max dist", field=True, v=0)
    pm.text(l="bake", w=200)

    pm.checkBox("baker_id", l="bake id", v=True)
    pm.checkBox("baker_ao", l="bake ao", v=True)
    pm.checkBox("baker_shadow", l="bake shadow", v=True)
    pm.checkBox("baker_mat", l="bake materials", v=True)
    pm.checkBox("baker_mesh", l="save mesh", v=True)
    pm.text(h=30, l="")
    pm.button("baker_run", l="bake", w=200, c=render)
    pm.text(l="materials", w=200)
    for color in materials:
        pm.button(l=color, w=200, c=functools.partial(applyMaterial, color))

    win.show()


ui()