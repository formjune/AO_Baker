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
import time


material_dir = "C:/textures/materials"   # location with material textures
textures_dir = "C:/textures/textures"    # location with AO, ID etc textures
redshift_dir = os.path.join(pm.workspace(fn=True), "images")
name_pattern = "%material_%channel"
size = 1024
materials = {"default": (128, 128, 128), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
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

    global size, material_dir, textures_dir, redshift_dir, name_pattern
    size = pm.intSliderGrp("baker_size", q=True, v=True)
    material_dir = pm.textField("baker_mat_dir", q=True, tx=True)
    textures_dir = pm.textField("baker_out_dir", tx=True, q=True)
    redshift_dir = os.path.join(pm.workspace(fn=True), "images")
    name_pattern = pm.textField("baker_pattern", tx=True, q=True)

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
        # material.setAttr("emission_weight", 1)
        texture = pm.createNode("RedshiftAmbientOcclusion", n="ao_texture_rs")
        texture.connectAttr("outColor", material.attr("color"))
        # texture.connectAttr("outColor", material.attr("emission_color"))

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
    # clean and create new dir
    print 2
    if not os.path.isdir(redshift_dir):
        os.makedirs(redshift_dir)

    for name in os.listdir(redshift_dir):
        name = os.path.join(redshift_dir, name)
        print name
        if os.path.isfile(name):
            os.remove(name)
        else:
            shutil.rmtree(name)

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


def renderID(*args):
    """ID and save obj"""

    settings()
    selected = pm.selected(type="transform") or getMeshes()
    selected.sort(key=lambda x: x.name())
    export = pm.optionMenu("baker_export", q=True, v=True)
    applyMaterial("default")

    for mesh in selected:
        mesh_name = mesh.name()
        mesh_full_name = os.path.join(textures_dir, mesh_name, mesh_name)
        mesh_full_dir = os.path.join(textures_dir, mesh_name)

        if not os.path.exists(mesh_full_dir):
            os.makedirs(mesh_full_dir)

        bakeID(mesh, mesh_full_name + "_id")
        pm.select(mesh)
        if export == "obj":
            cmds.file(mesh_full_name + ".obj", force=True, type='OBJexport', es=True,
                      options='groups=1;ptgroups=1;materials=0;smoothing=1;normals=1')

        elif export == "fbx":
            pm.mel.FBXExport(f=mesh_full_name.replace("\\", "/") + ".fbx", s=True)
        else:
            pm.mel.FBXExportFileVersion(v="FBX201600")
            pm.mel.FBXExportColladaTriangulate(True)
            pm.mel.FBXExport(s=True, f=mesh_full_name.replace("\\", "/") + ".dae", caller="FBXDAEMayaTranslator")


def renderAO(*args):
    """AO and SHADOW"""

    settings()
    selected = pm.selected(type="transform") or getMeshes()
    selected.sort(key=lambda x: x.name())
    export = pm.optionMenu("baker_export", q=True, v=True)

    for mesh in selected:
        t = time.time()
        mesh_name = mesh.name()
        mesh_full_name = os.path.join(textures_dir, mesh_name, mesh_name)
        mesh_full_dir = os.path.join(textures_dir, mesh_name)
        if not os.path.exists(mesh_full_dir):
            os.makedirs(mesh_full_dir)

        pm.select(mesh)
        pm.hyperShade(a="ao_material_rs")
        # if pm.checkBox("baker_ao", v=True, q=True):
        bake(mesh, mesh_full_name + "_light.png")

        print "%s ao: %s" % (mesh_name, time.time() - t)
        t = time.time()

        # if pm.checkBox("baker_shadow", v=True, q=True):
        #     pm.select(mesh)
        #     pm.hyperShade(a="shadow_material_rs")
        #     bake(mesh, mesh_full_name + "_shadow.png")

        print "%s shadow: %s" % (mesh_name, time.time() - t)

        pm.select(mesh)
        applyMaterial("default")
        if export == "obj":
            cmds.file(mesh_full_name + ".obj", force=True, type='OBJexport', es=True,
                      options='groups=1;ptgroups=1;materials=0;smoothing=1;normals=1')

        elif export == "fbx":
            pm.mel.FBXExport(f=mesh_full_name.replace("\\", "/") + ".fbx", s=True)
        else:
            pm.mel.FBXExportFileVersion(v="FBX201600")
            pm.mel.FBXExportColladaTriangulate(True)
            pm.mel.FBXExport(s=True, f=mesh_full_name.replace("\\", "/") + ".dae", caller="FBXDAEMayaTranslator")


def renderMaterial(*args):

    settings()
    selected = pm.selected(type="transform") or getMeshes()
    selected.sort(key=lambda x: x.name())
    mat_list = findNames()
    for mesh in selected:
        mesh_name = mesh.name()
        mesh_full_name = os.path.join(textures_dir, mesh_name, mesh_name)
        mesh_full_dir = os.path.join(textures_dir, mesh_name)
        if not os.path.exists(mesh_full_dir):
            os.makedirs(mesh_full_dir)

        bakeMaterials(mat_list, mesh_full_name)


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

    pm.text(l="bake", w=200)

    pm.optionMenu("baker_export", w=200)
    pm.menuItem(label='obj')
    pm.menuItem(label='fbx')
    pm.menuItem(label='dae')
    pm.button(l="bake id and mesh", w=200, c=renderID)
    pm.checkBox("baker_ao", l="bake ao", v=True)
    pm.checkBox("baker_shadow", l="bake shadow", v=True)
    pm.button(l="bake ao and shadow", w=200, c=renderAO)
    pm.button(l="bake material", w=200, c=renderMaterial)
    pm.text(l="materials", w=200)
    for color in materials:
        pm.button(l=color, w=200, c=functools.partial(applyMaterial, color))

    settings()
    win.show()

ui()