"""
place script, numpy and cv2.pyd into documents/maya/scripts.
run python code: import Baker_Redshift;reload(Baker_Redshift);

edit Default for setting up default values. They can be change in GUI at anytime
material_dir = directory with materials to place
output_dir = directory where all textures will be stored
name_pattern = pattern for material textures recognition
size = size of output textures
multiply_channels = which channels will be multiplied with AO and SHADOWS
materials = ID colors. key - material name, value - RGB color for mask
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
import time
from PySide2.QtGui import *


class Default(object):
    """default preferences"""

    material_dir = "C:/textures/materials"   # location with material textures
    output_dir = "C:/textures/textures"    # location with AO, ID etc textures
    name_pattern = "%material_%channel"
    size = 1024
    multiply_channels = "BaseColor",
    materials = {"default": (255, 255, 255), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}


# not for edit
if not pm.pluginInfo("redshift4maya.mll", q=True, loaded=True):
    pm.loadPlugin("redshift4maya.mll")
material_node = collections.namedtuple("material", "file name channel")
allowed_symbols = set(string.ascii_letters + string.digits + "_")


def applyMaterial(color, *args):
    """apply id material to selected"""
    try:
        material = nt.Lambert(color + "_id")
    except pm.MayaNodeError:
        selected = pm.selected()
        material = pm.createNode("lambert", n=color + "_id_material")  # vray node
        color = [c / 255. for c in Default.materials[color]]
        material.setAttr("color", color)
        pm.select(selected)
    pm.hyperShade(a=material)


def applyUV(*args):
    pm.polyAutoProjection(lm=0, pb=0, ibd=1, cm=0, l=2, sc=1, o=1, p=6, ps=.2, ws=0)


def multiImport(*args):
    for f in pm.fileDialog2(ff="*.obj", fm=4):
        pm.cmds.file(f, i=True)


def getMeshes():
    return [x for x in pm.ls(type="transform") if x.listRelatives(type="mesh")]


def findMask(image, rgb):
    """convert texture to mask. args (np.array, (r, g, b))"""
    def minMax(color):
        return min(255, max(color, 0))
    top = np.array([minMax(rgb[2] + 4), minMax(rgb[1] + 4), minMax(rgb[0] + 4)], dtype=np.uint8)
    bottom = np.array([minMax(rgb[2] - 4), minMax(rgb[1] - 4), minMax(rgb[0] - 4)], dtype=np.uint8)
    return cv2.inRange(image, bottom, top)


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


class Baker(object):

    def __init__(self):

        if pm.window("Baker", ex=True):
            pm.deleteUI("Baker")
        win = pm.window("Baker", wh=(620, 580), tlb=True, t="redshift baker")
        pm.rowLayout(w=420, nc=2, cw=((1, 200), (2, 400)))

        pm.columnLayout(w=200)

        pm.text(label="material directory", w=200)
        self.mat_folder = pm.textField(tx=Default.material_dir, w=200)
        pm.text(label="output directory", w=200)
        self.out_folder = pm.textField(tx=Default.output_dir, w=200)
        pm.text(label="material name pattern")
        self.pattern = pm.textField(tx=Default.name_pattern, w=200)
        self.size = pm.intSliderGrp(cw3=[30, 50, 110], ct3=["left", "left", "lfet"], label="size", field=True,
                                    v=Default.size, max=8192)
        pm.button(label="import meshes", c=multiImport, w=200)
        pm.button(label="check names", c=checkNames, w=200)
        pm.button(label="create uv", w=200, c=applyUV)
        pm.button(label="render settings", w=200, c=lambda *args: preferences("redshiftOptions"))
        pm.button(label="AO shader settings", w=200,
                  c=lambda *args: preferences("ao_texture", "ao_material"))
        pm.button(label="shadow shader settings", w=200, c=lambda *args: preferences("shadow_material"))
        pm.button(label="remove empty png", w=200, c=self.cleanEmptyFiles)
        self.format = pm.optionMenu(w=200)
        pm.menuItem(label='obj')
        pm.menuItem(label='fbx')
        pm.menuItem(label='dae')
        self.default_mat = pm.checkBox(label="default material for ID", v=True)
        self.auto_levels = pm.checkBox(label="auto levels for shadows", v=False)
        self.ignore_exist = pm.checkBox(label="ignore existence", v=True)
        self.shadow = pm.checkBox(label="bake shadows", v=True)
        self.ao = pm.checkBox(label="bake ambient occlusion", v=True)

        pm.button(label="bake id", w=200, c=self.renderID)
        pm.button(label="bake mesh", w=200, c=self.renderMesh)
        pm.button(label="bake AO and shadow", w=200, c=self.renderAO)
        pm.button(label="create material", w=200, c=self.renderMaterial)

        pm.text(l="materials", w=200)
        for color in Default.materials:
            pm.button(label=color, w=200, c=functools.partial(applyMaterial, color))

        pm.setParent(u=True)

        column = pm.columnLayout(w=400)
        self.progress = pm.progressBar(w=400)
        self.out_field = QTextEdit()
        self.out_field.setFixedSize(400, 500)
        self.out_field.setObjectName("baker_out")
        pm.control("baker_out", e=True, p=column)
        self.settings()
        win.show()

    def settings(self):
        """setup vray before baking"""

        if not pm.objExists("redshiftOptions"):
            pm.createNode("RedshiftOptions", n="redshiftOptions")
        pm.setAttr("redshiftOptions.imageFormat", 2)
        pm.optionVar(intValue=("redshiftBakeDefaultsHeight", self.size.getValue()))
        pm.optionVar(intValue=("redshiftBakeDefaultsWidth", self.size.getValue()))
        pm.optionVar(intValue=("redshiftBakeDefaultsTileMode", 1))
        pm.optionVar(stringValue=("redshiftBakeDefaultsUvSet", "map1"))
        pm.optionVar(intValue=("redshiftBakeMode", 2))

        # ao material
        if not pm.objExists("ao_material"):
            material = pm.createNode("RedshiftMaterial", n="ao_material")
            material.setAttr("emission_weight", 2)
            material.setAttr("overall_color", (0, 0, 0))
            texture = pm.createNode("RedshiftAmbientOcclusion", n="ao_texture")
            texture.connectAttr("outColor", material.attr("diffuse_color"))
            texture.connectAttr("outColor", material.attr("emission_color"))

        if not pm.objExists("shadow_material"):
            material = pm.createNode("RedshiftMaterial", n="shadow_material")
            material.setAttr("diffuse_color", (1, 1, 1))

    def findNames(self):
        """convert names into FileNode[]"""

        def makeNode(full_path):
            name = os.path.split(full_path)[1]
            if not os.path.isfile(full_path):
                return None
            keys = os.path.splitext(name)[0].split("_")
            try:
                return material_node(full_path, keys[name_id] if name_id != -1 else "",
                                     keys[channel_id] if channel_id != -1 else "")
            except IndexError:
                return None

        sample = self.pattern.getText()
        path = self.mat_folder.getText()
        print(path, sample)
        split_sample = os.path.splitext(sample)[0].split("_")
        name_id = split_sample.index("%material") if "%material" in split_sample else -1
        channel_id = split_sample.index("%channel") if "%channel" in split_sample else -1

        if os.path.isdir(path):
            nodes = [makeNode(os.path.join(path, file_name)) for file_name in os.listdir(path)]
            return [node for node in nodes if node is not None]
        else:
            return []

    def bakeID(self, mesh, obj_name):
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
            mapHeight=self.size.getValue(),
            mapWidth=self.size.getValue(),
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

    def cleanEmptyFiles(self, *args):
        for folder, subfolders, subfiles in os.walk(self.out_folder.getText()):
            for f in subfiles:
                name = os.path.join(folder, f)
                if os.path.splitext(name)[1] != ".png":
                    continue
                if not os.path.isfile(name):
                    continue

                if not os.path.getsize(name):
                    os.remove(name)

    def bakeAO(self, mesh, mesh_name, autolevel=False):
        """bake 0 - id, 1 - ao, 2 - shadows"""
        # clean and create new dir
        redshift_dir = os.path.join(pm.workspace(fn=True), "images")
        if not os.path.isdir(redshift_dir):
            os.makedirs(redshift_dir)

        for name in os.listdir(redshift_dir):
            name = os.path.join(redshift_dir, name)
            if os.path.isfile(name):
                os.remove(name)
            else:
                shutil.rmtree(name)

        if os.path.exists(mesh_name) and not self.ignore_exist.getValue():
            return "skipped"
        open(mesh_name, "w").close()    # claim file

        pm.select(mesh)
        pm.rsRender(bake=True)
        # raise IndexError in case of render error or pressing escape
        old_name = os.path.join(redshift_dir, os.listdir(redshift_dir)[0])
        open(mesh_name, "wb").write(open(old_name, "rb").read())

        if autolevel:
            array = cv2.imread(mesh_name)
            ar = array.flatten()
            min_v = min(ar)
            max_v = max(ar)
            array = (array - min_v) * 255. / (max_v - min_v)
            array = array.astype(np.uint8)
            cv2.imwrite(mesh_name, array)

        if "_ao.png" in mesh_name:
            matrix = cv2.imread(mesh_name)
            matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(mesh_name, matrix)

        return "done"

    def bakeMaterials(self, mat_list, mesh_name):
        """create material maps based on id"""

        mask_map = cv2.imread(mesh_name + "_id.png")
        if mask_map is None:
            self.out_field.append("skipped: " + mesh_name)
            pm.warning(mesh_name + " id map doesn't exists")
            return

        size = self.size.getValue()
        mask_map = cv2.resize(mask_map, (size, size))
        for channel in {ch.channel for ch in mat_list}:
            channel_map = np.zeros((size, size, 3), dtype=np.uint8)  # create empty channel

            for color in Default.materials:
                found = [mt for mt in mat_list if mt.name == color and mt.channel == channel]
                if not found:
                    pm.warning("%s_%s doesn't exists" % (color, channel))
                    continue
                image_file = cv2.resize(cv2.imread(found[0].file), (size, size))
                color_mask = findMask(mask_map, Default.materials[color])
                channel_map |= cv2.bitwise_or(image_file, image_file, mask=color_mask)

            # multiply
            text = "done: %s, %s " % (os.path.split(mesh_name)[1], channel)
            if channel in Default.multiply_channels:
                ao_mask = cv2.imread(mesh_name + "_ao.png")
                if ao_mask is not None:
                    ao_mask = cv2.resize(ao_mask, (size, size))
                    ao_mask = cv2.cvtColor(cv2.cvtColor(ao_mask, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) / 255.
                    channel_map *= ao_mask
                    text += "AO "

                shadow_mask = cv2.imread(mesh_name + "_shadow.png")
                if shadow_mask is not None:
                    shadow_mask = cv2.resize(shadow_mask, (size, size))
                    shadow_mask = cv2.cvtColor(cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) / 255.
                    channel_map *= shadow_mask
                    text += "SHADOW"

            self.out_field.append(text)
            cv2.imwrite("%s_mat_%s.png" % (mesh_name, channel), channel_map.astype(np.uint8))

    def renderObj(self, mesh, mesh_full_name):
        export = self.format.getValue()
        pm.select(mesh)
        if export == "obj":
            pm.cmds.file(mesh_full_name + ".obj", force=True, type='OBJexport', es=True,
                         options='groups=1;ptgroups=1;materials=0;smoothing=1;normals=1')
        elif export == "fbx":
            pm.mel.FBXExport(f=mesh_full_name.replace("\\", "/") + ".fbx", s=True)
        else:
            pm.mel.FBXExportFileVersion(v="FBX201600")
            pm.mel.FBXExportColladaTriangulate(True)
            pm.mel.FBXExport(s=True, f=mesh_full_name.replace("\\", "/") + ".dae", caller="FBXDAEMayaTranslator")

    def renderID(self, *args):
        """ID and save obj"""

        self.settings()
        selected = pm.selected(type="transform") or getMeshes()
        selected.sort(key=lambda x: x.name())
        if self.default_mat.getValue():
            pm.select(selected)
            applyMaterial("default")
        textures_dir = self.out_folder.getText()
        self.progress.setMaxValue(len(selected))
        self.progress.setProgress(0)
        self.out_field.setPlainText("")

        for mesh in selected:
            mesh_name = mesh.name()
            mesh_full_name = os.path.join(textures_dir, mesh_name, mesh_name)
            mesh_full_dir = os.path.join(textures_dir, mesh_name)

            if not os.path.exists(mesh_full_dir):
                os.makedirs(mesh_full_dir)

            self.bakeID(mesh, mesh_full_name + "_id")
            self.out_field.append("done, " + mesh_name)
            self.progress.step(1)
        self.progress.setProgress(0)

    def renderMesh(self, *args):
        """ID and save obj"""

        self.settings()
        selected = pm.selected(type="transform") or getMeshes()
        selected.sort(key=lambda x: x.name())
        textures_dir = self.out_folder.getText()
        self.progress.setMaxValue(len(selected))
        self.progress.setProgress(0)
        self.out_field.setPlainText("")

        for mesh in selected:
            mesh_name = mesh.name()
            mesh_full_name = os.path.join(textures_dir, mesh_name, mesh_name)
            mesh_full_dir = os.path.join(textures_dir, mesh_name)

            if not os.path.exists(mesh_full_dir):
                os.makedirs(mesh_full_dir)

            self.renderObj(mesh, mesh_full_name)
            self.out_field.append("done, " + mesh_name)
            self.progress.step(1)
        self.progress.setProgress(0)

    def renderAO(self, *args):
        """AO and SHADOW"""

        self.settings()
        selected = pm.selected(type="transform") or getMeshes()
        selected.sort(key=lambda x: x.name())
        textures_dir = self.out_folder.getText()
        self.progress.setMaxValue(len(selected))
        self.progress.setProgress(0)
        self.out_field.setPlainText("")

        for mesh in selected:

            mesh_name = mesh.name()
            mesh_full_name = os.path.join(textures_dir, mesh_name, mesh_name)
            mesh_full_dir = os.path.join(textures_dir, mesh_name)
            if not os.path.exists(mesh_full_dir):
                os.makedirs(mesh_full_dir)

            try:
                if self.ao.getValue():
                    t = time.time()
                    pm.select(mesh)
                    pm.hyperShade(a="ao_material")
                    result = self.bakeAO(mesh, mesh_full_name + "_ao.png")
                    self.out_field.append("%s: %s_ao, %s" % (result, mesh_name, time.time() - t))

                if self.shadow.getValue():
                    t = time.time()
                    pm.select(selected)
                    pm.hyperShade(a="shadow_material")
                    result = self.bakeAO(mesh, mesh_full_name + "_shadow.png", self.auto_levels.getValue())
                    self.out_field.append("%s: %s_shadow, %s" % (result, mesh_name, time.time() - t))

            except IndexError:
                self.out.field("render was forcefully stopped")
                break
            self.progress.step(1)
        self.progress.setProgress(0)

    def renderMaterial(self, *args):

        self.settings()
        textures_dir = self.out_folder.getText()
        selected = pm.selected(type="transform") or getMeshes()
        selected.sort(key=lambda x: x.name())
        mat_list = self.findNames()
        self.progress.setMaxValue(len(selected))
        self.progress.setProgress(0)
        self.out_field.setPlainText("")

        for mesh in selected:
            mesh_name = mesh.name()
            mesh_full_name = os.path.join(textures_dir, mesh_name, mesh_name)
            mesh_full_dir = os.path.join(textures_dir, mesh_name)
            if not os.path.exists(mesh_full_dir):
                os.makedirs(mesh_full_dir)

            self.bakeMaterials(mat_list, mesh_full_name)
            self.progress.step(1)
        self.progress.setProgress(0)

Baker()
