import os
import sys
import threading
import subprocess
import shutil
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from io import BytesIO
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageChops, ImageGrab, ImageOps
import json
import math
import random  # 【新增】需要用到随机数
from tkinter import simpledialog
import colorsys
import time
import concurrent.futures


from common import Config, check_path_safety, get_resource_path,CompareViewer
HAS_PYDNFEX = False
try:
    from pydnfex.npk import NPK
    from pydnfex.img import IMGFactory, ImageLink
    HAS_PYDNFEX = True
except ImportError:
    print("⚠️ 警告: 未检测到库，【调色盘】功能将不可用。")
# =========================================================================
# PART 4.9: 【Blender 渲染】核心逻辑 (纯净版 - 尊重 .blend 设置)
# =========================================================================

BLENDER_SCRIPT_CONTENT = r"""
import bpy
import sys
import os
import glob

# --- 1. 获取参数 ---
argv = sys.argv
try:
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
        input_dir = args[0]
        output_dir = args[1]
    else:
        input_dir = r"C:\Test\Input"
        output_dir = r"C:\Test\Output"
except:
    sys.exit(1)

print(f">>> [Auto Render] Start processing (Target: MainPlane ONLY)...")

# --- 2. 核心辅助函数：终极暴力设置图片 ---
def force_set_image(node, new_image):
    
    success = False
    node_name = node.name
    
    # 方式 A: 设置 .image 属性 (适用于绝大多数节点)
    if hasattr(node, "image"):
        try:
            node.image = new_image
            # print(f"    [OK] Set node.image for {node_name}")
            success = True
        except Exception as e:
            # print(f"    [Fail] Set node.image: {e}")
            pass

    # 方式 B: 设置输入端口 (Geometry Nodes 的 Image Texture 经常需要这个!)
    # 遍历所有输入，只要类型是 IMAGE 或者名字叫 Image，全给它填进去
    if hasattr(node, "inputs"):
        for inp in node.inputs:
            # 检查端口类型是否兼容图片
            is_image_socket = (inp.type == 'RGBA' or inp.name in ['Image', 'image', 'IMAGE', '图像'])
            
            if is_image_socket:
                try:
                    inp.default_value = new_image
                    # print(f"    [OK] Set input['{inp.name}'] for {node_name}")
                    success = True
                except:
                    pass
            
    return success

# --- 3. 递归查找函数 ---
def replace_image_recursive(node_tree, new_image, visited=None):
    if not node_tree: return 0
    if visited is None: visited = set()
    if node_tree in visited: return 0
    visited.add(node_tree)

    count = 0
    for node in node_tree.nodes:
        # 调试用：取消注释可以看到它扫描了哪些节点
        # print(f"Scanning node: {node.name} (Type: {node.type})")

        # 判断是否是目标节点 (名字包含 INPUT_NODE)
        is_target_name = "INPUT_NODE" in node.name.upper() or "INPUT_NODE" in node.label.upper()
        
        # 1. 命中目标：执行暴力替换
        if is_target_name:
            print(f"  -> Found Target: {node.name} in {node_tree.name}")
            if force_set_image(node, new_image):
                count += 1
            else:
                print(f"  -> WARNING: Found {node.name} but failed to set image!")

        # 2. 递归：如果是节点组，钻进去
        if node.type == 'GROUP' and node.node_tree:
            count += replace_image_recursive(node.node_tree, new_image, visited)
            
    return count

def update_main_plane_only(obj, new_image):
    if not obj: return 0
    total_replaced = 0

    # 1. 扫描【几何节点】 (Modifiers)
    # 这是你遇到问题的关键区域
    for mod in obj.modifiers:
        if mod.type == 'NODES' and mod.node_group:
            print(f"Scanning Geometry Nodes: {mod.name}")
            total_replaced += replace_image_recursive(mod.node_group, new_image)

    # 2. 扫描【材质着色器】 (Shader Nodes)
    for slot in obj.material_slots:
        if slot.material and slot.material.use_nodes and slot.material.node_tree:
            print(f"Scanning Shader: {slot.material.name}")
            total_replaced += replace_image_recursive(slot.material.node_tree, new_image)

    return total_replaced

# --- 4. 场景准备 ---
scene = bpy.context.scene
plane = scene.objects.get("MainPlane")
if not plane and len(bpy.context.selected_objects) > 0:
    plane = bpy.context.selected_objects[0]

if not plane:
    print("!!! ERROR: Object 'MainPlane' not found!")
else:
    print(f"Target Object: {plane.name}")

camera = scene.objects.get("MainCamera")
if not camera:
    for obj in scene.objects:
        if obj.type == 'CAMERA': camera = obj; break

if camera:
    scene.camera = camera
    camera.data.type = 'ORTHO'
    camera.data.sensor_fit = 'VERTICAL'

# --- 5. 渲染设置 ---
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.film_transparent = True 

if not os.path.exists(output_dir): os.makedirs(output_dir)
png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

# --- 6. 逐帧处理 ---
for i, png_path in enumerate(png_files):
    filename = os.path.basename(png_path)
    try:
        current_image = bpy.data.images.load(png_path)
        
        # 核心调用
        replaced_count = update_main_plane_only(plane, current_image)
        
        # 几何节点有时候需要强制更新一下依赖图
        if replaced_count > 0:
            bpy.context.view_layer.update()

        # 动态适配分辨率
        width = current_image.size[0]
        height = current_image.size[1]
        
        if height > 0:
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            
            if plane:
                ratio = width / height
                plane.scale.x = ratio
                plane.scale.y = 1.0
                plane.scale.z = 1.0
            
            if camera:
                camera.data.ortho_scale = 2.0

        scene.render.filepath = os.path.join(output_dir, filename)
        bpy.ops.render.render(write_still=True)
        bpy.data.images.remove(current_image)

    except Exception as e:
        print(f"Error: {e}")

print("Done_All_Tasks")
"""

class BlenderPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        # 配置变量
        self.blender_path = tk.StringVar(value=Config.get("blender_exe", ""))
        self.input_path = tk.StringVar(value=Config.get("blender_input_npk", "")) 
        self.output_dir = tk.StringVar(value=Config.get("blender_output_npk", ""))
        self.style_mode = tk.StringVar(value=Config.get("blender_style", "thunder"))
        self.template_dir = tk.StringVar(value=Config.get("blender_template_dir", os.path.join(os.getcwd(), "DNF")))
        # 【新增】用于缓存模版数据的字典
        # 结构: { "显示名称": { "path": "完整路径.blend", "desc": "描述..." } }
        self.template_cache = {} 
        # 自动寻找 Blender
        if not self.blender_path.get():
            default_paths = [
                r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
                r"C:\Program Files\Blender Foundation\Blender 3.3\blender.exe"
            ]
            for p in default_paths:
                if os.path.exists(p):
                    self.blender_path.set(p)
                    break

        self.create_ui()

    def create_ui(self):
        tk.Label(self, text="Blender 渲染", font=("微软雅黑", 16, "bold"), fg="#e67e22").pack(pady=10)

        # 1. 路径配置区
        f_cfg = ttk.LabelFrame(self, text=" 1. 环境与文件       Blender程序   请下载 : Blender 4.5.6 LTS", padding=10)
        f_cfg.pack(fill="x", padx=10, pady=5)

        self._add_row(f_cfg, "Blender程序:", self.blender_path, self.sel_exe)
        self._add_row(f_cfg, "模板文件夹:", self.template_dir, self.sel_template_dir, tip="(存放 .blend)")
        
        f_in = ttk.Frame(f_cfg); f_in.pack(fill="x", pady=2)
        ttk.Label(f_in, text="源文件(NPK):", width=15).pack(side="left")
        ttk.Entry(f_in, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(f_in, text="选文件", width=6, command=self.sel_npk_file).pack(side="left")
        ttk.Button(f_in, text="选目录", width=6, command=self.sel_npk_dir).pack(side="left", padx=2)

        self._add_row(f_cfg, "保存位置:", self.output_dir, self.sel_out_dir)

        # 2. 风格选择
        f_style = ttk.LabelFrame(self, text=" 2. 渲染风格 ", padding=10)
        f_style.pack(fill="x", padx=10, pady=5)
        
        f_s_inner = ttk.Frame(f_style)
        f_s_inner.pack(fill="x")
        ttk.Label(f_s_inner, text="选择特效模板:", width=15).pack(side="left")
        
        self.cb_style = ttk.Combobox(f_s_inner, textvariable=self.style_mode, state="readonly", width=20)
        self.cb_style.pack(side="left", padx=5)
        self.cb_style.bind("<<ComboboxSelected>>", self.on_template_select)
        
        ttk.Button(f_s_inner, text="🔄 刷新列表", command=self.refresh_templates).pack(side="left", padx=10)
        # 【新增】描述信息显示区域
        self.lbl_desc = ttk.Label(f_style, text="请选择一个模板...", foreground="gray", font=("微软雅黑", 9))
        self.lbl_desc.pack(fill="x", padx=5, pady=(5, 0))
        # 3. 操作区
        f_run = ttk.Frame(self, padding=10)
        f_run.pack(fill="both", expand=True)
        
        # 进度条
        self.pbar = ttk.Progressbar(f_run, mode='determinate')
        self.pbar.pack(fill="x", pady=(0, 5))
        
        # --- 按钮组 ---
        f_btns = ttk.Frame(f_run)
        f_btns.pack(fill="x", pady=5)
        
        self.btn_run = ttk.Button(f_btns, text="🚀 启动处理 (Extract -> Blender -> Repack)", command=self.start_process)
        self.btn_run.pack(side="left", fill="x", expand=True)
        
        # 【新增】试看与对比按钮
        self.btn_try = ttk.Button(f_btns, text="👁 试看一帧", command=self.try_process_one_frame)
        self.btn_try.pack(side="left", padx=5)
        
        self.btn_preview = ttk.Button(f_btns, text="🔍 对比预览", command=self.open_preview)
        self.btn_preview.pack(side="left", padx=5)

        # 日志框
        self.log_box = tk.Text(f_run, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9), state="disabled")
        sb = ttk.Scrollbar(f_run, command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        
        self.refresh_templates()
        
    # 【新增】当用户在下拉框选择某项时触发
    def on_template_select(self, event):
        name = self.cb_style.get()
        if name in self.template_cache:
            desc = self.template_cache[name].get("desc", "无描述")
            author = self.template_cache[name].get("author", "未知")
            self.lbl_desc.config(text=f"作者: {author} | 说明: {desc}", foreground="#2ecc71")
            Config.set("blender_style", name) # 保存选中的显示名称
        else:
            self.lbl_desc.config(text="模版无效", foreground="red")
            
    def _add_row(self, parent, label, var, cmd, tip=""):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=15).pack(side="left")
        ttk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
        if tip: ttk.Label(f, text=tip, foreground="gray", font=("Arial", 8)).pack(side="left", padx=2)
        ttk.Button(f, text="📂", width=4, command=cmd).pack(side="left")

    def sel_exe(self):
        p = filedialog.askopenfilename(filetypes=[("Blender", "blender.exe")])
        if p: self.blender_path.set(p); Config.set("blender_exe", p)

    def sel_template_dir(self):
        p = filedialog.askdirectory()
        if p: self.template_dir.set(p); Config.set("blender_template_dir", p); self.refresh_templates()

    def sel_npk_file(self):
        p = filedialog.askopenfilename(filetypes=[("NPK", "*.npk")])
        if p: self._set_in(p)
    
    def sel_npk_dir(self):
        p = filedialog.askdirectory()
        if p: self._set_in(p)

    def _set_in(self, p):
        self.input_path.set(p)
        Config.set("blender_input_npk", p)
        if not self.output_dir.get():
            base = os.path.dirname(p) if os.path.isfile(p) else p
            default_out = os.path.join(base, "Output_Blender_MOD")
            self.output_dir.set(default_out)
            Config.set("blender_output_npk", default_out)

    def sel_out_dir(self):
        p = filedialog.askdirectory()
        if p: self.output_dir.set(p); Config.set("blender_output_npk", p)

    def refresh_templates(self):
        root_dir = self.template_dir.get()
        self.template_cache = {} # 清空缓存
        display_names = []
        
        if not os.path.exists(root_dir):
            self.cb_style['values'] = ["(目录不存在)"]
            self.lbl_desc.config(text="❌ 模板目录不存在，请检查设置", foreground="red")
            return

        # 1. 遍历根目录下的所有子文件夹
        try:
            sub_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            
            count = 0
            for folder_name in sub_dirs:
                folder_path = os.path.join(root_dir, folder_name)
                meta_path = os.path.join(folder_path, "meta.json")
                
                # 2. 检查 meta.json 是否存在
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            
                        # 获取关键信息
                        name = data.get("name", folder_name) # 如果没写名字，用文件夹名
                        blend_file = data.get("file", "")
                        desc = data.get("description", "暂无介绍")
                        author = data.get("author", "佚名")
                        
                        # 3. 验证 .blend 文件是否存在
                        full_blend_path = os.path.join(folder_path, blend_file)
                        if os.path.exists(full_blend_path):
                            self.template_cache[name] = {
                                "path": full_blend_path, # 存绝对路径
                                "desc": desc,
                                "author": author
                            }
                            display_names.append(name)
                            count += 1
                    except Exception as e:
                        print(f"解析 {folder_name} 出错: {e}")
            
            # 4. 更新 UI
            if display_names:
                # 排序一下，好看
                display_names.sort()
                self.cb_style['values'] = display_names
                
                # 尝试恢复上次的选择
                current = self.style_mode.get()
                if current in display_names:
                    self.cb_style.set(current)
                    self.on_template_select(None) # 触发描述更新
                else:
                    self.cb_style.current(0)
                    self.on_template_select(None)
                    
                self.log(f"✅ 成功加载 {count} 个热插拔模版")
            else:
                self.cb_style['values'] = ["(未找到有效模版)"]
                self.lbl_desc.config(text="⚠️ 目录下没有包含 meta.json 的子文件夹", foreground="#e67e22")
                
        except Exception as e:
            self.log(f"刷新模版失败: {e}")

    def log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", str(msg) + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    # =================================================================
    # 【新增功能 1】 对比预览
    # =================================================================
    def open_preview(self):
        raw = self.input_path.get()
        out_dir = self.output_dir.get()
        if not raw or not os.path.exists(raw): 
            messagebox.showwarning("提示", "请先选择有效的源路径")
            return
        
        file_pairs = []
        if os.path.isfile(raw):
            fname = os.path.basename(raw)
            # 输出文件通常带 % 前缀
            dst_path = os.path.join(out_dir, "%" + fname)
            if fname.lower().endswith(".npk"): 
                file_pairs.append((raw, dst_path))
        else:
            for fname in os.listdir(raw):
                if fname.lower().endswith(".npk") and not fname.startswith("%"):
                    dst_path = os.path.join(out_dir, "%" + fname)
                    file_pairs.append((os.path.join(raw, fname), dst_path))
        
        if not file_pairs:
            messagebox.showinfo("提示", "没有找到 NPK 文件")
            return
        
        # 即使目标文件不存在，CompareViewer 也能打开，只是右侧不显示
        CompareViewer(self.winfo_toplevel(), file_pairs)

    # =================================================================
    # 【新增功能 2】 试看一帧
    # =================================================================
    def try_process_one_frame(self):
        exe = self.blender_path.get()
        src_path = self.input_path.get()
        # 【修改】从缓存获取路径
        selected_name = self.cb_style.get()
        if selected_name not in self.template_cache:
            return messagebox.showerror("错误", "请先选择一个有效的模版！")
            
        tpl_file_abs_path = self.template_cache[selected_name]["path"]

        if not os.path.exists(exe): return messagebox.showerror("错误", "Blender 路径无效")
        if not src_path or not os.path.exists(src_path): return messagebox.showerror("错误", "源路径无效")
        
        self.btn_try.config(state="disabled", text="渲染中...")
        
        threading.Thread(target=self._worker_try_one, args=(exe, tpl_file_abs_path, src_path)).start()

    def _worker_try_one(self, exe, tpl_file, src_path):
        try:
            # 1. 寻找有效图片帧
            target_npk_path = src_path
            if os.path.isdir(src_path):
                files = glob.glob(os.path.join(src_path, "*.npk"))
                if not files: raise Exception("文件夹内无NPK")
                target_npk_path = files[0]

            found_original = None
            original_pil = None
            found_info = ""
            
            # 临时目录
            temp_try_root = os.path.join(os.getcwd(), "_temp_blender_try")
            dir_in = os.path.join(temp_try_root, "in")
            dir_out = os.path.join(temp_try_root, "out")
            if os.path.exists(temp_try_root): shutil.rmtree(temp_try_root)
            os.makedirs(dir_in); os.makedirs(dir_out)

            # 2. 解包一张图
            with open(target_npk_path, 'rb') as f:
                npk = NPK.open(f)
                npk.load_all()
                for inner in npk.files:
                    if not inner.name.lower().endswith(".img"): continue
                    try:
                        img_obj = IMGFactory.open(BytesIO(inner.data))
                        # 找一张非空的、有一定大小的图
                        for i, frame in enumerate(img_obj.images):
                            pil_img = img_obj.build(frame)
                            if pil_img.width > 20 and pil_img.height > 20:
                                # 检查是否全透明
                                if pil_img.getextrema()[3][1] == 0: continue
                                
                                original_pil = pil_img
                                found_original = os.path.join(dir_in, f"try_{i}.png")
                                original_pil.save(found_original)
                                found_info = f"{os.path.basename(target_npk_path)} -> {inner.name} -> 帧{i}"
                                break
                        if found_original: break
                    except: continue
            
            if not found_original:
                raise Exception("未找到有效的测试帧")

            # 3. 生成脚本并运行 Blender
            script_path = os.path.join(temp_try_root, "runner.py")
            with open(script_path, "w", encoding="utf-8") as f: f.write(BLENDER_SCRIPT_CONTENT)

            cmd = [
                exe, tpl_file, "--background", "--python", script_path, "--",
                dir_in, dir_out
            ]
            
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', startupinfo=si)

            # 4. 读取结果
            out_files = glob.glob(os.path.join(dir_out, "*.png"))
            if not out_files:
                raise Exception("Blender 未生成输出文件 (渲染失败)")
            
            result_pil = Image.open(out_files[0]).convert("RGBA")
            
            # 5. 弹窗显示
            self.after(0, lambda: self._show_try_popup(original_pil, result_pil, found_info))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("试看失败", str(e)))
        finally:
            self.after(0, lambda: self.btn_try.config(state="normal", text="👁 试看一帧"))
            # 可以在这里清理 temp_try_root，也可以保留方便调试

    def _show_try_popup(self, img_src, img_dst, info_text):
        top = tk.Toplevel(self)
        top.title("Blender 效果试看")
        top.geometry("900x600")
        top.config(bg="#222") 
        
        ttk.Label(top, text=f"来源: {info_text}", background="#222", foreground="#aaa", font=("Consolas", 9)).pack(pady=5)
        
        f_imgs = tk.Frame(top, bg="#222")
        f_imgs.pack(expand=True, fill="both", padx=10, pady=10)
        
        # 智能缩放
        preview_h = 500
        scale_ratio = 1.0
        if img_src.height < 300: scale_ratio = 2.0 
        if img_src.height > 800: scale_ratio = 500 / img_src.height 
        
        def process_for_view(pil_img):
            w = int(pil_img.width * scale_ratio)
            h = int(pil_img.height * scale_ratio)
            return ImageTk.PhotoImage(pil_img.resize((w, h), Image.Resampling.NEAREST))
            
        tk_src = process_for_view(img_src)
        tk_dst = process_for_view(img_dst)
        
        # 左侧：原图
        f_l = tk.LabelFrame(f_imgs, text=" 修改前 (Original) ", bg="#222", fg="white")
        f_l.pack(side="left", expand=True, fill="both", padx=5)
        lbl_l = tk.Label(f_l, image=tk_src, bg="#333")
        lbl_l.image = tk_src
        lbl_l.pack(expand=True)
        
        tk.Label(f_imgs, text="▶", bg="#222", fg="#555", font=("Arial", 20)).pack(side="left", padx=5)

        # 右侧：效果图
        f_r = tk.LabelFrame(f_imgs, text=" 修改后 (Blender Render) ", bg="#222", fg="#00ff00")
        f_r.pack(side="left", expand=True, fill="both", padx=5)
        lbl_r = tk.Label(f_r, image=tk_dst, bg="#333")
        lbl_r.image = tk_dst
        lbl_r.pack(expand=True)

    # =================================================================
    # 主流程处理
    # =================================================================
    def start_process(self):
        # 1. 基础检查
        exe = self.blender_path.get()
        src_path = self.input_path.get()
        out_path = self.output_dir.get()
        
        # 【检查 1】Blender 路径
        if not os.path.exists(exe): 
            return messagebox.showerror("错误", "Blender 路径无效，请重新选择 blender.exe")
            
        # 【检查 2】输入路径
        if not src_path or not os.path.exists(src_path): 
            return messagebox.showerror("错误", "源 NPK 路径无效")
            
        # 【检查 3】模版选择 (热插拔逻辑)
        selected_name = self.cb_style.get()
        if not selected_name:
             return messagebox.showerror("错误", "未选择任何模板")
             
        if selected_name not in self.template_cache:
            # 尝试刷新一下，万一用户刚刚加了文件
            self.refresh_templates()
            if selected_name not in self.template_cache:
                return messagebox.showerror("错误", f"模版 '{selected_name}' 数据丢失，请尝试刷新列表或重新选择。")
            
        # 从缓存中获取 .blend 的【绝对路径】
        tpl_file_abs_path = self.template_cache[selected_name]["path"]
        
        if not os.path.exists(tpl_file_abs_path):
            return messagebox.showerror("错误", f"找不到模版文件:\n{tpl_file_abs_path}")

        # 【检查 4】NpkPatcher
        patcher_exe = get_resource_path("NpkPatcher.exe")
        if not os.path.exists(patcher_exe): 
            return messagebox.showerror("错误", "缺少 NpkPatcher.exe，无法进行封包")

        # 2. 锁定 UI
        Config.set("blender_style", selected_name)
        self.btn_run.config(state="disabled")
        self.btn_try.config(state="disabled") 
        
        # 清空并初始化日志
        self.log_box.config(state="normal")
        self.log_box.delete(1.0, "end")
        self.log_box.insert("end", ">>> 正在启动后台线程...\n")
        self.log_box.config(state="disabled")
        
        # 3. 启动线程 (传递绝对路径)
        threading.Thread(
            target=self.run_pipeline, 
            args=(exe, tpl_file_abs_path, src_path, out_path, patcher_exe),
            daemon=True # 设置为守护线程，防止关闭主程序后残留
        ).start()


    def run_pipeline(self, blender_exe, tpl_file, src_path, out_path, patcher_exe):
        # 这里的 tpl_file 已经是绝对路径了
        try:
            self.log(f"📋 任务开始")
            self.log(f"   - 模板: {os.path.basename(tpl_file)}")
            self.log(f"   - 源文件: {src_path}")
            
            # 1. 生成 Python 脚本文件
            # 这一步非常重要，必须确保 BLENDER_SCRIPT_CONTENT 是最新的（含 MainPlane 修复的）
            script_path = os.path.join(os.getcwd(), "_temp_blender_runner.py")
            try:
                with open(script_path, "w", encoding="utf-8") as f: 
                    f.write(BLENDER_SCRIPT_CONTENT)
            except Exception as e:
                self.log(f"❌ 无法写入临时脚本文件: {e}")
                return

            # 2. 扫描输入文件
            files = []
            if os.path.isfile(src_path): 
                files = [src_path]
            elif os.path.isdir(src_path):
                files = glob.glob(os.path.join(src_path, "*.npk"))

            if not files:
                self.log("❌ 未找到 NPK 文件，任务终止。")
                return

            if not os.path.exists(out_path): 
                os.makedirs(out_path)
            
            # 准备临时目录
            work_root = os.path.join(os.getcwd(), "_blender_work_temp")
            dir_extract = os.path.join(work_root, "raw")
            dir_render = os.path.join(work_root, "render")

            total_files = len(files)
            
            # 3. 开始循环处理
            for idx, npk_file in enumerate(files):
                fname = os.path.basename(npk_file)
                self.log(f"[{idx+1}/{total_files}] 处理 NPK: {fname}")
                
                # 更新进度条
                current_percent = (idx / total_files) * 100
                self.pbar['value'] = current_percent
                
                dst_npk_path = os.path.join(out_path, "%" + fname)
                
                try:
                    with open(npk_file, 'rb') as f:
                        npk = NPK.open(f)
                        npk.load_all()
                        
                        valid_imgs = [x for x in npk.files if x.name.lower().endswith('.img')]
                        if not valid_imgs:
                            self.log("   -> 跳过 (无 IMG 文件)")
                            continue
                        
                        modified_count = 0
                        
                        for i_img, img_entry in enumerate(valid_imgs):
                            # 清理并重建临时目录
                            if os.path.exists(work_root): 
                                try: shutil.rmtree(work_root)
                                except: pass # 偶尔会被占用，忽略
                            
                            os.makedirs(dir_extract, exist_ok=True)
                            os.makedirs(dir_render, exist_ok=True)
                            
                            # --- 解包逻辑 ---
                            img_obj = IMGFactory.open(BytesIO(img_entry.data))
                            csv_lines = []
                            has_valid_frames = False
                            
                            for i_frame, frame in enumerate(img_obj.images):
                                try:
                                    # 处理引用帧
                                    current_f = frame
                                    if isinstance(frame, ImageLink):
                                        t = -1
                                        if hasattr(frame, '_image'): t = frame._image
                                        elif hasattr(frame, 'link'): t = frame.link
                                        elif hasattr(frame, 'target'): t = frame.target
                                        if isinstance(t, int) and 0 <= t < len(img_obj.images):
                                            current_f = img_obj.images[t]

                                    pil_img = img_obj.build(current_f)
                                    
                                    # 保存 raw 图片给 Blender
                                    raw_name = f"frame_{i_frame:04d}.png"
                                    pil_img.save(os.path.join(dir_extract, raw_name))
                                    
                                    # 记录坐标 (封包用)
                                    ox = getattr(frame, 'x', getattr(frame, 'pos_x', 0))
                                    oy = getattr(frame, 'y', getattr(frame, 'pos_y', 0))
                                    
                                    # 记录渲染后的预期路径
                                    render_path = os.path.abspath(os.path.join(dir_render, raw_name))
                                    csv_lines.append(f"{render_path},{ox},{oy}")
                                    has_valid_frames = True
                                    
                                except Exception as e:
                                    # 出错则填个空位
                                    csv_lines.append(f"ERROR,0,0")

                            if not has_valid_frames: 
                                continue
                            
                            # --- 调用 Blender ---
                            self.log(f"   -> Blender 渲染中: {img_entry.name} ({len(csv_lines)}帧)")
                            
                            cmd = [
                                blender_exe, tpl_file, "--background", "--python", script_path, "--",
                                dir_extract, dir_render
                            ]
                            
                            # 隐藏控制台窗口
                            si = subprocess.STARTUPINFO()
                            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                            
                            ret = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', startupinfo=si)
                            
                            if ret.returncode != 0:
                                self.log(f"     ❌ Blender 报错 (Code {ret.returncode})")
                                # 如果出错，打印最后几行日志
                                err_lines = ret.stdout.split('\n')[-5:]
                                for l in err_lines: self.log(f"       {l}")
                                continue

                            # --- 检查渲染结果 ---
                            rendered_files = glob.glob(os.path.join(dir_render, "*.png"))
                            if not rendered_files:
                                self.log("     ⚠️ Blender 未输出图片，跳过合成")
                                continue

                            # --- 封包 ---
                            csv_path = os.path.join(work_root, "data.csv")
                            new_img_path = os.path.join(work_root, "new.img")
                            
                            with open(csv_path, "w", encoding="utf-8") as f_csv:
                                f_csv.write("\n".join(csv_lines))

                            p_ret = subprocess.run([patcher_exe, csv_path, new_img_path], capture_output=True, startupinfo=si)
                            
                            if p_ret.returncode == 0 and os.path.exists(new_img_path):
                                with open(new_img_path, "rb") as f_new:
                                    img_entry.set_data(f_new.read())
                                modified_count += 1
                            else:
                                self.log(f"     ❌ NpkPatcher 合成失败 (错误码: {p_ret.returncode})")
                                try:
                                    err_log = p_ret.stdout.decode('gbk', 'ignore')
                                    if not err_log: err_log = p_ret.stderr.decode('gbk', 'ignore')
                                    lines = err_log.splitlines()
                                    for l in lines[-3:]: # 只看最后3行
                                        if l.strip(): self.log(f"       [CMD] {l.strip()}")
                                except:
                                    pass

                        if modified_count > 0:
                            with open(dst_npk_path, "wb") as f_out:
                                npk.save(f_out)
                            self.log(f"   ✅ NPK 保存成功: {os.path.basename(dst_npk_path)}")
                        else:
                            self.log(f"   ⚠️ 无修改内容，跳过保存")
                        
                except Exception as e:
                    self.log(f"❌ 处理 NPK 异常: {e}")
                    import traceback
                    traceback.print_exc()

            self.log("-" * 30)
            self.log("🎉 所有任务结束！")
            self.pbar['value'] = 100
            messagebox.showinfo("完成", f"处理结束！\n文件已保存至: {out_path}")

        except Exception as thread_err:
            self.log(f"❌ 线程严重错误: {thread_err}")
            messagebox.showerror("线程错误", str(thread_err))
            
        finally:
            # 清理
            try: shutil.rmtree(work_root)
            except: pass
            if os.path.exists(script_path): 
                try: os.remove(script_path)
                except: pass
            
            # 恢复按钮状态
            self.after(0, lambda: self.btn_run.config(state="normal"))
            self.after(0, lambda: self.btn_try.config(state="normal"))