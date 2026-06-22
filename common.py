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

# =========================================================================
# 配置管理 (新增)
# =========================================================================
CONFIG_FILE = "settings.json"
try:
    from pydnfex.npk import NPK
    from pydnfex.img import IMGFactory, ImageLink
    HAS_PYDNFEX = True
except ImportError:
    print("⚠️ 警告: 未检测到库，【调色盘】功能将不可用。")
    
class Config:
    _data = {}

    @classmethod
    def load(cls):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cls._data = json.load(f)
            except:
                cls._data = {}

    @classmethod
    def get(cls, key, default=None):
        return cls._data.get(key, default)

    @classmethod
    def set(cls, key, value):
        cls._data[key] = value
        cls.save()

    @classmethod
    def save(cls):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cls._data, f, indent=4, ensure_ascii=False)
        except:
            pass

# 初始化加载配置
Config.load()
def is_ascii_path(path):
    """检查路径是否只包含英文字符 (ASCII)"""
    try:
        path.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False
# --- 【新增】 路径安全检查 (防止中文路径导致工具报错) ---
def check_path_safety():
    """
    检查当前运行路径是否包含中文或非ASCII字符。
    如果有，弹出警告并退出程序。
    """
    current_path = os.getcwd()
    try:
        # 尝试将路径编码为 ASCII，如果失败说明包含中文或其他特殊字符
        current_path.encode('ascii')
    except UnicodeEncodeError:
        # 创建一个临时的 root 窗口用于弹窗（因为主程序还没启动）
        root = tk.Tk()
        root.withdraw() # 隐藏主窗口
        messagebox.showerror(
            "路径错误)", 
            f"检测到程序运行在包含中文或特殊字符的路径下：\n\n{current_path}\n\n"
            "由于工具不支持中文路径\n"
            "继续运行将导致文件生成失败或程序崩溃\n\n"
            "请将本程序移动到【纯英文路径】下运行！\n"
            "(例如: D:\\DNF_Mod_Tool\\)"
        )
        root.destroy()
        sys.exit(1) # 强制退出

# 立即执行路径检查
# Packaged copies may live under a Chinese output folder; keep the original source untouched.
# check_path_safety()


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
    
    
class CompareViewer(tk.Toplevel):
    def __init__(self, master, file_pairs, jump_to_img=None, jump_to_frame=0):
        super().__init__(master)
        self.file_pairs = file_pairs
        
        # --- 判断是否为“单图查看模式” ---
        # 如果传入的文件对中，第二个路径是 None，则说明不需要对比
        self.is_single_mode = False
        if file_pairs and file_pairs[0][1] is None:
            self.is_single_mode = True
            self.title(f"素材预览 (单文件模式)")
        else:
            self.title(f"对比预览 ({len(file_pairs)} 个文件)")

        self.geometry("1100x700")
        
        self.jump_to_img = jump_to_img
        self.jump_to_frame = jump_to_frame
        
        self.current_src_npk = None
        self.current_dst_npk = None
        self.curr_src_img = None
        self.curr_dst_img = None
        self.curr_frames_count = 0
        self.bg_mode = 0 
        self.boost_visibility = tk.BooleanVar(value=False) 
        self.use_coords = tk.BooleanVar(value=False)
        self.is_playing = False
        self.play_job = None
        
        self.create_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        if self.file_pairs:
            self.lb_npk.selection_set(0)
            self.on_select_npk(None)

    def create_ui(self):
        main_paned = tk.PanedWindow(self, orient="horizontal", sashwidth=5)
        main_paned.pack(fill="both", expand=True)
        frame_npk = ttk.LabelFrame(main_paned, text="📂 1. NPK 文件列表", width=220)
        main_paned.add(frame_npk)
        sb_npk = ttk.Scrollbar(frame_npk)
        sb_npk.pack(side="right", fill="y")
        self.lb_npk = tk.Listbox(frame_npk, yscrollcommand=sb_npk.set, font=("Consolas", 9), exportselection=False)
        self.lb_npk.pack(fill="both", expand=True, padx=2, pady=2)
        sb_npk.config(command=self.lb_npk.yview)
        for src, _ in self.file_pairs: self.lb_npk.insert("end", os.path.basename(src))
        self.lb_npk.bind("<<ListboxSelect>>", self.on_select_npk)

        frame_img = ttk.LabelFrame(main_paned, text="📄 2. 内部 .img", width=250)
        main_paned.add(frame_img)
        sb_img = ttk.Scrollbar(frame_img)
        sb_img.pack(side="right", fill="y")
        self.lb_img = tk.Listbox(frame_img, yscrollcommand=sb_img.set, font=("Consolas", 9), exportselection=False)
        self.lb_img.pack(fill="both", expand=True, padx=2, pady=2)
        sb_img.config(command=self.lb_img.yview)
        self.lb_img.bind("<<ListboxSelect>>", self.on_select_img)

        frame_view = ttk.Frame(main_paned)
        main_paned.add(frame_view)
        view_paned = tk.PanedWindow(frame_view, orient="horizontal")
        view_paned.pack(fill="both", expand=True, pady=5)
        
        # 左侧画面 (始终显示)
        # 【修改】单图模式下标题改一下
        txt_l = "画面预览" if self.is_single_mode else "原始 (Original)"
        f_l = ttk.LabelFrame(view_paned, text=txt_l)
        view_paned.add(f_l)
        self.cvs_l = tk.Canvas(f_l, bg="#333", highlightthickness=0)
        self.cvs_l.pack(fill="both", expand=True)
        self.cvs_l.bind("<Configure>", lambda e: self.refresh_view())
        self.cvs_l.bind("<ButtonPress-1>", self.on_canvas_drag_start)
        self.cvs_l.bind("<B1-Motion>", self.on_canvas_drag)
        
        # 右侧画面 (仅在对比模式下显示)
        self.cvs_r = None
        if not self.is_single_mode:
            f_r = ttk.LabelFrame(view_paned, text="修改后 (Modified)")
            view_paned.add(f_r)
            self.cvs_r = tk.Canvas(f_r, bg="#333", highlightthickness=0)
            self.cvs_r.pack(fill="both", expand=True)
            self.cvs_r.bind("<Configure>", lambda e: self.refresh_view())
            self.cvs_r.bind("<ButtonPress-1>", self.on_canvas_drag_start)
            self.cvs_r.bind("<B1-Motion>", self.on_canvas_drag)
        
        ctrl_frame = ttk.Frame(frame_view)
        ctrl_frame.pack(fill="x", padx=5, pady=5)
        self.btn_play = ttk.Button(ctrl_frame, text="▶️ 播放", width=8, command=self.toggle_play)
        self.btn_play.pack(side="left", padx=(0, 5))
        ttk.Label(ctrl_frame, text="帧:").pack(side="left")
        self.scale_frame = ttk.Scale(ctrl_frame, from_=0, to=1, orient="horizontal", command=self.on_frame_change)
        self.scale_frame.pack(side="left", fill="x", expand=True, padx=10)
        self.lbl_info = ttk.Label(ctrl_frame, text="0/0", width=8)
        self.lbl_info.pack(side="left")
        ttk.Button(ctrl_frame, text="背景色", width=6, command=self.toggle_bg).pack(side="right", padx=2)
        cb_coord = ttk.Checkbutton(ctrl_frame, text="📍应用坐标", variable=self.use_coords, command=self.refresh_view)
        cb_coord.pack(side="right", padx=5)
        ttk.Button(ctrl_frame, text="回中", width=6, command=self.reset_view_pos).pack(side="right", padx=2)
        cb_boost = ttk.Checkbutton(ctrl_frame, text="🔥增强", variable=self.boost_visibility, command=self.refresh_view)
        cb_boost.pack(side="right", padx=5)
        self.view_offset_x = 0; self.view_offset_y = 0

    def refresh_view(self): self.update_view(int(self.scale_frame.get()))
    def reset_view_pos(self): self.view_offset_x = 0; self.view_offset_y = 0; self.refresh_view()
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False; self.btn_play.config(text="▶️ 播放")
            if self.play_job: self.after_cancel(self.play_job); self.play_job = None
        else:
            if self.curr_frames_count > 1:
                self.is_playing = True; self.btn_play.config(text="⏸️ 暂停")
                self._play_loop()
    def _play_loop(self):
        if not self.is_playing: return
        current = int(self.scale_frame.get())
        next_frame = (current + 1) % self.curr_frames_count
        self.scale_frame.set(next_frame); self.update_view(next_frame)
        self.play_job = self.after(80, self._play_loop)
    def on_close(self):
        self.is_playing = False; 
        if self.play_job: self.after_cancel(self.play_job)
        self.destroy()
    def on_select_npk(self, event):
        sel = self.lb_npk.curselection()
        if not sel: return
        idx = sel[0]
        src_path, dst_path = self.file_pairs[idx]
        self.current_src_npk = None; self.current_dst_npk = None
        if self.is_playing: self.toggle_play()
        try:
            self.current_src_npk = NPK.open(open(src_path, 'rb')); self.current_src_npk.load_all()
            # 【修改】只在 dst_path 存在时才加载第二个
            if dst_path and os.path.exists(dst_path):
                self.current_dst_npk = NPK.open(open(dst_path, 'rb')); self.current_dst_npk.load_all()
        except Exception as e: messagebox.showerror("错误", f"无法打开文件:\n{e}"); return
        self.lb_img.delete(0, "end"); self.file_map = {}
        for f in self.current_src_npk.files:
            if f.name.lower().endswith(".img"):
                fname = f.name
                dst_f = None
                if self.current_dst_npk:
                    for df in self.current_dst_npk.files:
                        if df.name == fname: dst_f = df; break
                self.file_map[fname] = (f, dst_f)
                self.lb_img.insert("end", fname)
        self.cvs_l.delete("all")
        if self.cvs_r: self.cvs_r.delete("all")

        # 自动选中目标
        if self.jump_to_img:
            # 查找目标 img 在列表中的索引
            target_idx = -1
            all_imgs = self.lb_img.get(0, "end")
            for i, name in enumerate(all_imgs):
                if name == self.jump_to_img:
                    target_idx = i
                    break
            
            if target_idx != -1:
                self.lb_img.selection_clear(0, "end")
                self.lb_img.selection_set(target_idx)
                self.lb_img.see(target_idx) # 滚动到可见
                self.on_select_img(None) # 触发加载
            else:
                self.jump_to_img = None # 没找到就清除，避免干扰

    def on_select_img(self, event):
        sel = self.lb_img.curselection()
        if not sel: return
        fname = self.lb_img.get(sel[0])
        src_entry, dst_entry = self.file_map.get(fname)
        if self.is_playing: self.toggle_play()
        try:
            self.curr_src_img = IMGFactory.open(BytesIO(src_entry.data))
            if dst_entry:
                self.curr_dst_img = IMGFactory.open(BytesIO(dst_entry.data))
            else:
                self.curr_dst_img = None
            count = len(self.curr_src_img.images)
            self.curr_frames_count = count
            self.scale_frame.config(from_=0, to=max(0, count-1), value=0)
            self.view_offset_x = 0
            self.view_offset_y = 0
            init_frame = 0
            if self.jump_to_img == fname and self.jump_to_frame is not None:
                if 0 <= self.jump_to_frame < count:
                    init_frame = self.jump_to_frame
                # 跳转一次后重置，防止用户手动切图时乱跳
                self.jump_to_img = None 
                
            self.scale_frame.set(init_frame)
            self.update_view(init_frame)
        except Exception as e: print(f"Error: {e}")

    def on_frame_change(self, val):
        self.update_view(int(float(val)))

    def update_view(self, idx):
        if not self.curr_src_img or idx >= self.curr_frames_count: return
        self.lbl_info.config(text=f"{idx+1}/{self.curr_frames_count}")
        
        def get_data(img_obj, index):
            if not img_obj or index >= len(img_obj.images): return None, 0, 0
            try: 
                frame = img_obj.images[index]
                # 【还原：诚实模式】
                # 直接读取当前帧的属性。
                # 如果它是引用帧且坐标为0，那就显示0（错位），这样您才能发现文件有问题。
                ox = getattr(frame, 'x', getattr(frame, 'pos_x', 0))
                oy = getattr(frame, 'y', getattr(frame, 'pos_y', 0))
                return img_obj.build(frame), ox, oy
            except: 
                return None, 0, 0
        
        im_l, x_l, y_l = get_data(self.curr_src_img, idx)
        # 【修改】只有在双屏模式下才加载右图
        im_r, x_r, y_r = (None, 0, 0)
        if not self.is_single_mode:
            im_r, x_r, y_r = get_data(self.curr_dst_img, idx)
        
        self._draw_canvas(self.cvs_l, im_l, x_l, y_l)
        if self.cvs_r:
            self._draw_canvas(self.cvs_r, im_r, x_r, y_r)

    def _draw_canvas(self, canvas, pil_img, off_x, off_y):
        canvas.delete("all")
        w, h = int(canvas.winfo_width()), int(canvas.winfo_height())
        cx, cy = w//2, h//2
        cx += self.view_offset_x
        cy += self.view_offset_y

        line_color = "#555" if self.bg_mode == 0 else ("#333" if self.bg_mode == 2 else "#666")
        bg_color = "#333"
        if self.bg_mode == 1: bg_color = "black"
        elif self.bg_mode == 2: bg_color = "white"
        
        canvas.config(bg=bg_color)
        canvas.create_line(cx, 0, cx, h, fill=line_color, dash=(4, 4))
        canvas.create_line(0, cy, w, cy, fill=line_color, dash=(4, 4))
        
        if pil_img:
            if self.boost_visibility.get():
                if pil_img.mode != "RGBA": pil_img = pil_img.convert("RGBA")
                r, g, b, a = pil_img.split()
                pil_img = Image.merge("RGB", (r, g, b))
            
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # 【修改点】核心逻辑：判断是否应用坐标
            if self.use_coords.get():
                # 应用坐标 (可能跑偏，但能看是否错位)
                draw_x = cx - off_x
                draw_y = cy - off_y
                canvas.create_image(draw_x, draw_y, image=tk_img, anchor="nw")
            else:
                # 强制居中 (不看坐标，只看图)
                canvas.create_image(cx, cy, image=tk_img, anchor="center")
            
            canvas.image = tk_img 

    def on_canvas_drag_start(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def on_canvas_drag(self, event):
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self.view_offset_x += dx
        self.view_offset_y += dy
        self.refresh_view()

    def toggle_bg(self):
        self.bg_mode = (self.bg_mode + 1) % 3
        self.refresh_view()
    def on_close(self):
        self.destroy()

def slice_sprite_sheet(src_img_path, temp_dir, frame_width, frame_count=18):
    try:
        # 【修改点1】强制转换为 RGBA，防止切片时丢失透明通道导致后续报错
        img = Image.open(src_img_path).convert("RGBA")
        
        w, h = img.size
        if frame_width <= 0: frame_width = w // frame_count
        processed_count = 0
        for i in range(frame_count):
            left = i * frame_width
            right = left + frame_width
            if left >= w: break
            crop_box = (left, 0, right, h)
            sub_img = img.crop(crop_box)
            save_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            sub_img.save(save_path)
            processed_count += 1
        return True, f"成功切割 {processed_count} 帧"
    except Exception as e:
        return False, f"切片失败: {e}"


# =========================================================================
# PART 2: 【幻色棱镜】核心逻辑 (NPK 处理)
# =========================================================================

PRESET_COLORS = {
    "🔥 火焰 (Fire)": ["#000000", "#640000", "#FF6400", "#FFDC00", "#FFFFFF"],
    "❄ 冰霜 (Ice)": ["#000000", "#00143C", "#0064C8", "#00FFFF", "#DCFFFF"],
    "🩸 血腥 (Blood)": ["#000000", "#3C0000", "#B40000", "#FF1400", "#FF5050"],
    "⚡ 雷电 (Blue)": ["#000000", "#0A0032", "#0032FF", "#64C8FF", "#FFFFFF"],
    "🟣 虚空 (Void)": ["#000000", "#1E003C", "#9600FF", "#FF64FF", "#FFE6FF"],
    "🟢 剧毒 (Poison)": ["#000000", "#32005A", "#8C00DC", "#FF00FF", "#FF8CFF"],
    "✨ 神圣 (Light)": ["#000000", "#50320A", "#C89632", "#FFF096", "#FFFFFF"],
    "🤖 科技 (Cyber)": ["#000000", "#001E1E", "#009696", "#00FFFF", "#C8FFFF"],
    "🌑 黑暗 (Dark)": ["#000000", "#140014", "#500078", "#0A0014"],
    "🔳 黑白 (Mono)": ["#000000", "#FFFFFF"],
    "🌸 粉金 (Pink Gold)": ["#000000", "#460020", "#FF5090", "#FFE0A0", "#FFFFF0"],
    "💖 冷艳粉 (Cool Rose)": ["#000000", "#300030", "#D04090", "#FFC0CB", "#FFFFFF"],
    "💖 玫瑰 (Rose Gold)": ["#000000", "#4B1E1E", "#D25F78", "#FFB4A0", "#FFF0E6"],
    "🦄 霓虹 (Neon Gold)": ["#000000", "#500050", "#FF0080", "#FFD700", "#FFFFFF"],
    "👻 全透明 (Hidden)": ["#000000"]
}

def normalize_hex(color_str):
    """
    校验并标准化 HEX 颜色代码
    输入: " f00 ", "#FF0000", "ff0000"
    输出: "#FF0000" 或 None (如果非法)
    """
    if not color_str: return None
    s = color_str.strip().replace("#", "").upper()
    
    # 处理缩写 (如 F00 -> FF0000)
    if len(s) == 3:
        s = "".join([c*2 for c in s])
        
    if len(s) != 6:
        return None
        
    # 检查字符是否合法
    try:
        int(s, 16)
        return "#" + s
    except ValueError:
        return None
        
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_lut_from_colors(hex_color_list, force_black_start=True):
    colors_hex = list(hex_color_list)
    if not colors_hex: colors_hex = ["#000000", "#FFFFFF"]
    if force_black_start:
        first_color = colors_hex[0].upper()
        if first_color not in ["#000000", "#000"]:
            colors_hex.insert(0, "#000000")
    colors = [hex_to_rgb(c) for c in colors_hex]
    num_colors = len(colors)
    lut = np.zeros((256, 3), dtype=np.uint8)
    if num_colors == 1:
        lut[:] = colors[0]
        return lut
    key_indices = np.linspace(0, 255, num_colors, dtype=int)
    for i in range(num_colors - 1):
        idx_start = key_indices[i]
        idx_end = key_indices[i+1]
        col_start = colors[i]
        col_end = colors[i+1]
        steps = idx_end - idx_start
        if steps <= 0: continue
        for step in range(steps):
            ratio = step / steps
            lut[idx_start + step] = [
                col_start[0] + (col_end[0] - col_start[0]) * ratio,
                col_start[1] + (col_end[1] - col_start[1]) * ratio,
                col_start[2] + (col_end[2] - col_start[2]) * ratio
            ]
    lut[255] = colors[-1]
    if key_indices[-1] < 255: lut[key_indices[-1]:] = colors[-1]
    if force_black_start: lut[0] = [0, 0, 0]
    return lut

def rgb_to_hsv_hue(rgb):
    r, g, b = rgb
    img = Image.new("RGB", (1,1), (r, g, b))
    h, s, v = img.convert("HSV").split()
    return h.getpixel((0,0))