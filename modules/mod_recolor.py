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

from common import Config, check_path_safety, get_resource_path,PRESET_COLORS,normalize_hex,hex_to_rgb,create_lut_from_colors,rgb_to_hsv_hue
HAS_PYDNFEX = False
try:
    from pydnfex.npk import NPK
    from pydnfex.img import IMGFactory, ImageLink
    HAS_PYDNFEX = True
except ImportError:
    print("⚠️ 警告: 未检测到库，【调色盘】功能将不可用。")
    
class ColorLogic:
    @staticmethod
    def analyze_colors(pil_img, max_count=256): # max_count 稍微改大一点，方便后续聚类
        """分析单张图片的颜色"""
        if not pil_img: return []
        img = pil_img.convert("RGBA")
        arr = np.array(img)
        pixels = arr.reshape(-1, 4)
        valid_mask = (pixels[:, 3] > 10)
        valid_pixels = pixels[valid_mask]
        if len(valid_pixels) == 0: return []
        unique_colors, counts = np.unique(valid_pixels, axis=0, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        top_indices = sorted_indices[:max_count]
        top_colors = unique_colors[top_indices]
        return [tuple(c) for c in top_colors[:, :3]] # 返回 RGB 元组列表

    @staticmethod
    def analyze_global(img_list, max_count=256, sample_size=10):
        """全局分析"""
        if not img_list: return []
        import random
        samples = img_list
        if len(img_list) > sample_size:
            samples = random.sample(img_list, sample_size)
        all_pixels = []
        for img in samples:
            arr = np.array(img.convert("RGBA"))
            pixels = arr.reshape(-1, 4)
            valid = pixels[pixels[:, 3] > 10]
            if len(valid) > 0:
                # 随机采样以提高性能
                if len(valid) > 2000:
                    indices = np.random.choice(len(valid), 2000, replace=False)
                    valid = valid[indices]
                all_pixels.append(valid)
        if not all_pixels: return []
        big_arr = np.vstack(all_pixels)
        unique_colors, counts = np.unique(big_arr, axis=0, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        return [tuple(c) for c in unique_colors[sorted_indices[:max_count]][:, :3]]

    @staticmethod
    def replace_colors(pil_img, mapping_dict, tolerance=10):
        """应用颜色替换 (保持不变)"""
        if not mapping_dict: return pil_img
        img = pil_img.convert("RGBA")
        arr = np.array(img)
        r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
        new_rgb = arr[:, :, :3].copy()
        new_a = a.copy() 
        
        # 优化：为了性能，这里依然保持原来的逻辑
        # 实际生产中，如果 mapping_dict 很大，建议反转逻辑（遍历像素而不是遍历字典），
        # 但考虑到聚类后字典反而变小了，所以维持现状即可。
        for src_rgb, dst_val in mapping_dict.items():
            sr, sg, sb = src_rgb
            diff = np.sqrt(
                (r.astype(np.int16) - sr)**2 + 
                (g.astype(np.int16) - sg)**2 + 
                (b.astype(np.int16) - sb)**2
            )
            mask = diff <= tolerance
            if dst_val == "TRANSPARENT":
                new_a[mask] = 0
            else:
                tr, tg, tb = dst_val
                new_rgb[mask, 0] = tr
                new_rgb[mask, 1] = tg
                new_rgb[mask, 2] = tb
        
        result_arr = np.dstack((new_rgb, new_a))
        return Image.fromarray(result_arr.astype(np.uint8))

    # --- 【新增】颜色聚类算法 ---
    @staticmethod
    def cluster_colors(raw_colors, threshold=10):
        """
        将相似颜色合并为簇。
        输入: raw_colors [(r,g,b), ...] (按频率排序)
        输出: 
           display_list: [(r,g,b), ...]  用于显示的代表色列表
           cluster_map:  { (r,g,b)_representative : [(r,g,b)_sub1, (r,g,b)_sub2, ...] }
        """
        if not raw_colors: return [], {}
        
        # 结果容器
        representatives = []
        cluster_map = {} # Key: 代表色, Value: [包含的原色列表]

        # 简单的贪婪聚类算法 (Greedy Clustering)
        # 因为 raw_colors 已经按频率排序，所以我们优先保留频率高的作为代表色
        for color in raw_colors:
            r1, g1, b1 = int(color[0]), int(color[1]), int(color[2])
            found_cluster = False
            
            # 尝试归入现有簇
            for rep in representatives:
                r2, g2, b2 = int(rep[0]), int(rep[1]), int(rep[2])
                # 计算欧氏距离
                dist = math.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
                
                if dist <= threshold:
                    cluster_map[rep].append(color)
                    found_cluster = True
                    break
            
            # 如果没找到相似的簇，则自己成为新的代表
            if not found_cluster:
                representatives.append(color)
                cluster_map[color] = [color]
                
        return representatives, cluster_map

class AdvancedColorDialog(tk.Toplevel):
    """自定义的高级颜色选择弹窗 (UI修复版)"""
    def __init__(self, parent, initial_color_hex, title="选择颜色"):
        super().__init__(parent)
        self.title(title)
        self.result = None 
        
        # 【修改点1】增大窗口尺寸，防止底部按钮被挤压
        w, h = 360, 300 
        x = parent.winfo_rootx() + 50
        y = parent.winfo_rooty() + 50
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.resizable(False, False)
        self.grab_set() 
        
        self.current_hex = initial_color_hex
        self.create_widgets()
        
    def create_widgets(self):
        # 1. 顶部预览区
        f_top = ttk.LabelFrame(self, text=" 颜色预览 ", padding=10)
        f_top.pack(fill="x", padx=10, pady=10)
        
        # 【修改点2】增大预览色块尺寸
        self.lbl_preview = tk.Label(f_top, bg=self.current_hex, width=15, height=4, relief="sunken", bd=2)
        self.lbl_preview.pack(side="left", padx=10)
        
        f_info = ttk.Frame(f_top)
        f_info.pack(side="left", fill="x", expand=True)
        ttk.Label(f_info, text="当前代码:").pack(anchor="w")
        self.lbl_code = ttk.Label(f_info, text=self.current_hex, font=("Consolas", 14, "bold"))
        self.lbl_code.pack(anchor="w", pady=5)

        # 2. 输入区
        f_input = ttk.Frame(self, padding=10)
        f_input.pack(fill="x", padx=5)
        
        f_hex = ttk.Frame(f_input)
        f_hex.pack(fill="x", pady=5)
        ttk.Label(f_hex, text="HEX修改:").pack(side="left")
        self.entry_hex = ttk.Entry(f_hex, font=("Consolas", 10))
        self.entry_hex.pack(side="left", fill="x", expand=True, padx=5)
        self.entry_hex.insert(0, self.current_hex)
        self.entry_hex.bind("<Return>", self.on_hex_enter)
        ttk.Button(f_hex, text="应用", width=6, command=self.on_hex_enter).pack(side="left")
        
        ttk.Button(f_input, text="🎨 打开系统调色板...", command=self.open_system_picker).pack(fill="x", pady=5)

        # 3. 底部按钮区
        # 【修改点3】增加底部 Frame 的高度和内边距，确保按钮显示完整
        f_btm = ttk.Frame(self, padding=15)
        f_btm.pack(fill="x", side="bottom")
        
        self.btn_trans = tk.Button(f_btm, text="👻 设为透明", bg="#ecf0f1", fg="red", 
                                   command=self.set_transparent, height=1)
        self.btn_trans.pack(side="left")
        
        ttk.Button(f_btm, text="确定", width=8, command=self.confirm).pack(side="right", padx=5)
        ttk.Button(f_btm, text="取消", width=8, command=self.destroy).pack(side="right")

    # ... (以下方法保持不变：open_system_picker, on_hex_enter, update_color, set_transparent, confirm) ...
    # 请保留原有的逻辑代码
    def open_system_picker(self):
        c = colorchooser.askcolor(color=self.current_hex)[1]
        if c: self.update_color(c)

    def on_hex_enter(self, event=None):
        code = self.entry_hex.get().strip()
        if not code.startswith("#"): code = "#" + code
        if len(code) != 7: return
        try:
            int(code[1:], 16)
            self.update_color(code)
        except: pass

    def update_color(self, hex_code):
        self.current_hex = hex_code
        self.lbl_preview.config(bg=hex_code, text="")
        self.lbl_code.config(text=hex_code)
        self.entry_hex.delete(0, "end")
        self.entry_hex.insert(0, hex_code)
        self.result = hex_to_rgb(hex_code)

    def set_transparent(self):
        self.result = "TRANSPARENT"
        self.lbl_preview.config(bg="white", text="透明", fg="gray")
        self.lbl_code.config(text="[透明]")
        self.destroy()

    def confirm(self):
        if self.result is None:
            self.result = hex_to_rgb(self.current_hex)
        self.destroy()


# =========================================================================
# 【新功能】调色盘面板组件
# =========================================================================
# =========================================================================
# 【新功能】调色盘面板组件 (UI 修改版：增加合并滑块)
# =========================================================================
# =========================================================================
# 【新功能】调色盘面板组件 (UI 修改版：支持筛选与反向高亮)
# =========================================================================


# =========================================================================
# 【新功能】调色盘面板组件 (终极版：多选 + 排序 + 渐变映射)
# =========================================================================
import colorsys 

# =========================================================================
# 【新功能】调色盘面板组件 (交互分离版：单击选中 / 双击修改)
# =========================================================================
class PalettePanel(ttk.LabelFrame):
    def __init__(self, parent, on_change_callback=None, on_scope_switch=None, on_highlight=None):
        super().__init__(parent, text=" 🎨 智能调色盘 ", padding=10)
        self.on_change_callback = on_change_callback
        self.on_scope_switch = on_scope_switch
        self.on_highlight = on_highlight
        
        self.current_scope = "global" 
        self.color_map = {} 
        self.raw_colors_cache = [] 
        self.cluster_data = {} 
        
        # --- 状态数据 ---
        self.selected_reps = set() # 被选中的代表色集合
        self.last_click_index = -1 
        self.sort_mode = tk.StringVar(value="count")
        
        # --- 控件引用 ---
        self.buttons = []      # 按钮对象列表
        self.display_reps = [] # 对应按钮的颜色数据列表

        self.create_ui()
        
    def create_ui(self):
        # 1. 顶部：模式切换
        f_scope = ttk.Frame(self)
        f_scope.pack(fill="x", pady=(0, 5))
        self.btn_global = tk.Button(f_scope, text="🌍 全局", bg="#3498db", fg="white", 
                                    relief="sunken", command=lambda: self.switch_scope("global"))
        self.btn_global.pack(side="left", fill="x", expand=True, padx=1)
        self.btn_single = tk.Button(f_scope, text="🖼️ 单帧", bg="#f0f0f0", fg="black", 
                                    relief="raised", command=lambda: self.switch_scope("single"))
        self.btn_single.pack(side="left", fill="x", expand=True, padx=1)
        self.lbl_status = tk.Label(self, text="状态: 编辑所有帧 (通用规则)", bg="#ecf0f1", fg="#555")#, font=("Arial", 8))
        self.lbl_status.pack(fill="x", pady=2)
        # 2. 参数控制
        f_ctrl = ttk.LabelFrame(self, text=" 参数控制 ", padding=5)
        f_ctrl.pack(fill="x", pady=5)
        
        f_merge = ttk.Frame(f_ctrl)
        f_merge.pack(fill="x", pady=1)
        ttk.Label(f_merge, text="合并:", width=5).pack(side="left")
        self.var_merge = tk.IntVar(value=25) 
        self.scale_merge = ttk.Scale(f_merge, from_=0, to=100, variable=self.var_merge, orient="horizontal")
        self.scale_merge.pack(side="left", fill="x", expand=True)
        self.scale_merge.bind("<ButtonRelease-1>", lambda e: self.re_cluster_and_draw())

        f_tol = ttk.Frame(f_ctrl)
        f_tol.pack(fill="x", pady=1)
        ttk.Label(f_tol, text="容差:", width=5).pack(side="left")
        self.var_tolerance = tk.IntVar(value=10)
        self.scale_tol = ttk.Scale(f_tol, from_=0, to=100, variable=self.var_tolerance, orient="horizontal")
        self.scale_tol.pack(side="left", fill="x", expand=True)
        self.scale_tol.bind("<ButtonRelease-1>", lambda e: self.trigger_update())

        # 3. 工具栏
        f_tools = ttk.Frame(self)
        f_tools.pack(fill="x", pady=(5, 2))
        ttk.Label(f_tools, text="排序:").pack(side="left")
        cb_sort = ttk.Combobox(f_tools, textvariable=self.sort_mode, state="readonly", width=8)#, font=("Arial", 8))
        cb_sort['values'] = ("默认(数量)", "按色相(H)", "按亮度(V)")
        cb_sort.current(0)
        cb_sort.bind("<<ComboboxSelected>>", lambda e: self.re_cluster_and_draw())
        cb_sort.pack(side="left", padx=2)
        ttk.Button(f_tools, text="全选", width=4, command=self.select_all).pack(side="right", padx=1)
        ttk.Button(f_tools, text="反选", width=4, command=self.select_invert).pack(side="right", padx=1)

        # 4. 颜色网格 (滚动区域)
        f_grid = ttk.Frame(self)
        f_grid.pack(fill="both", expand=True)
        tk.Label(f_grid, text="💡 单击选中 | 双击修改 | 右键高亮", font=("Arial", 7), fg="#555", bg="#e0e0e0").pack(fill="x")
        
        self.canvas = tk.Canvas(f_grid, bg="#e0e0e0", height=200)
        sb = ttk.Scrollbar(f_grid, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.inner_frame = tk.Frame(self.canvas, bg="#e0e0e0")
        self.canvas.create_window((0,0), window=self.inner_frame, anchor="nw")
        self.inner_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # 5. 批量操作区
        self.f_batch = ttk.LabelFrame(self, text=" 批量操作 (选中0个) ", padding=5)
        self.f_batch.pack(fill="x", pady=5)
        
        b1 = tk.Button(self.f_batch, text="🌈 渐变映射", bg="#9b59b6", fg="white", font=("Arial", 9, "bold"),
                       command=self.batch_gradient_map)
        b1.pack(side="left", fill="x", expand=True, padx=1)
        b2 = tk.Button(self.f_batch, text="👻 透明", bg="white", fg="red", relief="groove",
                       command=self.batch_set_transparent)
        b2.pack(side="left", fill="x", expand=True, padx=1)
        b3 = tk.Button(self.f_batch, text="🎨 统一", bg="#34495e", fg="white",
                       command=self.batch_set_flat_color)
        b3.pack(side="left", fill="x", expand=True, padx=1)
        
        b4 = tk.Button(self.f_batch, text="↩️ 还原", bg="#95a5a6", fg="white",
                       command=self.batch_reset_selection)
        b4.pack(side="left", fill="x", expand=True, padx=1)
        
        ttk.Button(self, text="重置所有映射", command=self.reset_map).pack(fill="x", pady=(0,5))

    # --- 逻辑方法 ---
    def switch_scope(self, mode):
        if self.current_scope == mode: return
        self.current_scope = mode
        if mode == "global":
            self.btn_global.config(bg="#3498db", fg="white", relief="sunken")
            self.btn_single.config(bg="#f0f0f0", fg="black", relief="raised")
            self.lbl_status.config(text="状态: 编辑所有帧 (通用规则)", bg="#ecf0f1", fg="#555")
        else:
            self.btn_global.config(bg="#f0f0f0", fg="black", relief="raised")
            self.btn_single.config(bg="#e67e22", fg="white", relief="sunken")
            self.lbl_status.config(text="状态: 仅编辑当前帧 (覆盖规则)", bg="#fff3cd", fg="#d35400")
        if self.on_scope_switch: self.on_scope_switch(mode)

    def refresh_colors(self, raw_colors, current_map):
        self.raw_colors_cache = raw_colors 
        self.color_map = current_map.copy()
        self.selected_reps.clear()
        self.re_cluster_and_draw()

    # --- 核心绘制 ---
    def re_cluster_and_draw(self):
        """重新计算聚类并生成按钮控件 (耗时操作)"""
        for b in self.buttons: b.destroy()
        self.buttons = []
        self.display_reps = [] # 清空显示列表
        
        # 1. 聚类
        threshold = self.var_merge.get()
        reps, self.cluster_data = ColorLogic.cluster_colors(self.raw_colors_cache, threshold)
        
        # 2. 排序
        mode = self.sort_mode.get()
        if "色相" in mode:
            reps.sort(key=lambda c: colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)[0])
        elif "亮度" in mode:
            reps.sort(key=lambda c: colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)[2])
            
        self.display_reps = reps 
        
        # 3. 绘制
        cols = 6
        for i, rep_rgb in enumerate(reps):
            row, col = divmod(i, cols)
            
            # Frame 容器 (充当边框)
            # 默认状态：背景色和底色一致(#e0e0e0)，padding=1 (稍微留点缝隙)
            f = tk.Frame(self.inner_frame, padx=1, pady=1, bg="#e0e0e0")
            f.grid(row=row, column=col, padx=1, pady=1) # grid 自身的间隔
            
            sub_colors = self.cluster_data[rep_rgb]
            f.cluster_colors = sub_colors 
            
            # Button (初始外观)
            # 【关键修改】不再在这里设置边框样式，全部交给 update_selection_visuals
            hex_c = '#%02x%02x%02x' % rep_rgb
            
            # takefocus=False 很重要，防止点击后产生虚线框干扰视觉
            btn = tk.Button(f, bg=hex_c, width=4, height=1, relief="raised", bd=1, takefocus=False)
            
            # 绑定事件
            btn.bind("<Button-1>", lambda e, idx=i: self.handle_click(idx, e))
            btn.bind("<Double-1>", lambda e, idx=i: self.handle_double_click(idx, e))
            btn.bind("<Button-3>", lambda e, s=sub_colors: self.trigger_highlight(s))
            btn.bind("<Leave>", lambda e: self.trigger_highlight(None))
            
            # 【关键修改】fill="both", expand=True 确保按钮填满 Frame
            # 这样 Frame 的背景色就变成了完美的边框
            btn.pack(fill="both", expand=True)
            self.buttons.append(btn)
            
        # 刷新一次视觉状态
        self.update_selection_visuals()

    def update_selection_visuals(self):
        """【优化版】通过 Frame 背景色模拟边框，解决显示不全的 Bug"""
        for i, btn in enumerate(self.buttons):
            rep_rgb = self.display_reps[i]
            
            # 获取按钮的父容器 (Frame)
            frame = btn.master
            
            # 1. 检查是否被修改 (决定按钮颜色和凹凸)
            sub_colors = self.cluster_data[rep_rgb]
            target = None
            for sub_c in sub_colors:
                if sub_c in self.color_map:
                    target = self.color_map[sub_c]
                    break 
            
            # 更新颜色显示
            if target:
                if target == "TRANSPARENT":
                    btn.config(bg="#ffffff", text="透", fg="#ccc")
                else:
                    new_hex = '#%02x%02x%02x' % target
                    btn.config(bg=new_hex, text="", fg="black")
                
                # 如果被修改了，按钮本身保持凹陷，表示“已处理”
                btn.config(relief="sunken", bd=1)
            else:
                orig_hex = '#%02x%02x%02x' % rep_rgb
                btn.config(bg=orig_hex, text="", fg="black")
                btn.config(relief="raised", bd=1)

            # 2. 检查选中状态 (决定 Frame 边框)
            if rep_rgb in self.selected_reps:
                # 【选中】：Frame 变黑 (或变蓝)，Padding 变大
                # 这样按钮周围就会出现一圈完美的粗边框
                frame.config(bg="#222222", padx=2, pady=2) 
            else:
                # 【未选中】：Frame 变回底色，Padding 变小
                frame.config(bg="#e0e0e0", padx=1, pady=1)

        self.update_batch_ui()

    # --- 交互逻辑 ---
    def handle_click(self, idx, event):
        """单击：仅处理选中逻辑"""
        rep_rgb = self.display_reps[idx]
        
        # Shift 连选
        if event.state & 0x0001: 
            if self.last_click_index != -1:
                start = min(self.last_click_index, idx)
                end = max(self.last_click_index, idx)
                for k in range(start, end + 1):
                    self.selected_reps.add(self.display_reps[k])
            else:
                self.selected_reps.add(rep_rgb)
                
        # Ctrl 加选
        elif event.state & 0x0004: 
            if rep_rgb in self.selected_reps:
                self.selected_reps.remove(rep_rgb)
            else:
                self.selected_reps.add(rep_rgb)
            self.last_click_index = idx
            
        # 普通单击 (单选)
        else:
            self.selected_reps.clear()
            self.selected_reps.add(rep_rgb)
            self.last_click_index = idx

        # 仅更新视觉，不重建，保证流畅和双击判定
        self.update_selection_visuals()

    def handle_double_click(self, idx, event):
        """双击：触发修改逻辑"""
        rep_rgb = self.display_reps[idx]
        
        # 如果双击的这个没被选中（防御性编程），先选中它
        if rep_rgb not in self.selected_reps:
            self.selected_reps.add(rep_rgb)
            self.update_selection_visuals()
            
        # 触发修改 (传入当前双击颜色的作为初始值)
        self.open_modify_dialog(initial_rgb=rep_rgb)

    def select_all(self):
        self.selected_reps = set(self.display_reps)
        self.update_selection_visuals()

    def select_invert(self):
        current = self.selected_reps
        all_reps = set(self.display_reps)
        self.selected_reps = all_reps - current
        self.update_selection_visuals()

    def update_batch_ui(self):
        count = len(self.selected_reps)
        self.f_batch.config(text=f" 批量操作 (已选 {count} 个) ")
        state = "normal" if count > 0 else "disabled"
        for child in self.f_batch.winfo_children():
            try: child.config(state=state)
            except: pass

    # --- 修改逻辑 ---
    def open_modify_dialog(self, initial_rgb):
        """打开修改颜色弹窗"""
        # 确定弹窗初始显示的颜色
        # 尝试查找当前是否有映射值
        sub_colors = self.cluster_data[initial_rgb]
        current_target = None
        for sub_c in sub_colors:
            if sub_c in self.color_map:
                current_target = self.color_map[sub_c]
                break
        
        initial_hex = '#%02x%02x%02x' % initial_rgb
        if current_target and current_target != "TRANSPARENT":
             initial_hex = '#%02x%02x%02x' % current_target
        
        title_str = "修改颜色" if len(self.selected_reps) == 1 else f"批量修改 {len(self.selected_reps)} 个颜色"
        dlg = AdvancedColorDialog(self, initial_hex, title=title_str)
        self.wait_window(dlg)
        
        if dlg.result is not None:
            self.apply_to_selection(dlg.result)

    def apply_to_selection(self, target_value):
        """将目标值应用到所有选中的簇"""
        for rep in self.selected_reps:
            sub_colors = self.cluster_data.get(rep, [])
            for sub_c in sub_colors:
                self.color_map[sub_c] = target_value
        
        self.update_selection_visuals()
        self.trigger_update()

    def batch_set_transparent(self):
        if not self.selected_reps: return
        self.apply_to_selection("TRANSPARENT")

    def batch_set_flat_color(self):
        if not self.selected_reps: return
        c = colorchooser.askcolor(title="选择统一颜色")[1]
        if c:
            self.apply_to_selection(hex_to_rgb(c))
            
    def batch_reset_selection(self):
        """将选中的颜色组还原为初始状态 (从映射表中删除)"""
        if not self.selected_reps: return
        
        # 遍历所有选中的代表色
        for rep in self.selected_reps:
            # 获取该代表色对应的所有原始近似色
            sub_colors = self.cluster_data.get(rep, [])
            
            for sub_c in sub_colors:
                # 如果这个颜色在映射表里，就删掉它
                if sub_c in self.color_map:
                    del self.color_map[sub_c]
        
        # 刷新视图 (按钮会变回原来的颜色，且变为凸起状态)
        self.update_selection_visuals()
        # 触发更新 (通知主界面重绘预览图，并检查是否要移除左侧列表的红字)
        self.trigger_update()
        
    def batch_gradient_map(self):
        if len(self.selected_reps) < 2:
            messagebox.showinfo("提示", "渐变映射至少需要选中 2 个颜色。")
            return
        
        c1 = colorchooser.askcolor(title="步骤 1/2: 选择暗部颜色 (Start)")[1]
        if not c1: return
        c2 = colorchooser.askcolor(title="步骤 2/2: 选择亮部颜色 (End)")[1]
        if not c2: return
        
        start_rgb = hex_to_rgb(c1)
        end_rgb = hex_to_rgb(c2)
        
        # 按亮度排序选中的颜色
        sorted_reps = sorted(list(self.selected_reps), 
                             key=lambda c: 0.299*c[0] + 0.587*c[1] + 0.114*c[2])
        n = len(sorted_reps)
        
        for i, rep in enumerate(sorted_reps):
            t = i / (n - 1) if n > 1 else 0
            nr = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t)
            ng = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t)
            nb = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t)
            
            # 应用到单个簇 (不调用 apply_to_selection，避免重复刷新)
            target_val = (nr, ng, nb)
            for sub_c in self.cluster_data.get(rep, []):
                self.color_map[sub_c] = target_val
                
        self.update_selection_visuals()
        self.trigger_update()
        messagebox.showinfo("成功", f"已应用渐变映射到 {n} 个颜色组。")

    # --- 筛选与重置 ---
    def reset_map(self):
        self.color_map = {}
        self.selected_reps.clear()
        self.update_selection_visuals() # 此时颜色恢复默认
        self.trigger_update()

    def filter_by_raw_colors(self, target_raw_colors):
        if not target_raw_colors: return
        target_set = set(target_raw_colors)
        visible_count = 0
        for child in self.inner_frame.winfo_children():
            if not isinstance(child, tk.Frame): continue
            if not hasattr(child, 'cluster_colors'): continue
            cluster_set = set(child.cluster_colors)
            if not cluster_set.isdisjoint(target_set):
                child.grid()
                visible_count += 1
            else:
                child.grid_remove()
        if visible_count == 0:
            self.reset_filter()

    def reset_filter(self):
        for child in self.inner_frame.winfo_children():
            if isinstance(child, tk.Frame):
                child.grid()
        self.trigger_highlight(None)

    def trigger_highlight(self, colors):
        if self.on_highlight:
            self.on_highlight(colors)

    def trigger_update(self):
        if self.on_change_callback:
            self.on_change_callback(self.color_map, self.var_tolerance.get())

# =========================================================================
# 【新页面】专业像素换色页 (带 IMG 列表版)
# =========================================================================
# =========================================================================
# 【新页面】专业像素换色页 (带 IMG 列表 + 框选交互)
# =========================================================================
# =========================================================================
# 【新页面】专业像素换色页 (带 IMG 列表 + 框选交互 + 批量导出)
# =========================================================================
# =========================================================================
# 【新页面】专业像素换色页 (多选 + 排序 + 渐变 + 状态管理 + 快速导航)
# =========================================================================
class AdvancedRecolorPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        # --- 读取上次保存的路径 ---
        last_path = Config.get("last_recolor_npk_path", "")
        self.input_path = tk.StringVar(value=last_path)
        
        # --- 数据存储 ---
        self.current_npk = None   
        self.all_npk_files = [] # 【新增】缓存所有文件列表，方便筛选切换
        self.current_img_data = None 
        self.current_img_name = None 
        
        self.frames = []          
        
        # --- 当前编辑状态 ---
        self.global_map = {}      
        self.frame_overrides = {} 
        self.current_frame_idx = 0
        
        # --- 多文件修改暂存区 ---
        self.modified_cache = {} 
        
        # --- 播放控制 ---
        self.is_playing = False
        self.play_job = None
        
        # --- 交互状态 ---
        self.rect_start = None 
        self.rect_id = None    
        self.preview_img_cache = None 
        self.tk_img_cache = None
        self.var_show_modified = tk.BooleanVar(value=False) # 【新增】筛选变量

        self.create_widgets()
        
        # 界面加载完毕后，尝试自动加载上次的文件
        #self.after(200, self.auto_load_initial_file)
        self.is_first_show = True 
    def on_page_show(self):
        """当页面被显示时调用"""
        if self.is_first_show:
            self.is_first_show = False
            # 只有第一次切换到这个页面时，才尝试加载上次的文件
            # 这样弹窗就只会在用户点进来的时候显示
            self.auto_load_initial_file()
    def create_widgets(self):
        # 标题
        tk.Label(self, text="指定色替换", font=("微软雅黑", 16, "bold"), fg="#333").pack(pady=10)
        
        # 1. 顶部：文件来源
        f_top = ttk.LabelFrame(self, text=" 文件来源 ", padding=5)
        f_top.pack(fill="x", padx=10)
        
        self.entry_path = ttk.Entry(f_top, textvariable=self.input_path)
        self.entry_path.pack(side="left", fill="x", expand=True)
        ttk.Button(f_top, text="📂 打开 NPK/IMG...", command=self.load_file).pack(side="left", padx=5)

        # 2. 主工作区
        paned = tk.PanedWindow(self, orient="horizontal", sashwidth=5, bg="#ccc")
        paned.pack(fill="both", expand=True, padx=10, pady=5)
        
        # --- [左侧] 文件列表 ---
        f_list = ttk.Frame(paned)
        paned.add(f_list, width=220)
        
        # 【新增】筛选复选框
        f_filter = ttk.Frame(f_list)
        f_filter.pack(fill="x", pady=(0, 2))
        ttk.Label(f_filter, text="IMG 列表:", font=("微软雅黑", 9, "bold")).pack(side="left")
        ttk.Checkbutton(f_filter, text="只看已修", variable=self.var_show_modified, 
                        command=self.refresh_tree_list).pack(side="right")
        
        self.tree = ttk.Treeview(f_list, show="tree", selectmode="browse")
        sb_tree = ttk.Scrollbar(f_list, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb_tree.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb_tree.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_img_selected)
        
        # 【方案一：变色】配置红色标记
        self.tree.tag_configure("modified", foreground="red", font=("Consolas", 9, "bold"))

        # --- [中间] 预览 ---
        f_center = ttk.Frame(paned)
        paned.add(f_center, width=500)
        
        tk.Label(f_center, text="🖱️ 拖拽框选 -> 筛选颜色 | 右键空白 -> 重置", bg="#eee").pack(fill="x")
        
        self.cvs = tk.Canvas(f_center, bg="#333", highlightthickness=0, cursor="crosshair")
        self.cvs.pack(fill="both", expand=True)
        
        self.cvs.bind("<ButtonPress-1>", self.on_canvas_press)
        self.cvs.bind("<B1-Motion>", self.on_canvas_drag)
        self.cvs.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.cvs.bind("<Button-3>", lambda e: self.palette.reset_filter()) 
        
        f_ctrl = ttk.Frame(f_center, padding=5)
        f_ctrl.pack(fill="x")
        self.btn_play = ttk.Button(f_ctrl, text="▶", width=5, command=self.toggle_play)
        self.btn_play.pack(side="left")
        self.scale_time = ttk.Scale(f_ctrl, from_=0, to=1, orient="horizontal", command=self.on_seek)
        self.scale_time.pack(side="left", fill="x", expand=True, padx=10)
        self.lbl_frame = ttk.Label(f_ctrl, text="0/0", width=8)
        self.lbl_frame.pack(side="left")

        # --- [右侧] 调色盘 ---
        f_right = ttk.Frame(paned)
        paned.add(f_right, width=320)
        
        self.palette = PalettePanel(f_right, 
                                    on_change_callback=self.on_palette_change,
                                    on_scope_switch=self.on_scope_switch,
                                    on_highlight=self.highlight_pixels)
        self.palette.pack(fill="both", expand=True)

        # 3. 底部导出与导航
        f_btm = ttk.Frame(self, padding=10)
        f_btm.pack(fill="x")
        
        # 【方案三：点击跳转】
        self.lbl_cache_info = tk.Label(f_btm, text="暂存修改: 0 个文件", fg="blue", cursor="hand2", font=("微软雅黑", 9, "underline"))
        self.lbl_cache_info.pack(side="left", padx=5)
        self.lbl_cache_info.bind("<Button-1>", self.show_nav_popup) # 绑定点击事件
        
        self.btn_export = ttk.Button(f_btm, text="💾 保存并导出 NPK (应用所有修改)", command=self.export_result)
        self.btn_export.pack(side="right", fill="x", expand=True, ipady=5)

    def auto_load_initial_file(self):
        p = self.input_path.get()
        if p and os.path.exists(p):
            self.process_file_content(p)

    def load_file(self):
        p = filedialog.askopenfilename(filetypes=[("NPK/IMG Files", "*.npk;*.img")])
        if not p: return
        self.input_path.set(p)
        Config.set("last_recolor_npk_path", p)
        self.process_file_content(p)

    def process_file_content(self, p):
        # 清理状态
        self.tree.delete(*self.tree.get_children())
        self.frames = []
        self.global_map = {}
        self.frame_overrides = {}
        self.modified_cache = {} 
        self.current_img_name = None
        self.all_npk_files = [] # 清空缓存列表
        self.update_cache_label()
        self.cvs.delete("all")
        self.current_npk = None
        
        try:
            if p.lower().endswith(".npk"):
                self.current_npk = NPK.open(open(p, 'rb'))
                try: self.current_npk.load_all() 
                except: pass 
                
                # 收集所有 IMG 文件
                for f in self.current_npk.files:
                    if f.name.lower().endswith(".img"):
                        self.all_npk_files.append(f.name)
                
                if not self.all_npk_files: 
                    messagebox.showinfo("提示", "该 NPK 中没有找到 .img 文件")
                
                # 填充 Treeview
                self.refresh_tree_list()
                
            elif p.lower().endswith(".img"):
                name = os.path.basename(p)
                self.all_npk_files.append(name)
                self.tree.insert("", "end", iid=name, text=name, values=("SINGLE_FILE",))
                self.load_frames_from_path(p)
            
            # 恢复 JSON 状态
            json_path = self._get_json_path(p)
            if os.path.exists(json_path):
                if messagebox.askyesno("恢复进度", f"检测到该文件有历史修改记录。\n是否恢复上次的编辑状态？"):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            raw_data = json.load(f)
                        self.modified_cache = self._deserialize_cache(raw_data)
                        self.update_cache_label()
                        self.refresh_tree_list() # 刷新变色状态
                        print(f"已恢复 {len(self.modified_cache)} 个文件的修改记录")
                    except Exception as e:
                        messagebox.showerror("错误", f"读取历史记录失败:\n{e}")

        except Exception as e: 
            messagebox.showerror("错误", f"文件加载失败: {e}")

    # --- 【核心】刷新列表 (实现方案一变色 & 方案二筛选) ---
    def refresh_tree_list(self):
        # 1. 清空现有
        self.tree.delete(*self.tree.get_children())
        
        # 2. 决定显示哪些文件
        is_filter_on = self.var_show_modified.get()
        
        for name in self.all_npk_files:
            is_modified = name in self.modified_cache
            
            # 如果开启了筛选，且未修改，则跳过
            if is_filter_on and not is_modified:
                continue
            
            # 3. 插入节点 (使用文件名作为 iid，方便查找)
            # 如果已修改，添加 'modified' 标签(变红)
            tags = ("modified",) if is_modified else ()
            
            self.tree.insert("", "end", iid=name, text=name, tags=tags)

    # --- 【方案三】点击标签弹出导航窗口 ---
    def show_nav_popup(self, event):
        if not self.modified_cache:
            return # 没东西就不弹了
            
        # 创建弹窗
        top = tk.Toplevel(self)
        top.title("快速跳转 (双击选择)")
        top.geometry("400x300")
        
        lb = tk.Listbox(top, font=("Consolas", 10))
        lb.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 填充修改过的文件
        for name in self.modified_cache.keys():
            lb.insert("end", name)
            
        # 双击跳转
        def jump(e):
            sel = lb.curselection()
            if not sel: return
            target_name = lb.get(sel[0])
            
            # 1. 确保目标在列表中可见 (取消筛选)
            if self.var_show_modified.get():
                self.var_show_modified.set(False)
                self.refresh_tree_list()
                
            # 2. 选中并滚动
            if self.tree.exists(target_name):
                self.tree.selection_set(target_name)
                self.tree.see(target_name)
                # 主动触发加载逻辑
                self.on_img_selected(None) 
                
            top.destroy()
            
        lb.bind("<Double-1>", jump)

    def save_current_state_to_cache(self):
        """保存状态并更新列表颜色 (修复版：支持取消修改状态)"""
        if not self.current_img_name: return

        # 检查是否还有有效的修改
        # 注意：这里判断字典是否为空
        has_modification = bool(self.global_map) or bool(self.frame_overrides)

        if has_modification:
            # 有修改 -> 存入/更新缓存
            self.modified_cache[self.current_img_name] = {
                "global": self.global_map.copy(),
                "overrides": self.frame_overrides.copy(),
                "tolerance": self.palette.var_tolerance.get()
            }
            # 更新列表 UI：变红
            if self.tree.exists(self.current_img_name):
                self.tree.item(self.current_img_name, tags=("modified",))
        else:
            # 无修改 (被重置了) -> 从缓存中移除
            if self.current_img_name in self.modified_cache:
                del self.modified_cache[self.current_img_name]
            
            # 更新列表 UI：恢复默认颜色 (移除 tags)
            if self.tree.exists(self.current_img_name):
                self.tree.item(self.current_img_name, tags=())

        # 更新底部计数文本
        self.update_cache_label()

    def update_cache_label(self):
        count = len(self.modified_cache)
        self.lbl_cache_info.config(text=f"暂存修改: {count} 个文件")

    # --- 各种 JSON 辅助函数 ---
    def _get_json_path(self, npk_path):
        dir_name, file_name = os.path.split(npk_path)
        json_dir = os.path.join(dir_name, "npk_change_json")
        if not os.path.exists(json_dir): os.makedirs(json_dir)
        return os.path.join(json_dir, file_name + ".json")

    def _serialize_cache(self):
        serializable_data = {}
        for img_name, data in self.modified_cache.items():
            s_global = {}
            for k, v in data["global"].items():
                k_str = f"{k[0]},{k[1]},{k[2]}"
                s_global[k_str] = v 
            s_overrides = {}
            for frame_idx, rules in data["overrides"].items():
                s_rules = {}
                for k, v in rules.items():
                    k_str = f"{k[0]},{k[1]},{k[2]}"
                    s_rules[k_str] = v
                s_overrides[str(frame_idx)] = s_rules 
            serializable_data[img_name] = {
                "global": s_global,
                "overrides": s_overrides,
                "tolerance": data.get("tolerance", 10)
            }
        return serializable_data

    def _deserialize_cache(self, json_data):
        restored_cache = {}
        try:
            for img_name, data in json_data.items():
                r_global = {}
                for k_str, v in data["global"].items():
                    r, g, b = map(int, k_str.split(","))
                    val = tuple(v) if isinstance(v, list) else v
                    r_global[(r, g, b)] = val
                r_overrides = {}
                for f_idx_str, rules in data["overrides"].items():
                    r_rules = {}
                    for k_str, v in rules.items():
                        r, g, b = map(int, k_str.split(","))
                        val = tuple(v) if isinstance(v, list) else v
                        r_rules[(r, g, b)] = val
                    r_overrides[int(f_idx_str)] = r_rules 
                restored_cache[img_name] = {
                    "global": r_global,
                    "overrides": r_overrides,
                    "tolerance": data.get("tolerance", 10)
                }
            return restored_cache
        except Exception as e:
            print(f"JSON 解析失败: {e}")
            return {}

    def save_project_state_to_disk(self):
        self.save_current_state_to_cache()
        if not self.modified_cache: return
        npk_path = self.input_path.get()
        if not npk_path: return
        json_path = self._get_json_path(npk_path)
        data_to_save = self._serialize_cache()
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2)
            print(f"工程状态已保存: {json_path}")
        except Exception as e:
            print(f"保存 JSON 失败: {e}")

    # --- NPK 操作逻辑 ---
    def on_img_selected(self, event):
        sel = self.tree.selection()
        if not sel: return
        # 使用 iid (也就是文件名)
        item_text = sel[0] 
        
        self.save_current_state_to_cache()
        
        if self.current_npk:
            target_file = None
            for f in self.current_npk.files:
                if f.name == item_text:
                    target_file = f
                    break
            
            if target_file:
                self.current_img_data = target_file
                self.current_img_name = item_text
                self.load_frames_from_memory(target_file.data)
                
                if item_text in self.modified_cache:
                    cache = self.modified_cache[item_text]
                    self.global_map = cache["global"]
                    self.frame_overrides = cache["overrides"]
                    self.palette.var_tolerance.set(cache.get("tolerance", 10))
                    self.on_scope_switch(self.palette.current_scope)
                    self.update_preview()
                else:
                    pass

    def load_frames_from_memory(self, data_bytes):
        try:
            img_obj = IMGFactory.open(BytesIO(data_bytes))
            self.frames = []
            if self.is_playing: self.toggle_play()
            
            success_count = 0
            for i in range(len(img_obj.images)):
                try:
                    frame = img_obj.images[i]
                    current_frame_to_use = frame
                    if isinstance(frame, ImageLink):
                        target_idx = -1
                        if hasattr(frame, '_image'): target_idx = frame._image
                        elif hasattr(frame, 'link'): target_idx = frame.link
                        elif hasattr(frame, 'target'): target_idx = frame.target
                        if isinstance(target_idx, int) and 0 <= target_idx < len(img_obj.images):
                            current_frame_to_use = img_obj.images[target_idx]
                    
                    pil_img = img_obj.build(current_frame_to_use)
                    self.frames.append(pil_img)
                    success_count += 1
                except Exception as frame_err:
                    print(f"Frame {i} load err: {frame_err}")
                    self.frames.append(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))

            if success_count == 0:
                raise Exception("没有帧解析成功")

            self.init_editor_state()
        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("解析错误", f"无法解析 IMG: {e}")

    def load_frames_from_path(self, path):
        try:
            with open(path, 'rb') as f:
                self.load_frames_from_memory(f.read())
            self.current_img_name = os.path.basename(path)
        except Exception as e:
            messagebox.showerror("错误", f"读取文件失败: {e}")

    def init_editor_state(self):
        self.global_map = {}
        self.frame_overrides = {}
        self.current_frame_idx = 0
        count = len(self.frames)
        self.scale_time.config(to=max(0, count-1), value=0)
        self.lbl_frame.config(text=f"1/{count}")
        self.palette.switch_scope("global") 
        self.on_scope_switch("global")      
        self.update_preview()

    # ... (on_scope_switch, on_palette_change, on_seek, toggle_play, _loop, 
    #      get_img_coords, on_canvas_press, on_canvas_drag, on_canvas_release, 
    #      highlight_pixels, draw_image_on_canvas, update_preview, export_result, run_export_thread 
    #      这些方法保持不变，请务必保留) ...
    # 为了代码简洁，这里不再重复粘贴未修改的方法
    
    def on_scope_switch(self, mode):
        if not self.frames: return
        colors = []
        current_map = {}
        if mode == "global":
            colors = ColorLogic.analyze_global(self.frames)
            current_map = self.global_map
        else:
            colors = ColorLogic.analyze_colors(self.frames[self.current_frame_idx])
            current_map = self.frame_overrides.get(self.current_frame_idx, {})
        self.palette.refresh_colors(colors, current_map)

    def on_palette_change(self, new_map, tol):
        mode = self.palette.current_scope
        if mode == "global":
            self.global_map = new_map
        else:
            self.frame_overrides[self.current_frame_idx] = new_map
        
        self.update_preview()
        
        # 【新增】每次调色盘变动，实时更新缓存状态和左侧列表颜色
        self.save_current_state_to_cache()

    def on_seek(self, val):
        idx = int(float(val))
        if idx != self.current_frame_idx:
            self.current_frame_idx = idx
            self.lbl_frame.config(text=f"{idx+1}/{len(self.frames)}")
            if self.palette.current_scope == "single":
                self.on_scope_switch("single") 
            self.update_preview()

    def get_img_coords(self, canvas_x, canvas_y):
        if not self.preview_img_cache: return None
        cvs_w, cvs_h = self.cvs.winfo_width(), self.cvs.winfo_height()
        img_w, img_h = self.preview_img_cache.size
        offset_x = (cvs_w - img_w) // 2
        offset_y = (cvs_h - img_h) // 2
        x = canvas_x - offset_x
        y = canvas_y - offset_y
        if 0 <= x < img_w and 0 <= y < img_h:
            return int(x), int(y)
        return None

    def on_canvas_press(self, event):
        if not self.frames: return
        self.rect_start = (event.x, event.y)
        if self.rect_id: self.cvs.delete(self.rect_id)
        self.rect_id = self.cvs.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

    def on_canvas_drag(self, event):
        if not self.rect_start: return
        self.cvs.coords(self.rect_id, self.rect_start[0], self.rect_start[1], event.x, event.y)

    def on_canvas_release(self, event):
        if not self.rect_start or not self.preview_img_cache: return
        x1, y1 = self.rect_start
        x2, y2 = event.x, event.y
        self.rect_start = None
        self.cvs.delete(self.rect_id)
        
        cx1, cx2 = min(x1, x2), max(x1, x2)
        cy1, cy2 = min(y1, y2), max(y1, y2)
        
        if (cx2 - cx1) < 5 and (cy2 - cy1) < 5:
            pt = self.get_img_coords(cx1, cy1)
            if pt:
                color = self.frames[self.current_frame_idx].convert("RGBA").getpixel(pt)[:3]
                self.palette.filter_by_raw_colors([color])
            return

        p1 = self.get_img_coords(cx1, cy1)
        p2 = self.get_img_coords(cx2, cy2)
        
        img_w, img_h = self.preview_img_cache.size
        cvs_w, cvs_h = self.cvs.winfo_width(), self.cvs.winfo_height()
        off_x = (cvs_w - img_w) // 2
        off_y = (cvs_h - img_h) // 2
        
        ix1 = max(0, cx1 - off_x)
        iy1 = max(0, cy1 - off_y)
        ix2 = min(img_w, cx2 - off_x)
        iy2 = min(img_h, cy2 - off_y)
        
        if ix2 <= ix1 or iy2 <= iy1: return

        raw_frame = self.frames[self.current_frame_idx]
        crop = raw_frame.crop((ix1, iy1, ix2, iy2))
        selected_colors = ColorLogic.analyze_colors(crop)
        self.palette.filter_by_raw_colors(selected_colors)

    def highlight_pixels(self, target_raw_colors):
        if not self.preview_img_cache: return
        if target_raw_colors is None:
            self.draw_image_on_canvas(self.preview_img_cache)
            return
        
        raw_frame = self.frames[self.current_frame_idx].convert("RGBA")
        raw_arr = np.array(raw_frame)
        r, g, b = raw_arr[:,:,0], raw_arr[:,:,1], raw_arr[:,:,2]
        final_mask = np.zeros(r.shape, dtype=bool)
        for tc in target_raw_colors:
            mask = (r == tc[0]) & (g == tc[1]) & (b == tc[2])
            final_mask |= mask
            
        preview_arr = np.array(self.preview_img_cache.convert("RGBA"))
        dimmed = preview_arr.copy()
        dimmed[:, :, :3] = (dimmed[:, :, :3] * 0.3).astype(np.uint8)
        result_arr = np.where(final_mask[..., None], preview_arr, dimmed)
        self.draw_image_on_canvas(Image.fromarray(result_arr))

    def draw_image_on_canvas(self, pil_img):
        self.tk_img_cache = ImageTk.PhotoImage(pil_img)
        w, h = int(self.cvs.winfo_width()), int(self.cvs.winfo_height())
        cx, cy = w//2, h//2
        self.cvs.delete("img_tag") 
        self.cvs.create_image(cx, cy, image=self.tk_img_cache, anchor="center", tags="img_tag")
        self.cvs.tag_lower("img_tag") 

    def update_preview(self):
        if not self.frames: return
        raw = self.frames[self.current_frame_idx]
        tol = self.palette.var_tolerance.get()
        res = ColorLogic.replace_colors(raw, self.global_map, tol)
        if self.current_frame_idx in self.frame_overrides:
            res = ColorLogic.replace_colors(res, self.frame_overrides[self.current_frame_idx], tol)
        self.preview_img_cache = res
        self.draw_image_on_canvas(res)
        self.cvs.delete("txt_tag")
        if self.current_frame_idx in self.frame_overrides:
             w, h = int(self.cvs.winfo_width()), int(self.cvs.winfo_height())
             self.cvs.create_text(w//2, 20, text="[当前帧特殊修改]", fill="yellow", font=("Arial", 9), tags="txt_tag")

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="▶")
            if self.play_job: self.after_cancel(self.play_job)
        else:
            self.is_playing = True
            self.btn_play.config(text="⏸")
            self._loop()
            
    def _loop(self):
        if not self.is_playing: return
        nxt = (self.current_frame_idx + 1) % len(self.frames)
        self.scale_time.set(nxt) 
        self.play_job = self.after(100, self._loop)

    def export_result(self):
        src_path = self.input_path.get()
        if not src_path or not os.path.exists(src_path):
            messagebox.showerror("错误", "源 NPK 路径无效")
            return
        
        self.save_current_state_to_cache()
        if not self.modified_cache:
            messagebox.showinfo("提示", "没有检测到任何修改，不需要导出。")
            return

        self.save_project_state_to_disk()

        exe = get_resource_path("NpkPatcher.exe")
        if not os.path.exists(exe):
            messagebox.showerror("错误", "缺少 NpkPatcher.exe，无法合成文件。")
            return

        d, f = os.path.split(src_path)
        dst_path = os.path.join(d, "%" + f)
        
        if os.path.exists(dst_path):
            if not messagebox.askyesno("覆盖确认", f"目标文件已存在：\n{dst_path}\n\n是否覆盖？"):
                return

        threading.Thread(target=self.run_export_thread, args=(src_path, dst_path, exe)).start()

    def run_export_thread(self, src_path, dst_path, exe):
        self.btn_export.config(state="disabled")
        temp_root = os.path.join(os.getcwd(), "temp_recolor_build")
        if os.path.exists(temp_root): shutil.rmtree(temp_root)
        os.makedirs(temp_root)
        
        try:
            with open(src_path, 'rb') as f:
                src_npk = NPK.open(f)
                src_npk.load_all()
                modified_count = 0
                for npk_file in src_npk.files:
                    if npk_file.name in self.modified_cache:
                        print(f"正在处理修改: {npk_file.name}")
                        rules = self.modified_cache[npk_file.name]
                        g_map = rules["global"]
                        overrides = rules["overrides"]
                        tol = rules.get("tolerance", 10)
                        
                        safe_name = npk_file.name.replace("/", "_").replace("\\", "_")
                        work_dir = os.path.join(temp_root, safe_name)
                        os.makedirs(work_dir, exist_ok=True)
                        
                        img_obj = IMGFactory.open(BytesIO(npk_file.data))
                        csv_lines = []
                        
                        for i, frame in enumerate(img_obj.images):
                            try:
                                current_frame_data_source = frame
                                if isinstance(frame, ImageLink):
                                    target_idx = -1
                                    if hasattr(frame, '_image'): target_idx = frame._image
                                    elif hasattr(frame, 'link'): target_idx = frame.link
                                    elif hasattr(frame, 'target'): target_idx = frame.target
                                    if isinstance(target_idx, int) and 0 <= target_idx < len(img_obj.images):
                                        current_frame_data_source = img_obj.images[target_idx]
                                
                                pil_img = img_obj.build(current_frame_data_source)
                                ox = getattr(frame, 'x', getattr(frame, 'pos_x', 0))
                                oy = getattr(frame, 'y', getattr(frame, 'pos_y', 0))
                                
                                current_map = overrides.get(i, g_map)
                                processed_img = ColorLogic.replace_colors(pil_img, current_map, tol)
                                
                                png_name = f"{i}.png"
                                png_path = os.path.join(work_dir, png_name)
                                processed_img.save(png_path)
                                csv_lines.append(f"{os.path.abspath(png_path)},{ox},{oy}")
                                
                            except Exception as frame_err:
                                print(f"  Frame {i} Error: {frame_err}")
                                dummy = Image.new("RGBA", (1, 1), (0,0,0,0))
                                d_path = os.path.join(work_dir, f"{i}_err.png")
                                dummy.save(d_path)
                                csv_lines.append(f"{os.path.abspath(d_path)},0,0")

                        csv_path = os.path.join(work_dir, "data.csv")
                        new_img_path = os.path.join(work_dir, "new.img")
                        with open(csv_path, "w", encoding="utf-8") as f_csv:
                            f_csv.write("\n".join(csv_lines))
                            
                        si = subprocess.STARTUPINFO()
                        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        ret = subprocess.run([exe, csv_path, new_img_path], capture_output=True, startupinfo=si)
                        if ret.returncode == 0 and os.path.exists(new_img_path):
                            with open(new_img_path, "rb") as f_new:
                                new_data = f_new.read()
                                npk_file.set_data(new_data)
                            modified_count += 1
                        else:
                            print(f"Patcher Failed for {npk_file.name}: {ret.stderr}")

                if modified_count > 0:
                    with open(dst_path, "wb") as f_out:
                        src_npk.save(f_out)
                    messagebox.showinfo("成功", f"处理完成！\n\n共修改了 {modified_count} 个 IMG 文件。\n新文件已保存至:\n{dst_path}")
                else:
                    messagebox.showinfo("提示", "未生成新文件 (可能合成失败或未检测到修改)")

        except Exception as e:
            messagebox.showerror("处理异常", f"发生错误:\n{e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.btn_export.config(state="normal")
            try: shutil.rmtree(temp_root)
            except: pass