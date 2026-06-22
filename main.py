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

# 假设这些是你提取出来的公共工具
from common import Config, check_path_safety, get_resource_path

# =========================================================================
# 模块导入区 (Lazy Import 策略)
# 这里的技巧是：先不导入具体模块，等用户点击按钮时再导入
# 这样程序启动极快，且某个模块坏了不影响主程序启动
# =========================================================================

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # 1. 基础窗口设置
        self.title("幻色棱镜V1.51 —— Powered by 物理世界的欺骗")
        self.geometry("1000x750")
        #self.iconbitmap(default="") # 如果有图标可设置
        
        # 2. 全局样式配置
        self.sidebar_bg = "#2c3e50"
        self.sidebar_fg = "#ecf0f1"
        self.hover_color = "#34495e"
        self.active_color = "#1abc9c"
        
        # 3. 初始化页面缓存容器
        # 结构: { "page_key": {"frame": 页面实例, "module": 模块引用} }
        self.loaded_pages = {} 
        self.current_page_key = None
        self.create_icon()

        # 4. 构建 UI 骨架
        self.setup_ui()
        
        # 5. 默认显示首页 (比如关于页)
        self.show_page("about")
        
    def create_icon(self):
        # 简单的程序图标
        try:
            size = 32
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse((2,2,30,30), fill="#e74c3c", outline="white", width=2)
            tk_img = ImageTk.PhotoImage(img)
            self.iconphoto(True, tk_img)
        except: pass
		
    def setup_ui(self):
        """搭建左侧导航和右侧内容区"""
        # --- 左侧导航栏 ---
        self.sidebar = tk.Frame(self, bg=self.sidebar_bg, width=200)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False) # 禁止随内容收缩

        # 标题
        tk.Label(self.sidebar, text="幻色棱镜", bg=self.sidebar_bg, fg="#e74c3c", 
                 font=("微软雅黑", 20, "bold")).pack(pady=(40, 20))

        # 导航按钮容器
        self.nav_frame = tk.Frame(self.sidebar, bg=self.sidebar_bg)
        self.nav_frame.pack(fill="x")
        
        # --- 注册导航按钮 ---
        # 格式: (显示名称, 唯一Key, 模块文件名)
        self.nav_buttons = {}
        
        self.add_nav_btn("🎨 阿拉德调色", "prism")
        self.add_nav_btn("🪐 Blender 渲染", "blender")
        self.add_nav_btn("✨ 指定色替换", "recolor")
        self.add_nav_btn("📹 BUFF替换", "buff")
        self.add_nav_btn("🔍 全库秒搜", "google")
        self.add_nav_btn("🧠 识图训练", "train")
        self.add_nav_btn("💁 关于 & 日志", "about") 

        # --- 右侧内容挂载点 ---
        self.content_area = tk.Frame(self, bg="#f0f0f0")
        self.content_area.pack(side="right", fill="both", expand=True)

    def add_nav_btn(self, text, key):
        """创建导航按钮"""
        btn = tk.Button(self.nav_frame, text=text, bg=self.sidebar_bg, fg=self.sidebar_fg,
                        font=("微软雅黑", 11), bd=0, cursor="hand2", anchor="w", padx=25, pady=10,
                        command=lambda: self.show_page(key))
        btn.pack(fill="x", pady=1)
        self.nav_buttons[key] = btn

    def show_page(self, key):
        """核心路由逻辑：切换页面"""
        
        # 1. 更新按钮样式 (高亮当前)
        if self.current_page_key:
            self.nav_buttons[self.current_page_key].config(bg=self.sidebar_bg, fg=self.sidebar_fg)
        self.nav_buttons[key].config(bg=self.active_color, fg="white")
        self.current_page_key = key

        # 2. 隐藏当前页面
        for p_data in self.loaded_pages.values():
            p_data["frame"].pack_forget()

        # 3. 懒加载：如果页面不存在，则导入模块并实例化
        if key not in self.loaded_pages:
            try:
                page_instance = self.load_module_dynamically(key)
                self.loaded_pages[key] = {"frame": page_instance}
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("加载失败", f"无法加载模块 [{key}]:\n{e}")
                return

        # 4. 显示目标页面
        target_page = self.loaded_pages[key]["frame"]
        target_page.pack(fill="both", expand=True)

    def load_module_dynamically(self, key):
        """根据 Key 动态导入 modules 下的文件"""
        
        # 这里定义 页面Key -> (模块名, 类名) 的映射
        # 这样主程序完全不需要 import mod_ai，只有运行到这里才 import
        mapping = {
            "ai":    ("modules.mod_ai", "GoogleSearchPage"),
            "train": ("modules.mod_ai", "TrainPage"),      # 复用同一个文件
            "prism": ("modules.mod_prism", "PrismPage"),   # 假设你已经拆分了
            "about": ("modules.mod_about", "AboutPage"),
            "blender": ("modules.mod_blender", "BlenderPage"),
            "recolor": ("modules.mod_recolor", "AdvancedRecolorPage"),
            "buff": ("modules.mod_buff", "BuffPage"),
            "google": ("modules.mod_ai", "GoogleSearchPage")
        }

        if key not in mapping:
            # 默认创建一个空页面占位
            f = tk.Frame(self.content_area)
            tk.Label(f, text=f"页面 [{key}] 正在启动中...", font=("微软雅黑", 20)).pack(expand=True)
            return f

        mod_path, class_name = mapping[key]
        
        # 动态导入技巧
        # 如果还没导入过，这里会执行 import；如果导入过，直接用内存里的
        if mod_path not in sys.modules:
            print(f"Loading module: {mod_path} ...")
            __import__(mod_path)
        
        module = sys.modules[mod_path]
        PageClass = getattr(module, class_name)
        
        # 实例化页面 (传入 content_area 作为父容器)
        return PageClass(self.content_area)

if __name__ == "__main__":
    # check_path_safety() # 来自 common
    # Config.load()       # 来自 common
    app = MainApplication()
    app.mainloop()