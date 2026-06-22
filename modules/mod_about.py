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
from common import Config, check_path_safety, get_resource_path
# =========================================================================
# PART 4.8: 【关于】更新日志与作者页 (B站风 UI)
# =========================================================================
class AboutPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.bg_color = "#f4f5f7" # B站常用的浅灰底色
        self.theme_color = "#FB7299" # B站粉
        self.text_color = "#333333"
        self.configure(bg=self.bg_color)
        
        # 模拟数据：实际使用时请替换图片路径
        # 如果路径不存在，代码会自动生成漂亮的占位图
        self.banner_path = get_resource_path("Background.jpeg")
        self.avatar_path = get_resource_path("avatar.jpg") 
        self.qrcode_path = get_resource_path("qrcode.png")
        self.qq_group_num = "1077552159" # 替换你的群号
        self.raw_banner_img = None 
        # 缓存图片对象防止回收
        self.img_cache = {} 
        
        self.create_widgets()
        self.load_log_data()

    def create_widgets(self):
        # --- 1. 顶部 Header (Canvas 实现背景+头像+文字叠加) ---
        # 高度设置为 180，模拟 Banner 区域
        self.header_h = 180
        self.cvs_header = tk.Canvas(self, height=self.header_h, bg=self.theme_color, 
                                    highlightthickness=0, cursor="arrow")
        self.cvs_header.pack(fill="x", side="top")
        
        # 绑定尺寸变化事件，用于动态调整右侧二维码的位置
        self.cvs_header.bind("<Configure>", self.on_header_resize)

        # 绘制背景 (Banner)
        self.draw_banner()
        
        # 绘制头像 (圆形)
        self.draw_avatar()
        
        # 绘制文字信息
        self.draw_info_text()
        
        # 绘制二维码 (初始绘制，位置会在 resize 中修正)
        self.draw_qrcode()

        # --- 2. 底部 Body (更新日志) ---
        f_body = tk.Frame(self, bg=self.bg_color)
        f_body.pack(fill="both", expand=True, padx=20, pady=20)
        
        # 标题栏
        lbl_title = tk.Label(f_body, text="📅 更新动态 / Update Log", 
                             font=("微软雅黑", 14, "bold"), bg=self.bg_color, fg="#222")
        lbl_title.pack(anchor="w", pady=(0, 10))
        
        # 日志文本框
        self.txt_log = tk.Text(f_body, font=("Consolas", 10), bg="white", relief="flat", padx=15, pady=15)
        sb = ttk.Scrollbar(f_body, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        
        # 配置 Text 标签样式 (颜色高亮)
        self.txt_log.tag_config("ver", foreground=self.theme_color, font=("微软雅黑", 12, "bold"), spacing1=15, spacing3=5)
        self.txt_log.tag_config("date", foreground="#999", font=("Arial", 9))
        self.txt_log.tag_config("new", foreground="#2ecc71", font=("微软雅黑", 10, "bold")) # 绿色
        self.txt_log.tag_config("fix", foreground="#e74c3c", font=("微软雅黑", 10, "bold")) # 红色
        self.txt_log.tag_config("opt", foreground="#f39c12", font=("微软雅黑", 10, "bold")) # 橙色
        self.txt_log.tag_config("del", foreground="#f39c12", font=("微软雅黑", 10, "bold")) # 橙色
        self.txt_log.tag_config("content", foreground="#555", font=("微软雅黑", 10), spacing1=2)
        self.txt_log.tag_config("line", foreground="#eee")

    def draw_banner(self):
        """只负责加载原始图片数据，不处理具体尺寸"""
        if os.path.exists(self.banner_path):
            try:
                # 1. 加载原始图片，转为 RGBA 模式
                self.raw_banner_img = Image.open(self.banner_path).convert("RGBA")
                
                # 2. 在 Canvas 最底层创建一个 image 对象
                # 先随便给个初始图，稍后 on_header_resize 会立马更新它
                self.cvs_header.create_image(0, 0, image=None, anchor="nw", tags="banner")
                
                # 确保背景在所有元素的最下面
                self.cvs_header.tag_lower("banner")
            except Exception as e:
                print(f"Banner load error: {e}")
                self.raw_banner_img = None
        else:
            # 如果没图，就不创建 tag="banner" 的对象，直接露底色
            pass

    def draw_avatar(self):
        # 加载头像并裁剪为圆形
        size = 100
        raw_img = self.try_load_image_raw(self.avatar_path)
        
        # 创建一个圆形的 Image
        if raw_img:
            raw_img = raw_img.resize((size, size), Image.Resampling.LANCZOS)
        else:
            # 如果没图，生成一个带字的占位图
            raw_img = Image.new("RGBA", (size, size), self.theme_color)
            draw = ImageDraw.Draw(raw_img)
            draw.text((25, 35), "Avatar", fill="white")

        # 制作圆形遮罩
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        
        output = ImageOps.fit(raw_img, mask.size, centering=(0.5, 0.5))
        output.putalpha(mask)
        
        tk_avatar = ImageTk.PhotoImage(output)
        self.img_cache["avatar"] = tk_avatar
        
        # 绘制头像 (左侧留白 40px, 垂直居中)
        y_pos = (self.header_h - size) // 2
        self.cvs_header.create_image(40, y_pos, image=tk_avatar, anchor="nw", tags="fixed_left")
        
        # 加一个白色边框圈圈效果
        self.cvs_header.create_oval(40, y_pos, 40+size, y_pos+size, outline="white", width=3, tags="fixed_left")

    def draw_info_text(self):
        # 名字坐标
        x_pos = 160
        y_base = 65
        
        # 1. 绘制名字 (增加交互 tag: "click_name")
        # 名字后面加个提示性的 📋 图标或者仅仅靠鼠标指针变化也行
        self.name_id = self.cvs_header.create_text(x_pos, y_base, text="物理世界的欺骗", 
                                    font=("微软雅黑", 20, "bold"), fill="white", anchor="nw", 
                                    tags=("fixed_left", "click_name"))
        
        # 2. 绘制 "+ 关注" 按钮 (替代原来的 Lv6)
        # B站蓝颜色: #00A1D6
        btn_x = x_pos + 230  # 名字右边一点
        btn_y = y_base + 5
        btn_w = 70
        btn_h = 26
        
        # 【技巧】在 Canvas 上画圆角矩形最简单的方法是画一条很粗的线，并把端点设为圆头
        # 线宽 = 高度，线的长度 = 宽度 - 高度
        self.cvs_header.create_line(btn_x + btn_h/2, btn_y + btn_h/2, 
                                    btn_x + btn_w - btn_h/2, btn_y + btn_h/2,
                                    width=btn_h, fill="#00A1D6", capstyle="round", 
                                    tags=("fixed_left", "follow_btn"))
        
        # 按钮文字
        self.cvs_header.create_text(btn_x + btn_w/2, btn_y + btn_h/2, 
                                    text="+ 关注", fill="white", font=("微软雅黑", 10, "bold"),
                                    tags=("fixed_left", "follow_btn"))

        # 3. 签名
        self.cvs_header.create_text(x_pos, y_base + 45, text="无所为而为", 
                                    font=("微软雅黑", 12), fill="#f0f0f0", anchor="nw", 
                                    tags="fixed_left")

        # --- 绑定交互事件 ---
        
        # 鼠标悬停在名字上：变手型，提示可点击
        self.cvs_header.tag_bind("click_name", "<Enter>", lambda e: self.cvs_header.config(cursor="hand2"))
        self.cvs_header.tag_bind("click_name", "<Leave>", lambda e: self.cvs_header.config(cursor="arrow"))
        # 点击名字：复制
        self.cvs_header.tag_bind("click_name", "<Button-1>", self.copy_name_action)

        # 鼠标悬停在关注按钮上：变手型
        self.cvs_header.tag_bind("follow_btn", "<Enter>", lambda e: self.cvs_header.config(cursor="hand2"))
        self.cvs_header.tag_bind("follow_btn", "<Leave>", lambda e: self.cvs_header.config(cursor="arrow"))
        # 点击关注：跳转浏览器或其他逻辑
        self.cvs_header.tag_bind("follow_btn", "<Button-1>", self.follow_action)
        
    def copy_name_action(self, event):
        """点击名字复制逻辑"""
        name_text = "物理世界的欺骗"
        self.clipboard_clear()
        self.clipboard_append(name_text)
        
        # 视觉反馈：名字变成“已复制”，颜色变绿，然后变回来
        original_text = self.cvs_header.itemcget(self.name_id, "text")
        original_fill = self.cvs_header.itemcget(self.name_id, "fill")
        
        self.cvs_header.itemconfigure(self.name_id, text="已复制 √", fill="#2ecc71") # 绿色反馈
        
        # 0.8秒后恢复
        self.after(800, lambda: self.cvs_header.itemconfigure(self.name_id, text=original_text, fill=original_fill))

    def follow_action(self, event):
        """点击关注按钮逻辑"""
        # 这里可以打开你的B站主页，或者仅仅弹个窗
        import webbrowser
        # 替换成你的B站主页链接
        bilibili_url = "https://space.bilibili.com/492488982" 
        
        # 弹窗询问
        if messagebox.askyesno("关注作者", "是否打开浏览器前往 Bilibili 关注作者？"):
            webbrowser.open(bilibili_url)
    def draw_qrcode(self):
        # 二维码背景卡片
        w, h = 140, 150
        
        # 绘制一个圆角矩形背景 (用 Canvas 多边形模拟或者简易矩形)
        # 这里用 Frame 还是 Canvas 画图？为了方便层级，用 Image 画好贴上去
        
        # 1. 生成卡片底图
        card = Image.new("RGBA", (w, h), (255, 255, 255, 240)) # 半透明白
        
        # 2. 尝试加载二维码
        qr_size = 100
        qr_raw = self.try_load_image_raw(self.qrcode_path)
        if not qr_raw:
            # 占位二维码
            qr_raw = Image.new("RGB", (qr_size, qr_size), "white")
            d = ImageDraw.Draw(qr_raw)
            d.rectangle((10,10,90,90), outline="black", width=2)
            d.text((20,40), "QR CODE", fill="black")
        
        qr_raw = qr_raw.resize((qr_size, qr_size))
        card.paste(qr_raw, ((w-qr_size)//2, 10))
        
        # 3. 绘制文字 "扫码加入组织"
        d_card = ImageDraw.Draw(card)
        # d_card.text( ... ) 略，直接贴图方便
        
        tk_card = ImageTk.PhotoImage(card)
        self.img_cache["qr_card"] = tk_card
        
        # 初始位置 (稍后 resize 会改)
        self.qr_id = self.cvs_header.create_image(800, 15, image=tk_card, anchor="nw")
        
        # 下方加个按钮文本
        self.qr_text_id = self.cvs_header.create_text(800 + w//2, 15 + h - 20, 
                                                      text="点击复制群号", font=("微软雅黑", 9, "bold"), 
                                                      fill="#FB7299")
        
        # 绑定点击复制群号
        self.cvs_header.tag_bind(self.qr_id, "<Button-1>", self.copy_qq)
        self.cvs_header.tag_bind(self.qr_text_id, "<Button-1>", self.copy_qq)

    def on_header_resize(self, event):
        """窗口大小改变时触发：处理背景自适应 + 二维码定位"""
        curr_w = event.width
        curr_h = self.header_h # 高度通常是固定的 180
        
        # --- A. 背景图自适应缩放 (核心代码) ---
        if self.raw_banner_img:
            # 使用 ImageOps.fit 保持比例填满 (裁切多余部分，居中)
            # method=Image.Resampling.LANCZOS 保证缩放质量
            resized_pil = ImageOps.fit(self.raw_banner_img, (curr_w, curr_h), 
                                       method=Image.Resampling.LANCZOS, 
                                       centering=(0.5, 0.5))
            
            # 转为 Tkinter 对象
            tk_img = ImageTk.PhotoImage(resized_pil)
            
            # 【重要】必须存引用，否则会被垃圾回收导致不显示
            self.img_cache["banner_resized"] = tk_img 
            
            # 更新 Canvas 上的图片
            self.cvs_header.itemconfig("banner", image=tk_img)

        # --- B. 二维码和文字的定位逻辑 (保持不变) ---
        card_w = 140
        # 让二维码始终靠右 40px，但不小于 400px (防止遮挡名字)
        new_x = curr_w - card_w - 40
        if new_x < 450: new_x = 450 
        
        # 如果二维码已经画了，更新位置
        if hasattr(self, 'qr_id'):
            self.cvs_header.coords(self.qr_id, new_x, 15)
            # 二维码下方的文字位置
            self.cvs_header.coords(self.qr_text_id, new_x + card_w//2, 15 + 150 - 20)

    def load_log_data(self):
        # === 在这里写你的更新日志 ===
        logs = [
            {
                "ver": "v1.5.1",
                "date": "2026-02-27",
                "content": [
                    ("del", "NPK字典页面"),
                    ("new", "阿拉德调色新增程序化幻彩(渐变色)")
                ]
            },
            {
                "ver": "v1.5.0",
                "date": "2026-02-12",
                "content": [
                    ("new", "Blender渲染页面"),
                    ("new", "阿拉德调色新增颜色预设：粉金/冷艳粉/玫瑰/霓虹")
                ]
            },
            {
                "ver": "v1.4.4",
                "date": "2026-02-08",
                "content": [
                    ("new", "全库秒搜增加文字筛选，先筛选再去重"),
                    ("new", "全库秒搜增加自定义结果个数"),
                    ("opt", "全库秒搜网格结果显示界面改为每页显示自定义个数，支持小键盘左右控制翻页，鼠标滚轮触底翻页，支持页数跳转"),
                    ("opt", "全库秒搜去除主界面显示，改用直接弹窗显示结果")
                ]
            },
            {
                "ver": "v1.4.3",
                "date": "2026-02-01",
                "content": [
                    ("new", "阿拉德调色增加调色算法：线稿化、故障色差"),
                    ("new", "阿拉德调色增加按钮试看一帧")
                ]
            },
            {
                "ver": "v1.4.2",
                "date": "2026-01-31",
                "content": [
                    ("new", "全库秒搜增加网格图片预览"),
                    ("new", "全库秒搜增加鼠标悬停图片预览"),
                    ("new", "全库秒搜增加右键移除重复NPK/IMG选项"),
                    ("new", "全库秒搜增加网格图片预览"),
                    ("fix", "修复阿拉德调色无法生成")
                ]
            },
            {
                "ver": "v1.4.0",
                "date": "2026-01-28",
                "content": [
                    ("new", "全库秒搜"),
                    ("new", "识图训练")
                ]
            },
            {
                "ver": "v1.3.0",
                "date": "2026-01-23",
                "content": [
                    ("new", "指定色替换")
                ]
            },
            {
                "ver": "v1.2.6",
                "date": "2026-01-20",
                "content": [
                    ("new", "NPK字典" ),
                    ("new", "阿拉德调色调色算法增加无选项"),
                    ("new", "阿拉德调色增加额外纹理：自定义：粒子散布（可自行选择图片），雷霆万钧（闪电纹理），像素描边（外发光）"),
                    ("new", "阿拉德调色增加颜色配置的颜色代码，可直接输入颜色代码进行颜色配置")
                ]
            },
            {
                "ver": "v1.2.3",
                "date": "2026-01-17",
                "content": [
                    ("new", "BUFF替换" )
                ]
            },
            {
                "ver": "v1.2.0",
                "date": "2026-01-20",
                "content": [
                    ("new", "阿拉德调色调色算法增加Max-RGB映射"),
                    ("new", "阿拉德调色调色算法增加统一色相"),
                    ("new", "阿拉德调色调色算法增加混合染色"),
                    ("new", "阿拉德调色调色算法增加灰度映射"),
                    ("fix", "修复阿拉德调色引用帧问题")
                ]
            },
            {
                "ver": "v1.1.0",
                "date": "2026-01-01",
                "content": [
                    ("new", "阿拉德调色" )
                ]
            }
        ]
        
        self.txt_log.config(state="normal")
        for log in logs:
            self.txt_log.insert("end", f"{log['ver']}  ", "ver")
            self.txt_log.insert("end", f"{log['date']}\n", "date")
            self.txt_log.insert("end", "_"*60 + "\n", "line") # 分割线
            
            for type_, text in log['content']:
                tag_map = {"new": "[新增]", "fix": "[修复]", "opt": "[优化]","del": "[删除]"}
                prefix = tag_map.get(type_, "[其他]")
                
                self.txt_log.insert("end", f" {prefix} ", type_)
                self.txt_log.insert("end", f"{text}\n", "content")
            
            self.txt_log.insert("end", "\n") # 段落间距
            
        self.txt_log.config(state="disabled")

    def copy_qq(self, event):
        self.clipboard_clear()
        self.clipboard_append(self.qq_group_num)
        self.cvs_header.itemconfigure(self.qr_text_id, text="已复制！", fill="white")
        self.after(2000, lambda: self.cvs_header.itemconfigure(self.qr_text_id, text="点击复制群号", fill="#FB7299"))
        messagebox.showinfo("提示", f"群号 {self.qq_group_num} 已复制到剪贴板！\n请打开 QQ 搜索加入。")

    # --- 辅助：图片加载 ---
    def try_load_image_raw(self, path):
        if not path or not os.path.exists(path): return None
        try:
            return Image.open(path).convert("RGBA")
        except: return None

    def try_load_image(self, path, size=None, mode="contain"):
        raw = self.try_load_image_raw(path)
        if not raw: return None
        if size:
            if mode == "cover":
                return ImageTk.PhotoImage(ImageOps.fit(raw, size, centering=(0.5, 0.5)))
            else:
                raw.thumbnail(size)
                return ImageTk.PhotoImage(raw)
        return ImageTk.PhotoImage(raw)