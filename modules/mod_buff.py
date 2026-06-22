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


HAS_MOVIEPY = True


# --- 核心构建函数 ---
def build_hidden_bk2(image_folder, output_bk2, tool_path, status_callback=None):
    def log(msg):
        if status_callback: status_callback(msg)
        print(msg)

    # 路径处理
    image_folder = os.path.abspath(image_folder)
    output_bk2 = os.path.abspath(output_bk2)
    tool_path = os.path.abspath(tool_path)

    if not os.path.exists(tool_path):
        log(f"❌ 错误：找不到工具: {tool_path}")
        return False

    # 整理图片
    log("⚙️ 正在整理图片序列...")
    pngs = glob.glob(os.path.join(image_folder, "*.png"))
    if not pngs:
        log(f"❌ 错误：文件夹为空: {image_folder}")
        return False

    needs_rename = False
    for p in pngs:
        if "frame_" not in os.path.basename(p): needs_rename = True; break
    
    if needs_rename:
        try: pngs.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
        except: pngs.sort()
        for index, old_path in enumerate(pngs):
            new_name = f"frame_{index:04d}.png"
            new_path = os.path.join(image_folder, new_name)
            if old_path != new_path:
                try: os.rename(old_path, new_path)
                except: pass
    
    pngs = glob.glob(os.path.join(image_folder, "frame_*.png"))
    pngs.sort()

    # 生成 List 文件
    list_file_path = os.path.join(image_folder, "files.lst")
    try:
        with open(list_file_path, "w", encoding="mbcs") as f:
            for img in pngs: f.write(os.path.abspath(img) + "\n")
    except:
        with open(list_file_path, "w", encoding="utf-8") as f:
            for img in pngs: f.write(os.path.abspath(img) + "\n")

    # 构造命令
    tool_name = os.path.basename(tool_path).lower()
    cmd = [tool_path]
    if "radvideo" in tool_name:
        cmd.append("bink")
        
    cmd.append(list_file_path)
    cmd.append(output_bk2)
    
    # 【修改点2】参数调整
    cmd.append("/Z3000")   # 告诉它处理Alpha (DNF需要)
    #cmd.append("/Z10000")  # 【关键】禁止弹出"没有Alpha"的警告，没有就强制按没有处理
    cmd.append("/O")       # 覆盖
    cmd.append("/#")       # 静默退出

    debug_cmd_str = " ".join([f'"{x}"' if " " in x else x for x in cmd])
    log(f"🚀 正在执行...")
    
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, startupinfo=startupinfo)

        if os.path.exists(output_bk2) and os.path.getsize(output_bk2) > 1024:
            log(f"✅ 成功！视频已生成")
            if os.path.exists(list_file_path): os.remove(list_file_path)
            return True
        else:
            log(f"❌ 生成失败。")
            error_msg = f"RAD工具执行失败。\n手动命令：\n{debug_cmd_str}\n\n输出: {result.stdout} {result.stderr}"
            print(error_msg)
            messagebox.showerror("生成失败", error_msg) 
            return False

    except Exception as e:
        log(f"❌ 异常: {e}")
        return False
        
        
class EffectFactory:

    # --- 工具: 缓动函数 ---
    
    # 减速曲线 (用于入场：快 -> 慢)
    @staticmethod
    def _ease_out_expo(t):
        return 1.0 if t == 1.0 else 1.0 - math.pow(2.0, -10.0 * t)

    # 加速曲线 (用于退场：慢 -> 快)
    @staticmethod
    def _ease_in_expo(t):
        return 0.0 if t == 0.0 else math.pow(2.0, 10.0 * (t - 1.0))

    # --- 工具: 左右渐变蒙版 (常驻) ---
    @staticmethod
    def _create_side_fade_mask(w, h):
        fade_ratio = 0.2 
        fade_w = int(w * fade_ratio)
        if fade_w < 1: fade_w = 1
        
        # 0(透) -> 255(实)
        grad_left = np.linspace(0, 255, fade_w)
        # 255(实) -> 0(透)
        grad_right = np.linspace(255, 0, fade_w)
        
        center_w = w - (fade_w * 2)
        if center_w < 0: center_w = 0
        grad_center = np.full(center_w, 255.0)
        
        base_line = np.concatenate([grad_left, grad_center, grad_right])
        
        # 修正长度误差
        if len(base_line) < w:
            base_line = np.append(base_line, np.zeros(w - len(base_line)))
        elif len(base_line) > w:
            base_line = base_line[:w]
            
        mask_2d = np.tile(base_line, (h, 1))
        return Image.fromarray(mask_2d.astype(np.uint8), mode="L")

    @staticmethod
    def generate(input_data, mode, frames=18):
        # 1. 准备基础数据
        is_sequence = isinstance(input_data, list)
        if is_sequence:
            ref_img = input_data[0]
            w, h = ref_img.size
        else:
            ref_img = input_data
            w, h = ref_img.size

        # 预先生成侧边蒙版 (数组化以提高性能)
        side_mask = None
        side_mask_arr = None
        
        # 只有在需要特效时才生成蒙版
        if mode == "简单 Cut-in":
            side_mask = EffectFactory._create_side_fade_mask(w, h)
            side_mask_arr = np.array(side_mask).astype(float)

        results = []

        # 关键帧定义
        # Frame 0-3: 入场 (4帧)
        # Frame 4-14: 保持 (11帧)
        # Frame 15-17: 退场 (3帧)
        
        for i in range(frames):
            # --- 步骤 1: 创建绝对透明的画布 ---
            canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            
            # --- 步骤 2: 获取当前帧的角色图 ---
            if is_sequence:
                char_img = input_data[i % len(input_data)]
            else:
                char_img = input_data
            
            if char_img.mode != "RGBA":
                char_img = char_img.convert("RGBA")

            # --- 步骤 3: 计算位移 (核心动作) ---
            offset_y = 0
            
            # 【修复核心】只有选中特效模式时，才计算位移
            if mode == "简单 Cut-in":
                if i < 4:
                    # [入场]: 从下往上冲
                    t = (i + 1) / 4.0 
                    ease = EffectFactory._ease_out_expo(t)
                    start_offset = h 
                    current_offset = start_offset * (1.0 - ease)
                    offset_y = int(current_offset)
                    
                elif i >= 15:
                    # [退场]: 从中往上冲
                    t = (i - 14) / 3.0
                    if t > 1.0: t = 1.0
                    ease = EffectFactory._ease_in_expo(t) 
                    target_offset = -h 
                    current_offset = target_offset * ease
                    offset_y = int(current_offset)
                else:
                    offset_y = 0
            else:
                # 模式为 "无" 时，没有任何位移
                offset_y = 0

            # --- 步骤 4: 绘制 ---
            # X居中，Y居中+偏移
            paste_x = (w - char_img.width) // 2
            paste_y = (h - char_img.height) // 2 + offset_y
            
            # 粘贴 (保持透明底)
            canvas.paste(char_img, (paste_x, paste_y), char_img)

            # --- 步骤 5: 应用侧边渐变蒙版 ---
            # 【修复核心】删除了 "or True"，只有名称匹配才应用蒙版
            if mode == "简单 Cut-in" and side_mask_arr is not None:
                r, g, b, a = canvas.split()
                a_arr = np.array(a).astype(float)
                
                # 简单乘法：保留原图透明度 * 侧边蒙版
                new_a_arr = (a_arr * side_mask_arr) / 255.0
                
                final_a = Image.fromarray(new_a_arr.astype(np.uint8))
                canvas.putalpha(final_a)
            
            results.append(canvas)

        return results

class VideoProcessor:
    @staticmethod
    def process_video_to_frames(video_path, output_dir, target_frames=18, crop_box=None):
        """
        读取视频 -> 抽帧(跳过首帧) -> 裁剪 -> 强制转RGBA -> 保存
        (已移除 chroma_key_mode 等背景去除参数)
        """
        try:
            from moviepy.editor import ImageClip
            HAS_MOVIEPY = True
        except Exception as e:
            HAS_MOVIEPY = False
            
        if not HAS_MOVIEPY:
            return False, "未安装 MoviePy 库"

        try:
            from moviepy.editor import VideoFileClip
            
            # 1. 加载视频
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            # 跳过前0.15秒，防止淡入黑屏
            start_time = 0.15 if duration > 0.3 else 0 
            end_time = duration - 0.1
            
            times = np.linspace(start_time, end_time, target_frames)
            
            processed_count = 0
            
            for i, t in enumerate(times):
                try:
                    frame = clip.get_frame(t)
                except OSError:
                    continue

                img = Image.fromarray(frame)
                
                # 裁剪
                if crop_box:
                    cx, cy, cw, ch = crop_box
                    img_w, img_h = img.size
                    cx = max(0, cx)
                    cy = max(0, cy)
                    cw = min(cw, img_w - cx)
                    ch = min(ch, img_h - cy)
                    if cw > 0 and ch > 0:
                        img = img.crop((cx, cy, cx + cw, cy + ch))
                
                # 强制转为 RGBA (保留Alpha通道能力，即使不扣像也保持格式统一)
                img = img.convert("RGBA")
                
                # --- 原背景去除逻辑已移除 ---
                
                save_name = f"frame_{i:04d}.png"
                img.save(os.path.join(output_dir, save_name))
                processed_count += 1
            
            clip.close()
            return True, f"成功转换 {processed_count} 帧"

        except Exception as e:
            return False, f"视频处理出错: {e}"


class VideoSettingsDialog(tk.Toplevel):
    def __init__(self, parent, video_path, callback):
        super().__init__(parent)
        self.title("🎬 视频预处理 - 拖动红框裁剪区域")
        self.video_path = video_path
        self.callback = callback
        self.result_data = None
        
        # 视频原始信息
        self.raw_w = 0
        self.raw_h = 0
        self.display_scale = 1.0 # 预览缩放比
        
        # 裁剪框坐标 (真实坐标)
        self.crop_x = 0
        self.crop_y = 0
        self.crop_w = 0
        self.crop_h = 0
        
        # 鼠标拖拽状态
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.rect_start_x = 0
        self.rect_start_y = 0
        
        # 设置变量
        self.var_frames = tk.IntVar(value=18)
        self.var_crop_w = tk.IntVar(value=0) # 绑定输入框
        self.var_crop_h = tk.IntVar(value=0) # 绑定输入框
        
        self.create_ui()
        self.load_preview_and_init() # 加载视频并初始化尺寸
        self.center_window()

    def center_window(self):
        self.update_idletasks()
        w, h = 900, 650 # 稍微增加高度以容纳新按钮
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def create_ui(self):
        # 左侧预览 (画布)
        f_left = ttk.LabelFrame(self, text=" 裁剪预览 (拖动红框) ", padding=10)
        f_left.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(f_left, bg="#333", width=500, height=500, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)

        # 右侧设置
        f_right = ttk.Frame(self, padding=10)
        f_right.pack(side="right", fill="y", padx=10)

        # 1. 基础设置
        grp_basic = ttk.LabelFrame(f_right, text="抽帧设置", padding=10)
        grp_basic.pack(fill="x", pady=5)
        ttk.Label(grp_basic, text="总帧数:").pack(anchor="w")
        ttk.Entry(grp_basic, textvariable=self.var_frames).pack(fill="x")

        # 2. 裁剪设置 (数字控制)
        grp_crop = ttk.LabelFrame(f_right, text="裁剪尺寸 (px)", padding=10)
        grp_crop.pack(fill="x", pady=5)
        
        # 宽
        f_cw = ttk.Frame(grp_crop)
        f_cw.pack(fill="x", pady=2)
        ttk.Label(f_cw, text="宽:", width=4).pack(side="left")
        entry_w = ttk.Entry(f_cw, textvariable=self.var_crop_w)
        entry_w.pack(side="left", fill="x", expand=True)
        entry_w.bind("<Return>", self.on_entry_change) 
        
        # 高
        f_ch = ttk.Frame(grp_crop)
        f_ch.pack(fill="x", pady=2)
        ttk.Label(f_ch, text="高:", width=4).pack(side="left")
        entry_h = ttk.Entry(f_ch, textvariable=self.var_crop_h)
        entry_h.pack(side="left", fill="x", expand=True)
        entry_h.bind("<Return>", self.on_entry_change)

        # 操作按钮区
        f_btns = ttk.Frame(grp_crop)
        f_btns.pack(fill="x", pady=5)
        
        # 【修改点】增加“应用尺寸”按钮
        ttk.Button(f_btns, text="应用尺寸", command=self.on_entry_change, width=10).pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        # 【修改点】改为“重置为原图”
        ttk.Button(f_btns, text="重置为原图", command=self.reset_to_original, width=10).pack(side="left", fill="x", expand=True, padx=(2, 0))

        # 底部按钮
        ttk.Button(f_right, text="✅ 开始转换", command=self.confirm, width=20).pack(side="bottom", pady=20)
        ttk.Button(f_right, text="❌ 取消", command=self.destroy).pack(side="bottom")

    def load_preview_and_init(self):
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            return
        if not HAS_MOVIEPY: return
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(self.video_path)
            frame = clip.get_frame(0) # 第0秒
            clip.close()
            
            pil_img = Image.fromarray(frame)
            self.raw_w, self.raw_h = pil_img.size
            
            # 【修改点】初始默认设置为 470 x 900
            default_w = 470
            default_h = 900
            
            self.crop_w = default_w if self.raw_w >= default_w else self.raw_w
            self.crop_h = default_h if self.raw_h >= default_h else self.raw_h
            
            # 居中放置裁剪框
            self.crop_x = (self.raw_w - self.crop_w) // 2
            self.crop_y = (self.raw_h - self.crop_h) // 2
            
            self.var_crop_w.set(self.crop_w)
            self.var_crop_h.set(self.crop_h)
            
            # 计算显示缩放
            max_view = 480
            scale_w = max_view / self.raw_w
            scale_h = max_view / self.raw_h
            self.display_scale = min(scale_w, scale_h, 1.0)
            
            display_w = int(self.raw_w * self.display_scale)
            display_h = int(self.raw_h * self.display_scale)
            
            # 生成预览图
            self.tk_img = ImageTk.PhotoImage(pil_img.resize((display_w, display_h)))
            
            cx = 250
            cy = 250
            self.img_offset_x = cx - display_w // 2
            self.img_offset_y = cy - display_h // 2
            
            self.canvas.create_image(self.img_offset_x, self.img_offset_y, image=self.tk_img, anchor="nw")
            
            self.rect_id = self.canvas.create_rectangle(0, 0, 1, 1, outline="red", width=3, tags="crop_rect")
            self.draw_rect()
            
        except Exception as e:
            print(f"Preview Error: {e}")

    def draw_rect(self):
        # 将真实坐标映射到画布坐标
        x1 = self.img_offset_x + (self.crop_x * self.display_scale)
        y1 = self.img_offset_y + (self.crop_y * self.display_scale)
        w = self.crop_w * self.display_scale
        h = self.crop_h * self.display_scale
        self.canvas.coords(self.rect_id, x1, y1, x1+w, y1+h)

    def on_mouse_down(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.rect_start_x = self.crop_x
        self.rect_start_y = self.crop_y

    def on_mouse_drag(self, event):
        dx = (event.x - self.drag_start_x) / self.display_scale
        dy = (event.y - self.drag_start_y) / self.display_scale
        
        new_x = self.rect_start_x + dx
        new_y = self.rect_start_y + dy
        
        max_x = self.raw_w - self.crop_w
        max_y = self.raw_h - self.crop_h
        
        self.crop_x = max(0, min(new_x, max_x))
        self.crop_y = max(0, min(new_y, max_y))
        
        self.draw_rect()

    def on_entry_change(self, event=None):
        try:
            w = int(self.var_crop_w.get())
            h = int(self.var_crop_h.get())
            
            # 限制输入不能超过原视频尺寸
            w = min(w, self.raw_w)
            h = min(h, self.raw_h)
            
            # 限制最小尺寸
            w = max(10, w)
            h = max(10, h)
            
            # 更新变量显示
            self.var_crop_w.set(w)
            self.var_crop_h.set(h)

            self.crop_w = w
            self.crop_h = h
            
            # 重置坐标防止越界
            if self.crop_x + self.crop_w > self.raw_w: self.crop_x = self.raw_w - self.crop_w
            if self.crop_y + self.crop_h > self.raw_h: self.crop_y = self.raw_h - self.crop_h
            self.draw_rect()
        except: pass

    def reset_to_original(self):
        # 【修改点】重置为原视频大小
        self.crop_x = 0
        self.crop_y = 0
        self.crop_w = self.raw_w
        self.crop_h = self.raw_h
        
        self.var_crop_w.set(self.crop_w)
        self.var_crop_h.set(self.crop_h)
        self.draw_rect()

    def confirm(self):
        # 收集数据
        self.result_data = {
            "frames": self.var_frames.get(),
            "crop_box": (int(self.crop_x), int(self.crop_y), int(self.crop_w), int(self.crop_h))
        }
        if self.callback:
            self.callback(self.result_data)
        self.destroy()
    

# =========================================================================
# PART 4: UI 页面 - BUFF替换 (BK2 生成器) - [已修改]
# =========================================================================

# 1. 定义职业对照数据 (直接嵌入代码，方便调用)
BUFF_MAPPING_SRC = """
男鬼剑士
03mghost_buf_asura——阿修罗
03mghost_buf_bsk——狂战士
03mghost_buf_ghost——剑影
03mghost_buf_soul——鬼泣
03mghost_buf_wep——剑魂
女鬼剑
(TN)fgs_buf_blade3——刃影
(TN)fgs_buf_demon3——剑魔
fgs_buf_darktemp3——暗帝
fgs_buf_swordma3——剑宗
fgs_buf_vega3——剑帝
女格斗家
(TN)05Ffighter_buf_grap——柔道家(女)
05Ffighter_buf_nen——气功师(女)
05Ffighter_buf_street——街霸(女)
05Ffighter_buf_strik——散打(女)
男格斗家
02mfigher_buf_grap——柔道家(男)
02mfigher_buf_nen——气功师(男)
02mfigher_buf_Street——街霸(男)
02mfighter_buf_strik——散打(男)
男神枪手
04mgunner_buf_assult——合金战士
04mgunner_buf_luncher——枪炮师(男)
04mgunner_buf_meca——机械师(男)
04mgunner_buf_ranger——漫游枪手(男)
04mgunner_buf_spit——弹药专家(男)
女神枪手
(TN)11Fgunner_buf_launcher——枪炮师(女)
11Fgunner_buf_meca——机械师(女)
11Fgunner_buf_ranger——漫游枪手(女)
11Fgunner_buf_spit——弹药专家(女)
11Fgunner_buf_paramedic——协战师(女)
男魔法师
10Mmage_buf_bloodm——血法师
10Mmage_buf_dimension——次元行者
10Mmage_buf_elbomber——元素爆破师
10Mmage_buf_glancial——冰结师
10Mmage_buf_swiftma——逐风者
女魔法师
07fmage_buf_battlemage——战斗法师
07fmage_buf_element——元素师
07fmage_buf_enchant——小魔女
07fmage_buf_summoner——召唤师
07fmage_buf_witch——魔道学者
男圣职者
06Mprist_buf_avenger——复仇者
06Mprist_buf_battlecru——圣骑士(审判)
06Mprist_buf_buffcru——圣骑士(奶爸)
06Mprist_buf_exorcist——驱魔师
06Mprist_buf_infight——蓝拳圣使
女圣职者
(TN)08Fpriest_buf_sorcer——驱魔师
08Fpriest_buf_crusager——光明骑士
08Fpriest_buf_inquis——正义审判者
08Fpriest_buf_mistress——除恶者
暗夜使者
(TN)09thief_buf_necro——黑夜术士
(TN)09thief_buf_rogue——暗星
09thief_buf_kuno——忍者
09thief_buf_shadow——影舞者
守护者
(TN)12knight_buf_chaos——混沌魔灵
(TN)12knight_buf_dragonkn——龙骑士
12knight_buf_eleven——精灵骑士
12knight_buf_paladin——帕拉丁
魔枪士
13demolancer_buf_darklancer——暗枪士
13demolancer_buf_dralancer——狩猎者
13demolancer_buf_duelist——决战者
13demolancer_buf_vanguard——征战者
枪剑士
14GunBla_buf_agent——特工
14GunBla_buf_hitman——暗刃
14GunBla_buf_specilist——源能专家
14GunBla_buf_trouble——战线佣兵
弓箭手
(TN)17archer_buf_hunter——猎人
(TN)17archer_buf_vigil——妖护使
17archer_buf_muse——缪斯
17archer_buf_traveler——旅人
(TN)17archer_buf_chimera——奇美拉
外传职业
15darkknight_buf——黑暗武士
(TN)16Creater_buf——缔造者
"""

# =========================================================================
# 辅助类：手动裁剪窗口
# =========================================================================
# =========================================================================
# 辅助类：手动裁剪窗口 (修复版：左上角对齐 + 智能缩放)
# =========================================================================
# =========================================================================
# 辅助类：手动裁剪窗口 (修复版：支持宽高自定义 + 智能窗口 + 边界限制)
# =========================================================================
# =========================================================================
# 辅助类：手动裁剪窗口 (修改版：移除ESC/回车快捷键，保留输入框回车更新预览)
# =========================================================================
class ManualCropper(tk.Toplevel):
    def __init__(self, master, img_path, callback=None):
        super().__init__(master)
        # 【修改点】标题去掉了快捷键提示
        self.title("✂️ 图片裁剪 - 拖动红框 / 输入尺寸")
        self.callback = callback
        self.src_img = Image.open(img_path)
        
        # 1. 初始尺寸逻辑
        self.img_w, self.img_h = self.src_img.size
        
        # 默认裁剪尺寸 (470x668)，如果原图小，则取原图大小
        self.crop_w_real = 470 if self.img_w >= 470 else self.img_w
        self.crop_h_real = 668 if self.img_h >= 668 else self.img_h
        
        # 2. 智能计算窗口显示比例
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight() - 150
        
        target_view_h = int(screen_h * 0.8)
        self.scale = target_view_h / self.img_h
        
        # 如果宽度超标，改用宽度适配
        if (self.img_w * self.scale) > (screen_w * 0.9):
            self.scale = (screen_w * 0.9) / self.img_w

        # 显示尺寸
        self.display_w = int(self.img_w * self.scale)
        self.display_h = int(self.img_h * self.scale)
        
        # 3. 窗口大小设置 (最小宽度 500)
        win_content_w = self.display_w + 20
        self.win_w = max(win_content_w, 500) 
        self.win_h = self.display_h + 100    
        
        self.geometry(f"{self.win_w}x{self.win_h}")
        
        self.tk_img = ImageTk.PhotoImage(self.src_img.resize((self.display_w, self.display_h)))

        # 4. 初始化红框位置 (居中)
        self.box_display_w = self.crop_w_real * self.scale
        self.box_display_h = self.crop_h_real * self.scale
        
        self.rect_x = (self.display_w - self.box_display_w) / 2
        self.rect_y = (self.display_h - self.box_display_h) / 2

        self.create_ui()
        self.center_window()

    def create_ui(self):
        # --- 顶部操作栏 ---
        top_bar = ttk.Frame(self, padding=10)
        top_bar.pack(fill="x", side="top")
        
        # 宽度控制
        ttk.Label(top_bar, text="宽:").pack(side="left")
        self.var_w = tk.IntVar(value=self.crop_w_real)
        e_w = ttk.Entry(top_bar, textvariable=self.var_w, width=6)
        e_w.pack(side="left", padx=(0, 5))
        # 这里的回车保留：只更新红框大小，不关闭窗口
        e_w.bind("<Return>", lambda e: self.update_rect_from_entry())
        
        # 高度控制
        ttk.Label(top_bar, text="高:").pack(side="left")
        self.var_h = tk.IntVar(value=self.crop_h_real)
        e_h = ttk.Entry(top_bar, textvariable=self.var_h, width=6)
        e_h.pack(side="left", padx=(0, 10))
        # 这里的回车保留：只更新红框大小，不关闭窗口
        e_h.bind("<Return>", lambda e: self.update_rect_from_entry())
        
        ttk.Button(top_bar, text="应用尺寸", command=self.update_rect_from_entry).pack(side="left")
        ttk.Button(top_bar, text="重置为全图", command=self.reset_full).pack(side="left", padx=5)
        
        ttk.Button(top_bar, text="✅ 确认裁剪", command=self.confirm).pack(side="right", padx=10, fill="y")

        # --- 画布区域 ---
        canvas_container = tk.Frame(self, bg="#333")
        canvas_container.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(canvas_container, width=self.display_w, height=self.display_h, bg="#222", cursor="fleur")
        self.canvas.pack(pady=10) 
        
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")
        
        # 遮罩层
        self.mask_color = "black"
        self.mask_stipple = "gray50"
        self.mask_top = self.canvas.create_rectangle(0,0,0,0, fill=self.mask_color, stipple=self.mask_stipple, width=0)
        self.mask_btm = self.canvas.create_rectangle(0,0,0,0, fill=self.mask_color, stipple=self.mask_stipple, width=0)
        self.mask_lft = self.canvas.create_rectangle(0,0,0,0, fill=self.mask_color, stipple=self.mask_stipple, width=0)
        self.mask_rgt = self.canvas.create_rectangle(0,0,0,0, fill=self.mask_color, stipple=self.mask_stipple, width=0)

        # 红框
        self.rect_id = self.canvas.create_rectangle(0, 0, 1, 1, outline="#ff0000", width=2, tag="rect")
        
        self.draw_rect()

        # 事件
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        
        # 【修改点】移除了全局的 <Return> 和 <Escape> 绑定
        # self.bind("<Return>", lambda e: self.confirm())  <-- 已删除
        # self.bind("<Escape>", lambda e: self.destroy())  <-- 已删除

    def center_window(self):
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.win_w // 2)
        y = (self.winfo_screenheight() // 2) - (self.win_h // 2)
        self.geometry(f"+{x}+{y}")

    def update_rect_from_entry(self):
        try:
            w = int(self.var_w.get())
            h = int(self.var_h.get())
            
            # 限制不能超过原图
            w = min(w, self.img_w)
            h = min(h, self.img_h)
            w = max(1, w)
            h = max(1, h)
            
            self.var_w.set(w)
            self.var_h.set(h)
            
            self.crop_w_real = w
            self.crop_h_real = h
            self.box_display_w = w * self.scale
            self.box_display_h = h * self.scale
            
            max_x = self.display_w - self.box_display_w
            max_y = self.display_h - self.box_display_h
            self.rect_x = min(max(0, self.rect_x), max_x)
            self.rect_y = min(max(0, self.rect_y), max_y)
            
            self.draw_rect()
        except: pass

    def reset_full(self):
        self.var_w.set(self.img_w)
        self.var_h.set(self.img_h)
        self.rect_x = 0
        self.rect_y = 0
        self.update_rect_from_entry()

    def on_click(self, event):
        self.move_rect_to_mouse(event.x, event.y)

    def on_drag(self, event):
        self.move_rect_to_mouse(event.x, event.y)

    def move_rect_to_mouse(self, mx, my):
        new_x = mx - (self.box_display_w / 2)
        new_y = my - (self.box_display_h / 2)
        
        max_x = self.display_w - self.box_display_w
        max_y = self.display_h - self.box_display_h
        
        self.rect_x = max(0, min(new_x, max_x))
        self.rect_y = max(0, min(new_y, max_y))
        
        self.draw_rect()

    def draw_rect(self):
        x1, y1 = self.rect_x, self.rect_y
        x2 = x1 + self.box_display_w
        y2 = y1 + self.box_display_h
        
        self.canvas.coords(self.rect_id, x1, y1, x2, y2)
        
        # Top
        self.canvas.coords(self.mask_top, 0, 0, self.display_w, y1)
        # Bottom
        self.canvas.coords(self.mask_btm, 0, y2, self.display_w, self.display_h)
        # Left
        self.canvas.coords(self.mask_lft, 0, y1, x1, y2)
        # Right
        self.canvas.coords(self.mask_rgt, x2, y1, self.display_w, y2)

    def confirm(self):
        real_x = int(self.rect_x / self.scale)
        real_y = int(self.rect_y / self.scale)
        real_w = int(self.crop_w_real)
        real_h = int(self.crop_h_real)
        
        if real_x + real_w > self.img_w: real_x = self.img_w - real_w
        if real_y + real_h > self.img_h: real_y = self.img_h - real_h
        
        box = (real_x, real_y, real_x + real_w, real_y + real_h)
        cropped = self.src_img.crop(box)
        
        if self.callback:
            self.callback(cropped)
        self.destroy()

def parse_buff_data():
    data = {}
    current_cat = None
    lines = BUFF_MAPPING_SRC.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        if "——" in line:
            code, name = line.split("——", 1)
            if current_cat:
                data[current_cat][name.strip()] = code.strip()
        else:
            current_cat = line
            data[current_cat] = {}
    return data

BUFF_DATA = parse_buff_data()

# =========================================================================
# PART 4: UI 页面 - BUFF替换 (带预览功能版)
# =========================================================================

# (请保留 BUFF_MAPPING_SRC 和 parse_buff_data 函数，不要删除)

class BuffPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        # 1. 从配置读取值
        self.input_path = tk.StringVar(value=Config.get("buff_input", ""))
        self.output_dir = tk.StringVar(value=Config.get("buff_output", ""))
        
        # 【修改点】默认路径改为当前文件夹下的 RADVideo\radvideo64.exe
        # os.getcwd() 获取当前运行目录
        default_rad = os.path.join(os.getcwd(), "RADVideo", "radvideo64.exe")
        self.rad_path = tk.StringVar(value=Config.get("buff_rad", default_rad))
        
        # --- 状态变量 ---
        # 【核心修改】新增影子变量，存储程序在后台生成的临时文件路径
        # input_path 显示原文件路径，actual_processed_path 存储裁剪/抽帧后的路径
        self.actual_processed_path = None 
        
        self.last_generated_file = None 
        self.temp_video_path = None
        
        self.selected_category = tk.StringVar(value=Config.get("buff_category", ""))
        self.selected_job_name = tk.StringVar(value=Config.get("buff_job", ""))
        self.target_filename_preview = tk.StringVar(value="等待选择...")
        self.effect_mode = tk.StringVar(value=Config.get("buff_effect", "简单 Cut-in"))

        self.create_widgets()
        
        # 2. 触发联动，确保职业列表正确回显
        if self.selected_category.get():
            self.on_category_change(None) # 填充职业列表
            # 重新设置职业（因为 on_category_change 会清空职业）
            saved_job = Config.get("buff_job", "")
            if saved_job in self.cb_job['values']:
                self.cb_job.set(saved_job)
                self.update_preview_label(None)
        
        # 绑定输入框变化事件：如果用户手动修改了路径，必须清空影子变量，防止逻辑错乱
        self.input_path.trace("w", self.on_input_changed)

    def create_widgets(self):
        tk.Label(self, text="BUFF替换", font=("微软雅黑", 16, "bold"), fg="#333").pack(pady=20)
        
        f_main = ttk.Frame(self, padding=20)
        f_main.pack(fill="both", expand=True)

        # 1. 设置
        f_set = ttk.LabelFrame(f_main, text=" 1. 工具设置 ", padding=10)
        f_set.pack(fill="x", pady=5)
        f_r = ttk.Frame(f_set)
        f_r.pack(fill="x")
        ttk.Label(f_r, text="RAD工具路径:", width=12).pack(side="left")
        ttk.Entry(f_r, textvariable=self.rad_path).pack(side="left", fill="x", expand=True)
        ttk.Button(f_r, text="浏览...", command=self.sel_rad).pack(side="left", padx=5)

        # 2. 职业与特效
        f_job = ttk.LabelFrame(f_main, text=" 2. 职业与特效 ", padding=10)
        f_job.pack(fill="x", pady=5)
        
        f_row1 = ttk.Frame(f_job)
        f_row1.pack(fill="x", pady=2)
        ttk.Label(f_row1, text="职业选择:", width=12).pack(side="left")
        self.cb_category = ttk.Combobox(f_row1, textvariable=self.selected_category, state="readonly", width=15)
        self.cb_category.pack(side="left", padx=2)
        self.cb_category['values'] = list(BUFF_DATA.keys())
        self.cb_category.bind("<<ComboboxSelected>>", self.on_category_change)
        
        self.cb_job = ttk.Combobox(f_row1, textvariable=self.selected_job_name, state="readonly", width=15)
        self.cb_job.pack(side="left", padx=2)
        self.cb_job.bind("<<ComboboxSelected>>", self.update_preview_label)
        
        ttk.Label(f_row1, textvariable=self.target_filename_preview, foreground="#e74c3c").pack(side="left", padx=10)

        f_row2 = ttk.Frame(f_job)
        f_row2.pack(fill="x", pady=5)
        ttk.Label(f_row2, text="动态特效:", width=12).pack(side="left")
        effect_list = ["简单 Cut-in", "无"]
        self.cb_effect = ttk.Combobox(f_row2, textvariable=self.effect_mode, values=effect_list, state="readonly")
        self.cb_effect.pack(side="left", fill="x", expand=True)
        self.cb_effect.bind("<<ComboboxSelected>>", self.on_effect_change)

        # 3. 来源
        f_io = ttk.LabelFrame(f_main, text=" 3. 图片来源与保存 ", padding=10)
        f_io.pack(fill="x", pady=5)
        
        f_i = ttk.Frame(f_io)
        f_i.pack(fill="x", pady=5)
        ttk.Label(f_i, text="源文件:", width=12).pack(side="left")
        ttk.Entry(f_i, textvariable=self.input_path).pack(side="left", fill="x", expand=True)
        
        # 【修改点】 按钮改为 "选择源文件" 和 "选择序列文件夹"
        ttk.Button(f_i, text="📄 选择源文件 (图片/视频)", command=self.sel_source_file).pack(side="left", padx=2)
        ttk.Button(f_i, text="📂 选择序列文件夹", command=self.sel_img_dir).pack(side="left", padx=2)

        f_o = ttk.Frame(f_io)
        f_o.pack(fill="x", pady=5)
        ttk.Label(f_o, text="保存位置:", width=12).pack(side="left")
        ttk.Entry(f_o, textvariable=self.output_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(f_o, text="选择...", command=self.sel_out_dir).pack(side="left", padx=5)

        # 4. 运行 & 预览
        f_run = ttk.Frame(f_main)
        f_run.pack(pady=15, fill="x")
        
        ttk.Button(f_run, text="🚀 生成动态 BK2 视频", command=self.start_build).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(f_run, text="▶️ 播放/预览", command=self.preview_video, width=15).pack(side="right", padx=2)

        self.log_text = tk.Text(f_main, height=8, bg="#f0f0f0", font=("Consolas", 9), state="disabled")
        self.log_text.pack(fill="both", expand=True)

    # --- 逻辑处理 ---
    def on_input_changed(self, *args):
        # 只要输入框变了，就认为之前的临时文件无效了
        if self.actual_processed_path:
            # 可以选择这里是否立即删除旧文件，为了保险起见，只重置变量
            self.actual_processed_path = None
            # self.log("ℹ️ 检测到路径手动变更，将使用新路径作为源。")

    def on_effect_change(self, event):
        Config.set("buff_effect", self.effect_mode.get())

    def on_category_change(self, event):
        cat = self.selected_category.get()
        Config.set("buff_category", cat) # 保存大类
        if cat in BUFF_DATA:
            self.cb_job['values'] = list(BUFF_DATA[cat].keys())
            self.cb_job.set("")
            self.target_filename_preview.set("")
        else:
            self.cb_job['values'] = []

    def update_preview_label(self, event):
        cat = self.selected_category.get()
        name = self.selected_job_name.get()
        if cat in BUFF_DATA and name in BUFF_DATA[cat]:
            self.target_filename_preview.set(f"-> {BUFF_DATA[cat][name]}.bk2")
            Config.set("buff_job", name) # 保存具体职业

    def sel_rad(self):
        p = filedialog.askopenfilename(filetypes=[("Exe", "*.exe")])
        if p: 
            self.rad_path.set(p)
            Config.set("buff_rad", p)

    def sel_img_dir(self):
        p = filedialog.askdirectory()
        if p: 
            self.input_path.set(p)
            self.actual_processed_path = None # 文件夹模式不需要临时路径
            if not self.output_dir.get(): self.output_dir.set(p)
            self.log(f"📂 已选择文件夹模式(不适用特效): {os.path.basename(p)}")
            Config.set("buff_input", p)
            Config.set("buff_output", self.output_dir.get())

    # ---------------------------------------------------------
    # 【核心修改】整合后的选择文件逻辑
    # ---------------------------------------------------------
    def sel_source_file(self):
        # 扩展文件过滤器，加入视频格式
        file_types = [
            ("All Supported", "*.png;*.jpg;*.bmp;*.mp4;*.avi;*.webm;*.mov;*.gif"),
            ("Images", "*.png;*.jpg;*.bmp"),
            ("Videos", "*.mp4;*.avi;*.webm;*.mov;*.gif")
        ]
        p = filedialog.askopenfilename(filetypes=file_types)
        
        if not p: return

        # 【核心】UI 只显示原文件路径
        self.input_path.set(p)
        self.actual_processed_path = None # 重置
        
        # 保存选择的路径
        Config.set("buff_input", p)
        if not self.output_dir.get():
            out_d = os.path.dirname(p)
            self.output_dir.set(out_d)
            Config.set("buff_output", out_d)
        
        ext = os.path.splitext(p)[1].lower()
        if ext in ['.mp4', '.avi', '.webm', '.mov', '.gif', '.mkv']:
            if not HAS_MOVIEPY:
                messagebox.showerror("错误", "处理视频需要安装 moviepy 库！\n请运行: pip install moviepy")
                return
            
            self.temp_video_path = p 
            VideoSettingsDialog(self.winfo_toplevel(), p, self.on_video_config_done)
            
        # --- 情况 B: 普通图片 ---
        else:
            ManualCropper(self.winfo_toplevel(), p, callback=self.on_crop_finished)

    # --- 视频设置完成后的回调 ---
    def on_video_config_done(self, settings):
        if not settings: return
        
        # 1. 准备临时文件夹
        temp_dir = os.path.join(os.getcwd(), "_temp_video_frames")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        self.log(f"🎬 正在后台处理视频...")
        self.update_idletasks()
        
        # 2. 执行转换
        success, msg = VideoProcessor.process_video_to_frames(
            self.temp_video_path, 
            temp_dir, 
            target_frames=settings['frames'],
            crop_box=settings['crop_box']
        )
        
        if success:
            # 【核心】只更新内部影子变量，不改变 UI 上的路径
            self.actual_processed_path = temp_dir
            self.log(f"✅ 视频预处理完成！(临时路径已记录)")
            self.log(f"ℹ️ 点击 [生成] 按钮即可开始制作。")
        else:
            messagebox.showerror("处理失败", msg)
            self.log(f"❌ 失败: {msg}")

    def on_crop_finished(self, pil_image):
        try:
            pil_image = pil_image.convert("RGBA")
            temp_name = "temp_cropped_buff_source.png"
            temp_path = os.path.join(os.getcwd(), temp_name)
            pil_image.save(temp_path)
            
            # 【核心】只更新内部影子变量
            self.actual_processed_path = temp_path
            self.log(f"✅ 图片裁剪完成！(临时路径已记录)")
            self.log(f"ℹ️ 点击 [生成] 按钮即可开始制作。")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")

    def sel_out_dir(self):
        p = filedialog.askdirectory()
        if p: 
            self.output_dir.set(p)
            Config.set("buff_output", p)

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", str(msg) + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # --- 预览功能 (保持不变) ---
    def preview_video(self):
        # 1. 确定目标文件
        target_file = self.last_generated_file
        
        # 如果没有记录，尝试根据当前选项推断
        if not target_file or not os.path.exists(target_file):
            out_dir = self.output_dir.get()
            cat = self.selected_category.get()
            job = self.selected_job_name.get()
            if out_dir and cat and job:
                if cat in BUFF_DATA and job in BUFF_DATA[cat]:
                    code = BUFF_DATA[cat][job]
                    target_file = os.path.join(out_dir, f"{code}.bk2")
        
        # 2. 检查文件是否存在
        if not target_file or not os.path.exists(target_file):
            messagebox.showwarning("无法预览", f"找不到生成的 .bk2 文件：\n{target_file}\n\n请先点击[生成]按钮。")
            return

        # 3. 寻找播放器
        rad_dir = os.path.dirname(self.rad_path.get())
        player_candidates = ["bink2play.exe", "binkplay.exe"] # 优先用 bink2play
        real_player = None
        for p_name in player_candidates:
            p_path = os.path.join(rad_dir, p_name)
            if os.path.exists(p_path):
                real_player = p_path
                break
        
        # 4. 执行播放
        try:
            target_file_abs = os.path.abspath(target_file)
            self.log(f"▶️ 正在播放: {os.path.basename(target_file_abs)}")
            
            if real_player:
                cmd_str = f'"{real_player}" "{target_file_abs}" /L'
                subprocess.Popen(cmd_str, shell=False)
            else:
                self.log("⚠️ 未找到专用播放器，尝试使用系统默认方式...")
                os.startfile(target_file_abs)
                
        except Exception as e:
            messagebox.showerror("启动失败", f"无法预览。\n错误信息: {e}")

    def start_build(self):
        rad = self.rad_path.get()
        
        # 【核心逻辑】确定真实的源路径
        # 如果有后台处理好的临时路径，优先用它；否则用输入框的路径
        real_src_path = self.actual_processed_path if self.actual_processed_path else self.input_path.get()
        
        out_dir = self.output_dir.get()
        cat = self.selected_category.get()
        job = self.selected_job_name.get()

        if not os.path.exists(rad): return messagebox.showerror("错误", "RAD工具路径无效！")
        if not real_src_path or not os.path.exists(real_src_path): return messagebox.showerror("错误", "源文件处理后的临时文件不存在！请重新处理。")
        if not cat or not job: return messagebox.showerror("错误", "请先选择职业！")
        if not out_dir: return messagebox.showerror("错误", "请设置保存位置！")

        file_code = BUFF_DATA[cat][job]
        final_bk2_path = os.path.join(out_dir, f"{file_code}.bk2")
        effect = self.effect_mode.get()
        
        self.last_generated_file = final_bk2_path
        
        # 传入 real_src_path
        threading.Thread(target=self.run_thread, args=(real_src_path, final_bk2_path, rad, effect)).start()

    def run_thread(self, src_path, dst_path, rad, effect_mode):
        self.log(f">>> 开始生成: {os.path.basename(dst_path)}")
        temp_dir = os.path.join(os.getcwd(), "_temp_buff_frames")
        
        try:
            target_folder = src_path
            
            input_data = None
            # 1. 读取源数据 
            # (此时 src_path 指向的是 actual_processed_path，即预处理好的 PNG 或 文件夹)
            if os.path.isfile(src_path):
                input_data = Image.open(src_path).convert("RGBA")
            elif os.path.isdir(src_path):
                pngs = glob.glob(os.path.join(src_path, "*.png"))
                try: pngs.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
                except: pngs.sort()
                
                if not pngs:
                    self.log("❌ 错误: 文件夹内没有PNG图片")
                    return
                
                input_data = []
                for p in pngs:
                    input_data.append(Image.open(p).convert("RGBA"))
                self.log(f"📂 读取到序列帧: {len(input_data)} 张")
            
            # 2. 生成特效
            if input_data:
                if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
                os.makedirs(temp_dir)
                
                self.log(f"✨ 正在渲染特效: {effect_mode} ...")
                
                target_frame_count = len(input_data) if isinstance(input_data, list) else 18
                frames = EffectFactory.generate(input_data, effect_mode, frames=target_frame_count)
                
                for i, frame in enumerate(frames):
                    frame.save(os.path.join(temp_dir, f"frame_{i:04d}.png"))
                
                target_folder = temp_dir
            
            # 3. 调用 BK2 生成器
            success = build_hidden_bk2(target_folder, dst_path, rad, status_callback=self.log)
            if success:
                messagebox.showinfo("成功", f"生成完毕！\n您可以点击[预览]按钮查看效果。")
                self.log(">>> ✅ 任务完成")
            else:
                self.log(">>> ❌ 任务失败")
                
        except Exception as e:
            self.log(f"❌ 异常，请先抽帧处理: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 1. 清理【特效生成过程】中的中间文件 (必须删，否则下次残留)
            if os.path.exists(temp_dir):
                try: shutil.rmtree(temp_dir)
                except: pass
            
            # 【核心修改】
            # 移除了清理 src_path (临时源文件) 的代码。
            # 这样 self.actual_processed_path 依然有效，
            # 下次点击生成时，依然会使用预处理好的图片/帧，而不会回滚去读 MP4。