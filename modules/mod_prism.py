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

from common import Config, check_path_safety, get_resource_path,PRESET_COLORS,normalize_hex,hex_to_rgb,create_lut_from_colors,rgb_to_hsv_hue,CompareViewer

HAS_PYDNFEX = False
try:
    from pydnfex.npk import NPK
    from pydnfex.img import IMGFactory, ImageLink
    HAS_PYDNFEX = True
except ImportError:
    print("⚠️ 警告: 未检测到库，【调色盘】功能将不可用。")


# =========================================================================
# 【新增】独立的多进程工作函数 (必须放在类外面)
# =========================================================================
def process_one_npk_task(task_data):
    """
    单个 NPK 处理任务，将在子进程中运行。
    task_data 包含所有需要的参数 (避免传递 self)
    """
    try:
        # 1. 解包参数
        (npk_path, out_dir, patcher_exe, 
         mode, lut, target_hue, 
         overlay, overlay_tex_path, density, scale, eff_rgb, algo_rgb, 
         cos_params, remove_black) = task_data

        fname = os.path.basename(npk_path)
        pid = os.getpid() # 获取当前进程ID
        
        # 为每个进程创建独立的临时目录，避免冲突
        temp_root = os.path.join(os.getcwd(), f"_temp_proc_{pid}")
        if os.path.exists(temp_root): shutil.rmtree(temp_root)
        os.makedirs(temp_root)

        # 加载自定义贴图 (因为Image对象不能跨进程传递，所以传路径在子进程重新加载)
        custom_tex_img = None
        if overlay_tex_path and os.path.exists(overlay_tex_path):
            try:
                img = Image.open(overlay_tex_path).convert("RGBA")
                if remove_black:
                    dt = np.array(img)
                    mask = (dt[:,:,0]<30) & (dt[:,:,1]<30) & (dt[:,:,2]<30)
                    dt[:,:,3][mask] = 0
                    img = Image.fromarray(dt)
                custom_tex_img = img
            except: pass

        # 开始处理 NPK
        mod_cnt = 0
        with open(npk_path, 'rb') as f:
            npk = NPK.open(f)
            npk.load_all()
            valid_imgs = [x for x in npk.files if x.name.lower().endswith('.img')]
            
            for file_in in valid_imgs:
                sname = file_in.name.replace("/", "_")
                wdir = os.path.join(temp_root, sname)
                os.makedirs(wdir, exist_ok=True)
                
                try:
                    img_obj = IMGFactory.open(BytesIO(file_in.data))
                    csv = []
                    valid_frame_count = 0
                    
                    for i, frame in enumerate(img_obj.images):
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
                            ox = getattr(current_f, 'x', getattr(current_f, 'pos_x', 0))
                            oy = getattr(current_f, 'y', getattr(current_f, 'pos_y', 0))
                            
                            # 核心处理
                            res = ImageProcessor.process(
                                pil_img, mode, lut, target_hue,
                                overlay_type=overlay,
                                overlay_texture=custom_tex_img,
                                overlay_density=density,
                                overlay_scale=scale,
                                overlay_rgb=eff_rgb,
                                target_rgb=algo_rgb,
                                cosine_params=cos_params
                            )
                            
                            p_out = os.path.join(wdir, f"{i}.png")
                            res.save(p_out)
                            # CSV 使用绝对路径
                            csv.append(f"{os.path.abspath(p_out)},{ox},{oy}")
                            valid_frame_count += 1
                        except:
                            # 兜底
                            dummy = os.path.join(wdir, f"{i}_d.png")
                            Image.new("RGBA", (1,1),(0,0,0,0)).save(dummy)
                            csv.append(f"{os.path.abspath(dummy)},0,0")

                    if valid_frame_count > 0:
                        m_path = os.path.join(temp_root, "data.csv") # 复用文件名
                        i_path = os.path.join(temp_root, "new.img")
                        
                        # 写入 CSV (使用 gbk 兼容旧版)
                        with open(m_path, 'w', encoding="gbk") as fcsv: 
                            fcsv.write('\n'.join(csv))
                            
                        # 调用 Patcher (注意 cwd 切换)
                        si = subprocess.STARTUPINFO()
                        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                        
                        # 为了安全，把 patcher 复制过来或者使用绝对路径
                        # 这里假设 patcher_exe 是绝对路径
                        sub = subprocess.run(
                            [patcher_exe, "data.csv", "new.img"], 
                            cwd=temp_root, # 【关键】在子进程的独立目录运行
                            capture_output=True, 
                            startupinfo=si
                        )
                        
                        if sub.returncode == 0 and os.path.exists(i_path):
                            with open(i_path, 'rb') as nf: file_in.set_data(nf.read())
                            mod_cnt += 1
                            
                except Exception as img_err:
                    print(f"[{pid}] IMG Error: {img_err}")
                    pass # 继续下一个 img
                
                # 清理 img 临时文件
                try: shutil.rmtree(wdir) 
                except: pass

            # 保存 NPK
            if mod_cnt > 0:
                dst = os.path.join(out_dir, "%" + fname)
                with open(dst, 'wb') as fo: npk.save(fo)
                
        # 任务结束，清理整个临时目录
        try: shutil.rmtree(temp_root)
        except: pass
        
        return (True, f"完成: {fname} (修改 {mod_cnt} 个文件)")

    except Exception as e:
        # 出错也要尝试清理
        try: shutil.rmtree(temp_root)
        except: pass
        return (False, f"失败: {os.path.basename(npk_path)} -> {str(e)}")

        
class OverlayGenerator:
    
    # --- 辅助：闪电路径 (保持不变) ---
    @staticmethod
    def _get_lightning_nodes(p1, p2, displacement):
        if displacement < 5: return [p1, p2]
        mid_x = (p1[0] + p2[0]) / 2 + (random.random() - 0.5) * displacement
        mid_y = (p1[1] + p2[1]) / 2 + (random.random() - 0.5) * displacement
        mid = (mid_x, mid_y)
        return OverlayGenerator._get_lightning_nodes(p1, mid, displacement / 1.8) + \
               OverlayGenerator._get_lightning_nodes(mid, p2, displacement / 1.8)

    # --- 辅助：粒子缩放 (保持不变) ---
    @staticmethod
    def _prepare_custom_stamp(source_img, scale_factor):
        BASE_SIZE = 64
        w, h = source_img.size
        aspect = w / h
        if w > h:
            target_w = int(BASE_SIZE * scale_factor)
            target_h = int(target_w / aspect)
        else:
            target_h = int(BASE_SIZE * scale_factor)
            target_w = int(target_h * aspect)
        target_w = max(1, target_w)
        target_h = max(1, target_h)
        stamp = source_img.resize((target_w, target_h), resample=Image.NEAREST)
        return stamp

    # --- 新增辅助：安全的非循环位移 ---
    @staticmethod
    def _safe_shift(arr, direction):
        """
        移动数组，移出的部分丢弃并填0，防止像素从另一边飘回来
        """
        rows, cols = arr.shape
        result = np.zeros_like(arr)
        
        if direction == 'up':
            result[:-1, :] = arr[1:, :]  # 整体上移，底部填0
        elif direction == 'down':
            result[1:, :] = arr[:-1, :]  # 整体下移，顶部填0
        elif direction == 'left':
            result[:, :-1] = arr[:, 1:]  # 整体左移，右边填0
        elif direction == 'right':
            result[:, 1:] = arr[:, :-1]  # 整体右移，左边填0
            
        return result
        
    # --- 【恢复】辅助：生成发光星星 ---
    @staticmethod
    def _create_sparkle_stamp(size, color):
        canvas_size = int(size * 2.5)
        w, h = canvas_size, canvas_size
        cx, cy = w // 2, h // 2
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        r, g, b = color
        
        # 光晕
        for i in range(5):
            radius = (w * 0.4) * (1 - i/5)
            alpha = int(30 + (i/5) * 100)
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(r, g, b, alpha))

        # 十字星芒
        spike_len = w * 0.45
        spike_width = w * 0.12
        color_mid = (r, g, b, 150)
        color_core = (255, 255, 255, 255)
        
        def draw_spike(d, l, wid, c):
            d.polygon([(cx, cy-l), (cx+wid, cy), (cx, cy+l), (cx-wid, cy)], fill=c)
            d.polygon([(cx-l, cy), (cx, cy-wid), (cx+l, cy), (cx, cy+wid)], fill=c)
        
        draw_spike(draw, spike_len, spike_width, color_mid)
        img = img.filter(ImageFilter.GaussianBlur(w * 0.08)) # 需要 import ImageFilter
        
        # 核心高亮
        core_layer = Image.new("RGBA", (w, h), (0,0,0,0))
        c_draw = ImageDraw.Draw(core_layer)
        draw_spike(c_draw, spike_len * 0.6, spike_width * 0.5, (255, 255, 255, 230))
        c_draw.ellipse((cx-w*0.05, cy-w*0.05, cx+w*0.05, cy+w*0.05), fill=color_core)
        core_layer = core_layer.filter(ImageFilter.GaussianBlur(w * 0.02))
        
        img.alpha_composite(core_layer)
        return img.resize((size, size), resample=Image.LANCZOS)

    # --- 【恢复】辅助：生成樱花 ---
    @staticmethod
    def _create_sakura_stamp(size, color):
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        r, g, b = color
        # 外晕
        draw.ellipse((0, 0, size, size), fill=(r, g, b, 50))
        # 核心花瓣
        inner = (size*0.2, size*0.2, size*0.8, size*0.8)
        draw.ellipse(inner, fill=(r, g, b, 200))
        # 缺口
        draw.polygon([(cx, 0), (cx+size*0.2, size*0.3), (cx-size*0.2, size*0.3)], fill=(0,0,0,0))
        return img
        
    @staticmethod
    def generate(target_img, mode_name, custom_texture=None, density=50, scale=1.0, color_rgb=None):
        width, height = target_img.size
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        
        # 1. 坐标提取 (仅用于粒子/闪电)
        valid_coords = []
        if mode_name in ["custom_particle", "lightning", "sakura", "starlight"]:
            data = np.array(target_img.convert("RGBA"))
            alpha_channel = data[:, :, 3]
            max_rgb = np.max(data[:, :, :3], axis=2)
            valid_y, valid_x = np.where((alpha_channel > 20) & (max_rgb > 20))
            if len(valid_x) == 0: return overlay
            valid_coords = list(zip(valid_x, valid_y))
            area = len(valid_coords)

        # ==================== 1. 自定义粒子 ====================
        if mode_name == "sakura":
            # 默认粉色，如果传了颜色则用传入的
            base_color = color_rgb if color_rgb else (255, 192, 203) 
            base_size = int(20 * scale)
            if base_size < 5: base_size = 5
            
            stamp = OverlayGenerator._create_sakura_stamp(base_size, base_color)
            
            # 根据密度计算数量
            num_petals = max(3, int((area / 400) * (density / 50.0)))
            
            for _ in range(num_petals):
                cx, cy = random.choice(valid_coords)
                # 随机旋转和微调大小
                curr = stamp.copy().rotate(random.randint(0, 360))
                s_rand = random.uniform(0.7, 1.3)
                final_w, final_h = int(curr.width * s_rand), int(curr.height * s_rand)
                curr = curr.resize((final_w, final_h))
                
                w_s, h_s = curr.size
                overlay.paste(curr, (cx - w_s//2, cy - h_s//2), curr)

        # ==================== 2. 星光 (Starlight) ====================
        elif mode_name == "starlight":
            base_color = color_rgb if color_rgb else (255, 255, 255)
            
            # 预生成三种大小的星星
            sizes = [int(45*scale), int(25*scale), int(15*scale)]
            stamps = [OverlayGenerator._create_sparkle_stamp(s, base_color) for s in sizes]
            
            num_stars = max(2, int((area / 300) * (density / 50.0)))
            
            for _ in range(num_stars):
                cx, cy = random.choice(valid_coords)
                rnd = random.random()
                if rnd < 0.6:   stamp = stamps[2] # 小
                elif rnd < 0.9: stamp = stamps[1] # 中
                else:           stamp = stamps[0] # 大
                
                curr = stamp.copy()
                # 随机透明度
                alpha_scale = random.uniform(0.6, 1.0)
                r,g,b,a = curr.split()
                a = a.point(lambda i: int(i * alpha_scale))
                curr.putalpha(a)
                
                w_s, h_s = curr.size
                overlay.paste(curr, (cx - w_s//2, cy - h_s//2), curr)
            
            # 增加一些微小的尘埃点
            draw_d = ImageDraw.Draw(overlay)
            num_dust = num_stars * 4
            for _ in range(num_dust):
                dx, dy = random.choice(valid_coords)
                d_sz = random.randint(1, max(2, int(3*scale)))
                # 颜色稍微扰动
                rc = (
                    min(255, base_color[0] + random.randint(-20, 20)),
                    min(255, base_color[1] + random.randint(-20, 20)),
                    min(255, base_color[2] + random.randint(-20, 20)),
                    random.randint(50, 200)
                )
                off_x = dx + random.randint(-5, 5)
                off_y = dy + random.randint(-5, 5)
                draw_d.ellipse((off_x, off_y, off_x+d_sz, off_y+d_sz), fill=rc)
                
        elif mode_name == "custom_particle" and custom_texture:
            stamp = OverlayGenerator._prepare_custom_stamp(custom_texture, scale)
            base_count = area / 800 
            actual_count = int(base_count * (density / 25.0)) 
            actual_count = max(1, actual_count) 
            for _ in range(actual_count):
                cx, cy = random.choice(valid_coords)
                curr = stamp
                rand_s = random.uniform(0.8, 1.2)
                cur_w = int(curr.width * rand_s)
                cur_h = int(curr.height * rand_s)
                if cur_w > 0 and cur_h > 0:
                     curr = curr.resize((cur_w, cur_h), resample=Image.NEAREST)
                w_s, h_s = curr.size
                overlay.paste(curr, (cx - w_s//2, cy - h_s//2), curr)

        # ==================== 2. 闪电 ====================
        elif mode_name == "lightning":
            base_rgb = color_rgb if color_rgb else (150, 50, 255)
            num_bolts = max(1, int(area / 2000 * (density / 20.0)))
            if num_bolts > 8: num_bolts = 8
            lightning_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw_L = ImageDraw.Draw(lightning_layer)
            for _ in range(num_bolts):
                p1 = random.choice(valid_coords)
                p2 = random.choice(valid_coords)
                dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                if dist < 20: continue 
                points = OverlayGenerator._get_lightning_nodes(p1, p2, dist/4)
                if len(points) < 2: continue
                width_main = max(1, int(4 * scale))
                width_core = max(1, int(1 * scale))
                draw_L.line(points, fill=(*base_rgb, 255), width=width_main, joint="curve")
                draw_L.line(points, fill=(255, 255, 255, 255), width=width_core, joint="curve")
            original_alpha = target_img.split()[-1]
            lightning_layer.putalpha(ImageChops.multiply(original_alpha, lightning_layer.split()[-1]))
            overlay = lightning_layer

        # ==================== 3. 逻辑描边 (无漂移版) ====================
        elif mode_name == "outer_glow":
            base_rgb = color_rgb if color_rgb else (255, 215, 0)
            
            src_arr = np.array(target_img.convert("RGBA"))
            alpha = src_arr[:, :, 3].astype(int) 
            
            # 二值化 Mask
            mask = (alpha > 10).astype(int) 
            
            # 决定层数 (1-5层)
            layers = max(1, int(2 * scale))
            if layers > 5: layers = 5
            
            # 结果画布
            final_glow = np.zeros((height, width, 4), dtype=np.uint8)
            
            current_mask = mask.copy()
            
            # 循环生成描边
            for i in range(layers):
                opacity = int(255 * (0.6 ** i)) # 透明度递减
                
                # --- 核心修复：使用非循环位移 ---
                up    = OverlayGenerator._safe_shift(current_mask, 'up')
                down  = OverlayGenerator._safe_shift(current_mask, 'down')
                left  = OverlayGenerator._safe_shift(current_mask, 'left')
                right = OverlayGenerator._safe_shift(current_mask, 'right')
                
                # 合并 (逻辑或)
                dilated = np.maximum.reduce([current_mask, up, down, left, right])
                
                # 只有新长出来的部分才是这一层的描边
                rim = dilated - current_mask
                
                # 上色
                y_idxs, x_idxs = np.where(rim == 1)
                
                if len(y_idxs) > 0:
                    # 只有当该像素点还没有被画过时才画（避免层与层之间的干扰，虽然从内向外画其实无所谓）
                    # 只要 alpha 为 0 就说明是空的
                    empty_mask = (final_glow[y_idxs, x_idxs, 3] == 0)
                    
                    # 过滤出空的坐标
                    valid_y = y_idxs[empty_mask]
                    valid_x = x_idxs[empty_mask]
                    
                    if len(valid_y) > 0:
                        final_glow[valid_y, valid_x, 0] = base_rgb[0]
                        final_glow[valid_y, valid_x, 1] = base_rgb[1]
                        final_glow[valid_y, valid_x, 2] = base_rgb[2]
                        final_glow[valid_y, valid_x, 3] = opacity
                
                # 更新Mask，向外推进
                current_mask = dilated
            
            overlay = Image.fromarray(final_glow, "RGBA")

        return overlay

class CosineGradientGenerator:
    """
    程序化余弦渐变生成器
    模式 A: 预设 (纯数学公式 A + B * cos...)
    模式 B: 自定义 (分段余弦插值查表法)
    """
    # 预设参数 [A, B, C, D] - 仅用于预设模式
    PRESETS = {
        "Rainbow (彩虹)": [[0.5,0.5,0.5], [0.5,0.5,0.5], [1.0,1.0,1.0], [0.0,0.33,0.67]],
        "Cyberpunk (赛博)": [[0.5,0.5,0.5], [0.5,0.5,0.5], [1.0,1.0,1.0], [0.5,0.20,0.25]],
        "Gold (黑金)": [[0.5,0.5,0.5], [0.5,0.5,0.5], [1.0,1.0,0.5], [0.0,0.15,0.20]],
        "Aurora (极光)": [[0.5,0.5,0.5], [0.5,0.5,0.5], [1.0,0.5,1.5], [0.5,0.2,0.25]],
        "Neon (霓虹)": [[0.8,0.5,0.4], [0.2,0.4,0.2], [2.0,1.0,1.0], [0.0,0.25,0.25]],
    }

    @staticmethod
    def _create_cosine_lut(colors, size=256):
        """
        核心算法：生成分段余弦插值 LUT
        输入: colorsList [(r,g,b), (r,g,b)...]
        输出: (256, 3) 的 numpy 数组
        """
        if not colors: return np.zeros((size, 3))
        if len(colors) == 1: return np.tile(colors[0], (size, 1))

        num_segments = len(colors) - 1
        segment_len = size / num_segments
        
        lut = np.zeros((size, 3))
        
        for i in range(num_segments):
            # 起始颜色和终点颜色
            c1 = np.array(colors[i])
            c2 = np.array(colors[i+1])
            
            # 计算当前段在 LUT 中的起止索引
            idx_start = int(i * segment_len)
            idx_end = int((i + 1) * segment_len)
            
            # 防止最后一段越界或填不满
            if i == num_segments - 1:
                idx_end = size
                
            length = idx_end - idx_start
            if length <= 0: continue

            # === 核心数学公式：分段余弦插值 ===
            # 1. 生成线性进度 x (0.0 -> 1.0)
            x = np.linspace(0, 1, length)
            
            # 2. 将线性进度映射为余弦曲线进度 mu
            # mu = (1 - cos(x * π)) / 2
            # 结果是一条 S 形曲线：两头慢，中间快
            mu = (1 - np.cos(x * np.pi)) / 2.0
            
            # 3. 扩展维度以进行 RGB 广播计算 (length, 1)
            mu = mu[:, np.newaxis]
            
            # 4. 插值: Color = C1 * (1 - mu) + C2 * mu
            segment_colors = c1 * (1.0 - mu) + c2 * mu
            
            # 填入 LUT
            lut[idx_start:idx_end] = segment_colors
            
        return lut

    @staticmethod
    def generate(pil_img, scan_mode="diagonal", preset_name="Rainbow (彩虹)", 
                 frequency=1.0, offset=0.0, custom_colors=None):
        """
        custom_colors: 如果传入 RGB 列表 [(255,0,0), ...], 则启用自定义模式
        """
        img = pil_img.convert("RGBA")
        width, height = img.size
        arr = np.array(img)
        
        # 1. 提取原图亮度 (Luminosity Mask)
        # 解决黑底问题：亮度低的地方保持黑，亮度高的地方上色
        brightness = np.max(arr[:, :, :3], axis=2) / 255.0
        
        if np.sum(brightness) < 0.1: return pil_img

        # 2. 生成几何波形 t (0.0 - 1.0)
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xv, yv = np.meshgrid(x, y) 

        if scan_mode == "horizontal": t = xv
        elif scan_mode == "vertical": t = yv
        elif scan_mode == "diagonal": t = (xv + yv) * 0.5
        elif scan_mode == "radial":   t = np.sqrt((xv-0.5)**2 + (yv-0.5)**2) * 2.0
        elif scan_mode == "angular":  t = (np.arctan2(yv-0.5, xv-0.5)/np.pi)*0.5 + 0.5
        else: t = xv

        # 3. 颜色计算分支
        gradient_rgb = None
        
        # --- 分支 A: 用户自定义模式 (查表法 + 余弦插值) ---
        if custom_colors and len(custom_colors) > 0:
            # 【核心修复开始】: 自动闭环逻辑
            # 为了消除接缝，我们检查首尾颜色是否一致
            # 如果不一致，强制把“起点颜色”追加到“终点”后面，形成完美循环
            process_colors = list(custom_colors) # 复制一份，防止修改原数据
            
            if len(process_colors) >= 2:
                c_start = np.array(process_colors[0])
                c_end = np.array(process_colors[-1])
                # 计算欧氏距离，如果首尾颜色差异过大 (> 5)，就自动补一个
                if np.linalg.norm(c_start - c_end) > 5.0:
                    process_colors.append(process_colors[0])

            # 使用处理过的 process_colors 生成 LUT
            lut = CosineGradientGenerator._create_cosine_lut(process_colors, size=512)
            # 【核心修复结束】

            # 计算索引
            # 应用频率和偏移 -> 取小数部分(循环)
            t_final = (t * frequency + offset) % 1.0
            
            indices = (t_final * 511).astype(int)
            indices = np.clip(indices, 0, 511)
            
            # 查表取色
            gradient_rgb = lut[indices]
            
            # 归一化到 0-1
            if np.max(gradient_rgb) > 1.0:
                gradient_rgb /= 255.0

        # --- 分支 B: 预设模式 (纯数学公式) ---
        else:
            params = CosineGradientGenerator.PRESETS.get(preset_name, CosineGradientGenerator.PRESETS["Rainbow (彩虹)"])
            A, B, C, D = [np.array(p) for p in params]
            C = C * frequency # 调整频率
            
            t_exp = t[..., np.newaxis]
            gradient_rgb = A + B * np.cos(2 * np.pi * (C * t_exp + D + offset)) # 注意 offset 加在这里
            gradient_rgb = np.clip(gradient_rgb, 0.0, 1.0)

        # 4. 最终混合 (应用亮度蒙版)
        brightness_exp = brightness[..., np.newaxis]
        final_rgb = gradient_rgb * brightness_exp * 255.0
        
        final_arr = np.dstack((final_rgb.astype(np.uint8), arr[:, :, 3]))
        
        return Image.fromarray(final_arr, "RGBA")

class ImageProcessor:
    @staticmethod
    def _safe_offset(img_channel, x_offset, y_offset):
        """辅助函数：安全位移单通道图像（不卷绕，空白处填0）"""
        w, h = img_channel.size
        # 创建全黑底图
        new_img = Image.new("L", (w, h), 0)
        # 计算粘贴坐标
        paste_x = int(x_offset)
        paste_y = int(y_offset)
        # 粘贴原图，超出部分自动被截断，留空部分保持黑色
        new_img.paste(img_channel, (paste_x, paste_y))
        return new_img
    @staticmethod
    def process(pil_img, mode, lut=None, target_hue_val=0, 
                overlay_type="none", overlay_texture=None, overlay_density=50, overlay_scale=1.0, overlay_rgb=None, target_rgb=None, cosine_params=None):
        try:
            img = pil_img.convert("RGBA")
            alpha = img.split()[-1]
            rgb_img = img.convert("RGB")
            arr_rgb = np.array(rgb_img)
            result_data = None  
            result_rgba = img
            # 1. 调色 (逻辑保持不变)
            if mode == "none":
                result_data = arr_rgb
            # --- 【新增】故障色差 (Glitch) ---
            # 假设你在 algo_mode 里加了一个选项叫 "cosine_grad"

            elif mode == "cosine_grad": # <--- 【修改分支】
                if cosine_params:
                    # 从字典中提取参数
                    c_scan = cosine_params.get('scan', 'diagonal')
                    c_preset = cosine_params.get('preset', 'Rainbow (彩虹)')
                    c_freq = cosine_params.get('freq', 1.0)
                    c_offset = cosine_params.get('offset', 0.0)
                    c_colors = cosine_params.get('custom_colors', None) # 【新增】

                    result_rgba = CosineGradientGenerator.generate(
                        pil_img, 
                        scan_mode=c_scan,
                        preset_name=c_preset,
                        frequency=c_freq,
                        offset=c_offset,
                        custom_colors=c_colors # 【新增】传给生成器
                    )
                    result_data = None 
                else:
                    result_data = arr_rgb
            elif mode == "glitch":
                # 计算位移量：利用 overlay_density (1-100) 控制偏移像素
                # 范围大约 0 - 20 像素
                offset = max(1, int(overlay_density / 4.0))
                
                # 分离通道
                r, g, b, a = img.split()
                
                # 1. 位移 RGB 通道
                # 红：向左上移
                r_new = ImageProcessor._safe_offset(r, -offset, -offset)
                # 绿：保持不动 (或者微动)
                g_new = g 
                # 蓝：向右下移
                b_new = ImageProcessor._safe_offset(b, offset, offset)
                
                # 2. 【关键】位移 Alpha 通道并合并
                # 如果不移动Alpha，错位的颜色会被原来的透明区切掉
                a_r = ImageProcessor._safe_offset(a, -offset, -offset)
                a_b = ImageProcessor._safe_offset(a, offset, offset)
                
                # 合并Alpha：取最大值 (即并集)，保证所有颜色的像素都能显示
                # 利用 ImageChops.lighter (变亮模式) 等同于 max
                a_new = ImageChops.lighter(a, a_r)
                a_new = ImageChops.lighter(a_new, a_b)
                
                # 3. 合并回 RGBA
                result_rgba = Image.merge("RGBA", (r_new, g_new, b_new, a_new))
                
                # 这一步已经生成了 result_rgba，不需要后续的 putalpha
                # 标记 result_data 为 None 跳过通用处理
                result_data = None 
            elif mode == "transparent":
                return Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            # =========== 【修改 2】 新增线稿化逻辑 ===========
            elif mode == "line_art":
                # A. 转灰度 (用于提取边缘 + 获取亮度信息)
                gray = img.convert("L")
                
                # B. 提取边缘 (这是透明度蒙版的基础)
                edges = gray.filter(ImageFilter.FIND_EDGES)
                
                # C. 决定线条颜色 (核心修改)
                if lut is not None:
                    # --- 方案 1: 渐变模式 ---
                    # 根据原图的亮度 (gray) 在 LUT 中查找颜色
                    # 这样线条的颜色会随着原图的明暗变化
                    gray_arr = np.array(gray)
                    color_data = lut[gray_arr]
                    base_layer = Image.fromarray(color_data, "RGB")
                else:
                    # --- 方案 2: 单色模式 ---
                    fill_color = target_rgb if target_rgb else (255, 255, 255)
                    base_layer = Image.new("RGB", img.size, fill_color)
                
                # D. 计算 Alpha 通道 (保持不变)
                arr_edge = np.array(edges).astype(float)
                arr_alpha = np.array(alpha).astype(float)
                
                # 增强线条可见度
                new_alpha = (arr_edge / 255.0) * (arr_alpha / 255.0) * 255.0 * 2.0
                new_alpha = np.clip(new_alpha, 0, 255).astype(np.uint8)
                
                final_alpha = Image.fromarray(new_alpha, "L")
                
                # E. 组合：把计算出的透明度应用到颜色层上
                # 转为 RGBA 以便支持 putalpha
                if base_layer.mode != "RGBA":
                    base_layer = base_layer.convert("RGBA")
                    
                base_layer.putalpha(final_alpha)
                
                result_rgba = base_layer
            # ===============================================
            elif mode == "gray_map" and lut is not None:
                gray = np.array(img.convert("L"))
                result_data = lut[gray]
            elif mode == "max_rgb" and lut is not None:
                max_val = np.max(arr_rgb, axis=2)
                result_data = lut[max_val]
            elif mode == "target_hue":
                hsv_img = img.convert("HSV")
                h, s, v = hsv_img.split()
                new_h = Image.new("L", h.size, int(target_hue_val))
                result_data = np.array(Image.merge("HSV", (new_h, s, v)).convert("RGB"))
            elif mode == "tint" and lut is not None:
                orig_hsv = img.convert("HSV")
                _, _, v_orig = orig_hsv.split()
                brightness = np.max(arr_rgb, axis=2)
                mapped_rgb = lut[brightness]
                result_img = Image.merge("HSV", (Image.fromarray(mapped_rgb, "RGB").convert("HSV").split()[0], 
                                                 Image.fromarray(mapped_rgb, "RGB").convert("HSV").split()[1], 
                                                 v_orig)).convert("RGB")
                result_data = np.array(result_img)
            else:
                result_data = arr_rgb

            if result_data is not None:
                result_rgba = Image.fromarray(result_data, "RGB")
                result_rgba.putalpha(alpha)

            # 2. 纹理叠加
            if overlay_type and overlay_type != "none":
                overlay_layer = OverlayGenerator.generate(
                    target_img=result_rgba, 
                    mode_name=overlay_type, 
                    custom_texture=overlay_texture,
                    density=overlay_density,
                    scale=overlay_scale,
                    color_rgb=overlay_rgb
                )
                
                # 【核心】：Outer Glow 放在最底层 (Backlight)
                if overlay_type == "outer_glow":
                    # 创建空画布
                    base = Image.new("RGBA", result_rgba.size, (0, 0, 0, 0))
                    # 先画描边 (背景)
                    base = Image.alpha_composite(base, overlay_layer)
                    # 再画原图 (前景压住描边)
                    result_rgba = Image.alpha_composite(base, result_rgba)
                else:
                    # 其他特效覆盖在上面
                    result_rgba = Image.alpha_composite(result_rgba, overlay_layer)
            
            return result_rgba
        except Exception as e:
            print(f"Process Error: {e}")
            return pil_img

class GradientEditor(ttk.LabelFrame):
    def __init__(self, parent, title=" 🎨 颜色配置 "):
        super().__init__(parent, text=title, padding=10)
        self.color_list = [] 
        self.force_black = tk.BooleanVar(value=True) 
        self.create_ui()
        self.load_preset("🔥 火焰 (Fire)")

    def create_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", pady=2)
        ttk.Label(top, text="预设:").pack(side="left")
        self.combo_preset = ttk.Combobox(top, values=list(PRESET_COLORS.keys()), state="readonly", width=15)
        self.combo_preset.pack(side="left", padx=5)
        self.combo_preset.bind("<<ComboboxSelected>>", lambda e: self.load_preset(self.combo_preset.get()))
        self.chk_black = ttk.Checkbutton(top, text="自动黑底", variable=self.force_black, command=self.update_preview)
        self.chk_black.pack(side="right")
        self.canvas = tk.Canvas(self, height=35, bg="#333", highlightthickness=1, highlightbackground="gray")
        self.canvas.pack(fill="x", pady=8)
        self.canvas.bind("<Configure>", lambda e: self.update_preview())
        btns = ttk.Frame(self)
        btns.pack(fill="x")
        ttk.Button(btns, text="➕", width=3, command=self.add_color).pack(side="left", padx=1)
        ttk.Button(btns, text="↩️", width=3, command=self.undo_color).pack(side="left", padx=1)
        ttk.Button(btns, text="🗑️", width=3, command=self.clear_colors).pack(side="left", padx=1)
         # 【新增】批量输入按钮
        ttk.Button(btns, text="📝 颜色代码", width=10, command=self.input_batch_codes).pack(side="left", padx=5)
        self.lbl_info = ttk.Label(btns, text="..", font=("Arial", 8), foreground="gray")
        self.lbl_info.pack(side="right")

    def input_batch_codes(self):
        # 1. 把当前颜色列表转成字符串，方便用户修改
        current_str = ", ".join(self.color_list)
        
        # 2. 弹窗
        input_str = simpledialog.askstring(
            "批量颜色编辑", 
            "请输入颜色代码，用逗号分隔:\n(例如: #FF0000, #00FF00, #0000FF)", 
            parent=self, 
            initialvalue=current_str
        )
        
        if input_str is not None: # 如果用户没点取消
            raw_items = input_str.split(",")
            valid_colors = []
            for item in raw_items:
                v = normalize_hex(item)
                if v:
                    valid_colors.append(v)
            
            if valid_colors:
                self.color_list = valid_colors
                self.update_preview()
            else:
                # 如果用户清空了输入或者是无效的，可以视情况清空或报错
                if not input_str.strip():
                    self.color_list = []
                    self.update_preview()
                else:
                    messagebox.showwarning("提示", "未检测到有效的颜色代码")
                    
    def load_preset(self, name):
        if name in PRESET_COLORS:
            self.color_list = list(PRESET_COLORS[name])
            self.update_preview()

    def add_color(self):
        c = colorchooser.askcolor()[1]
        if c: self.color_list.append(c); self.update_preview()
    def undo_color(self):
        if self.color_list: self.color_list.pop(); self.update_preview()
    def clear_colors(self):
        self.color_list = []; self.update_preview()
    def update_preview(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        if w < 50: w = 800 
        lut = create_lut_from_colors(self.color_list, self.force_black.get())
        preview = np.tile(lut, (35, 1, 1))
        img = ImageTk.PhotoImage(Image.fromarray(preview, "RGB").resize((w, 35)))
        self.canvas.create_image(0, 0, image=img, anchor="nw")
        self.photo = img 
        cnt = len(self.color_list)
        self.lbl_info.config(text=f"{cnt} 色 (+黑)" if self.force_black.get() else f"{cnt} 色")

    def get_lut(self):
        return create_lut_from_colors(self.color_list, self.force_black.get())

class SingleColorPicker(ttk.LabelFrame):
    def __init__(self, parent, title=" 🌈 目标颜色选择 "):
        super().__init__(parent, text=title, padding=15)
        self.target_rgb = (255, 0, 0)
        self.target_hex = "#FF0000"
        self.create_ui()

    def create_ui(self):
        # 预览色块
        self.lbl_preview = tk.Label(self, bg=self.target_hex, width=10, height=2, relief="sunken")
        self.lbl_preview.pack(side="left", padx=10)
        
        # 色板选择按钮
        btn_pick = ttk.Button(self, text="🎨 点击选择颜色...", command=self.pick_color)
        btn_pick.pack(side="left", fill="x", expand=True, padx=(5, 2))
        
        # 【新增】代码输入按钮 (小按钮)
        btn_code = ttk.Button(self, text="📝 颜色代码", width=10, command=self.input_color_code)
        btn_code.pack(side="left", padx=(0, 5))

    def pick_color(self):
        c = colorchooser.askcolor(color=self.target_hex, title="选择颜色")[1]
        if c:
            self.apply_color(c)

    # 【新增】弹窗输入逻辑
    def input_color_code(self):
        # 弹窗询问，默认显示当前颜色
        code = simpledialog.askstring("输入颜色", "请输入 HEX 颜色代码 (例: #FF0000):", 
                                    parent=self, initialvalue=self.target_hex)
        if code:
            valid_hex = normalize_hex(code)
            if valid_hex:
                self.apply_color(valid_hex)
            else:
                messagebox.showwarning("格式错误", "颜色代码无效！\n请使用 #RRGGBB 格式 (如 #FF0000)")

    # 【新增】统一更新逻辑
    def apply_color(self, hex_code):
        self.target_hex = hex_code
        self.target_rgb = hex_to_rgb(hex_code)
        self.lbl_preview.config(bg=hex_code)
    
    def get_hue_value(self):
        return rgb_to_hsv_hue(self.target_rgb)
        
class PrismPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        # 1. 从配置读取初始值
        self.input_path = tk.StringVar(value=Config.get("prism_input", ""))
        self.output_dir = tk.StringVar(value=Config.get("prism_output", ""))
        self.algo_mode = tk.StringVar(value=Config.get("prism_mode", "max_rgb"))
        
        # --- 新增/修改变量 ---
        self.overlay_mode = tk.StringVar(value="none") # 下拉框变量
        self.custom_tex_path = tk.StringVar(value="")  # 自定义图片路径
        self.var_density = tk.IntVar(value=50)         # 密度/强度 (1-100)
        self.var_scale = tk.DoubleVar(value=1.0)       # 大小缩放 (0.1-3.0)
        self.var_remove_black = tk.BooleanVar(value=False)

        # 缓存加载的自定义图片对象
        self.cached_custom_img = None 

        if not HAS_PYDNFEX:
            tk.Label(self, text="❌ 缺少 pydnfex 库，此功能无法使用", fg="red", font=("微软雅黑", 14)).pack(pady=50)
            return

        self.create_widgets()
        
        # 恢复上次的颜色预设 (如果有)
        last_preset = Config.get("prism_preset", "🔥 火焰 (Fire)")
        if hasattr(self, 'editor'):
            self.editor.combo_preset.set(last_preset)
            self.editor.load_preset(last_preset)
        
        # 触发一次UI刷新，确保控件显示正确
        self.on_mode_change()
        self.on_overlay_change()

    def create_widgets(self):
        # ... (这部分 UI 代码保持不变，省略以节省空间) ...
        # ... 请保留原本的 create_widgets 内容 ...
        # 顶部标题 (模拟原来的 App title)
        tk.Label(self, text="阿拉德特效调色板", font=("微软雅黑", 16, "bold"), fg="#333").pack(pady=10)

        f_file = ttk.LabelFrame(self, text=" 1. 路径设置  请使用英文路径", padding=10)
        f_file.pack(fill="x", padx=10, pady=5)
        f_in = ttk.Frame(f_file); f_in.pack(fill="x", pady=2)
        ttk.Label(f_in, text="源路径:", width=8).pack(side="left") 
        ttk.Entry(f_in, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(f_in, text="选文件", width=8, command=self.sel_file).pack(side="left")
        ttk.Button(f_in, text="选目录", width=8, command=self.sel_dir).pack(side="left", padx=2)
        f_out = ttk.Frame(f_file); f_out.pack(fill="x", pady=2)
        ttk.Label(f_out, text="保存至:", width=8).pack(side="left") 
        ttk.Entry(f_out, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(f_out, text="📂 选择...", width=18, command=self.sel_out_dir).pack(side="left")
        
        # --- 算法与特效 ---
        f_algo = ttk.LabelFrame(self, text=" 2. 模式与特效 ", padding=10)
        f_algo.pack(fill="x", padx=10, pady=5)

        # 左侧：调色算法
        f_algo_l = ttk.Frame(f_algo)
        f_algo_l.pack(side="left", fill="y", padx=5)
        ttk.Label(f_algo_l, text="[调色算法]").pack(anchor="w")
        modes = [
            ("⚡ Max-RGB 映射 (鲜艳)", "max_rgb"),
            ("🌈 统一色相 (换色)", "target_hue"),
            ("🖌 混合染色 (光影)", "tint"),
            ("🎨 灰度映射 (平滑)", "gray_map"),
            ("✒️ 线稿化 (Line Art)", "line_art"),
            ("📺 故障色差 (Glitch)", "glitch"),
            ("🌊 程序化幻彩 (Cosine Gradient)", "cosine_grad"),
            ("🚫 原图 (不调色)", "none"),  # 【新增】无模式
        ]
        for text, val in modes:
            ttk.Radiobutton(f_algo_l, text=text, variable=self.algo_mode, value=val, command=self.on_mode_change).pack(anchor="w", pady=1)

        ttk.Separator(f_algo, orient="vertical").pack(side="left", fill="y", padx=15)
        
        # 右侧：纹理特效
        f_algo_r = ttk.Frame(f_algo)
        f_algo_r.pack(side="left", fill="both", expand=True, padx=5)
        
        ttk.Label(f_algo_r, text="[🌸 额外纹理/粒子]").pack(anchor="w")
        
        # 模式下拉框
        self.cb_overlay = ttk.Combobox(f_algo_r, textvariable=self.overlay_mode, state="readonly", width=25)
        self.overlay_map = {
            "none": "🚫 无 (None)",
            "sakura":          "🌸 漫天樱花 (Sakura)",    # 【新增】
            "starlight":       "✨ 璀璨星光 (Starlight)", # 【新增】
            "custom_particle": "📂 自定义: 粒子散布 (Scatter)",
            "lightning":       "⚡ 内置: 雷霆万钧 (Lightning)",
            "outer_glow":      "🌟 内置: 像素描边 (Outline)"
        }
        self.cb_overlay['values'] = list(self.overlay_map.values())
        self.cb_overlay.set(self.overlay_map.get(Config.get("prism_overlay", "none"), "🚫 无 (None)")) 
        self.cb_overlay.bind("<<ComboboxSelected>>", self.on_overlay_change)
        self.cb_overlay.pack(fill="x", pady=5)

        # --- 参数控制区 (动态显示/隐藏) ---
        self.f_params = ttk.Frame(f_algo_r)
        self.f_params.pack(fill="x", pady=5)

        # A. 自定义图片选择
        self.f_custom_file = ttk.Frame(self.f_params)
        ttk.Label(self.f_custom_file, text="图片:").pack(side="left")
        ttk.Entry(self.f_custom_file, textvariable=self.custom_tex_path, width=8).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(self.f_custom_file, text="📂", width=3, command=self.sel_custom_tex).pack(side="left", padx=(0,5))
        
        # 【新增】Checkbutton
        ttk.Checkbutton(self.f_custom_file, text="去黑底", variable=self.var_remove_black).pack(side="left")
        
        # B. 大小滑块
        self.f_scale = ttk.Frame(self.f_params)
        ttk.Label(self.f_scale, text="大小:").pack(side="left")
        s_scale = ttk.Scale(self.f_scale, from_=0.1, to=3.0, variable=self.var_scale, orient="horizontal")
        s_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        # C. 密度/数量滑块
        self.f_density = ttk.Frame(self.f_params)
        ttk.Label(self.f_density, text="数量:").pack(side="left")
        s_density = ttk.Scale(self.f_density, from_=1, to=100, variable=self.var_density, orient="horizontal")
        s_density.pack(side="left", fill="x", expand=True, padx=5)

        # 颜色配置
        self.f_config = ttk.Frame(self)
        self.f_config.pack(fill="x", padx=10, pady=5)
        
        # 【修复点】将 f_cosine_ctrl 的代码移动到这里 (f_config 创建之后)
        self.f_cosine_ctrl = ttk.LabelFrame(self.f_config, text=" 🌊 幻彩参数设置 ", padding=10)
        
        # 行1: 模式与预设
        f_row1 = ttk.Frame(self.f_cosine_ctrl)
        f_row1.pack(fill="x", pady=2)
        
        ttk.Label(f_row1, text="扫描方式:").pack(side="left")
        self.scan_map = {
            "horizontal": "水平",
            "vertical":   "垂直",
            "diagonal":   "对角",
            "radial":     "径向",
            "angular":    "角度"
        }
        self.scan_mode_var = tk.StringVar(value="对角")
        cb_scan = ttk.Combobox(f_row1, textvariable=self.scan_mode_var, width=15, state="readonly", 
                               values=list(self.scan_map.values()))
        cb_scan.pack(side="left", padx=5)
        
        ttk.Label(f_row1, text="配色方案:").pack(side="left", padx=(10, 0))
        self.palette_var = tk.StringVar(value="Rainbow (彩虹)")
        preset_values = list(CosineGradientGenerator.PRESETS.keys()) + ["Custom (自定义)"]
        
        self.cb_pal = ttk.Combobox(f_row1, textvariable=self.palette_var, width=15, state="readonly",
                              values=preset_values)
        self.cb_pal.pack(side="left", padx=5)
        
        # 【修改点 B】: 绑定事件，选择“自定义”时显示编辑器
        self.cb_pal.bind("<<ComboboxSelected>>", self.on_cosine_preset_select)
        
        
        # =================【修改开始：方案一 UI 实现】=================
        # 行2: 重复次数与相位 (Material Tiling UI)
        f_row2 = ttk.Frame(self.f_cosine_ctrl)
        f_row2.pack(fill="x", pady=5)
        
        # --- A. 重复次数 (Repeats / Tiling) ---
        # 逻辑：控制彩虹在画面中循环出现的次数 (1.0x = 出现1次, 2.0x = 出现2次)
        ttk.Label(f_row2, text="重复次数:").pack(side="left")
        self.freq_var = tk.DoubleVar(value=1.0)
        
        # 数值显示 Label (例如: 1.0x)
        self.lbl_freq_val = ttk.Label(f_row2, text="1.0x", width=4, anchor="center", foreground="blue")
        
        # 滑块范围建议 0.1x 到 5.0x (太大就太密了看不清)
        s_freq = ttk.Scale(f_row2, from_=0.1, to=5.0, variable=self.freq_var, orient="horizontal")
        s_freq.pack(side="left", fill="x", expand=True, padx=5)
        
        # 绑定事件：拖动时实时更新文字
        s_freq.configure(command=lambda v: self.lbl_freq_val.config(text=f"{float(v):.1f}x"))
        
        # 将数字放在滑块右边，符合阅读习惯
        self.lbl_freq_val.pack(side="left")
        
        # --- B. 相位偏移 (Phase Shift / Offset) ---
        # 逻辑：控制彩虹的起始位置 (0% - 100%)，解决接缝或对齐视觉重心
        ttk.Label(f_row2, text="相位偏移:").pack(side="left", padx=(15,0)) # 增加一点左间距
        self.offset_var = tk.DoubleVar(value=0.0)
        
        # 数值显示 Label (例如: 50%)
        self.lbl_off_val = ttk.Label(f_row2, text="0%", width=4, anchor="center", foreground="blue")
        
        s_off = ttk.Scale(f_row2, from_=0.0, to=1.0, variable=self.offset_var, orient="horizontal")
        s_off.pack(side="left", fill="x", expand=True, padx=5)
        
        # 绑定事件：拖动时显示百分比
        s_off.configure(command=lambda v: self.lbl_off_val.config(text=f"{int(float(v)*100)}%"))
        
        self.lbl_off_val.pack(side="left")
        
        # 继续后续控件
        self.editor = GradientEditor(self.f_config)
        self.editor.pack(fill="x")
        # 绑定预设改变事件以保存配置
        self.editor.combo_preset.bind("<<ComboboxSelected>>", self.on_preset_change)
        self.hue_picker = SingleColorPicker(self.f_config)
        self.star_color_picker = SingleColorPicker(self.f_config, title=" ✨ 特效颜色 ")
        # ================= 【增加开始】 =================
        # 2. 新增：故障强度控制面板 (默认隐藏)
        self.f_glitch_ctrl = ttk.LabelFrame(self.f_config, text=" 📺 故障参数 ", padding=10)
        
        f_g_inner = ttk.Frame(self.f_glitch_ctrl)
        f_g_inner.pack(fill="x")
        
        ttk.Label(f_g_inner, text="错位强度 (Intensity):").pack(side="left")
        self.var_glitch_power = tk.IntVar(value=20) # 默认值
        # 范围 0-100，对应 ImageProcessor 里的逻辑
        self.scale_glitch = ttk.Scale(f_g_inner, from_=0, to=100, variable=self.var_glitch_power, orient="horizontal")
        self.scale_glitch.pack(side="left", fill="x", expand=True, padx=10)
        
        lbl_glitch_val = ttk.Label(f_g_inner, text="20", width=4) 
        lbl_glitch_val.pack(side="left")
        # 绑定滑块拖动显示数值
        self.scale_glitch.configure(command=lambda v: lbl_glitch_val.config(text=f"{int(float(v))}"))
        # 4. 运行按钮
        f_run = ttk.Frame(self, padding=10)
        f_run.pack(fill="both", expand=True)
        f_btns = ttk.Frame(f_run)
        f_btns.pack(fill="x", pady=5)
        
        self.btn_run = ttk.Button(f_btns, text="🚀 开始处理", command=self.start)
        self.btn_run.pack(side="left", fill="x", expand=True)
        self.btn_try = ttk.Button(f_btns, text="👁 试看一帧", command=self.try_process_one_frame)
        self.btn_try.pack(side="left", padx=5)
        self.btn_preview = ttk.Button(f_btns, text="🔍 对比预览", command=self.open_preview)
        self.btn_preview.pack(side="left", padx=5)

        self.pbar = ttk.Progressbar(f_run)
        self.pbar.pack(fill="x")
        self.log_box = tk.Text(f_run, height=8, state="disabled", bg="#f0f0f0", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True, pady=5)
    
    def get_real_scan_mode(self):
            """将UI显示的中文名称转换回代码所需的英文名称"""
            val = self.scan_mode_var.get()
            for k, v in self.scan_map.items():
                if v == val: return k
                if k == val: return k # 防止配置文件存的是旧英文
            return "对角" # 默认兜底    
            
    def on_cosine_preset_select(self, event):
            """当幻彩模式的下拉框改变时触发"""
            val = self.palette_var.get()
            
            # 如果选了自定义，显示颜色编辑器；否则隐藏
            if val == "Custom (自定义)":
                self.editor.pack(fill="x", after=self.f_cosine_ctrl) # 确保显示在参数面板下方
                self.editor.config(text=" 🎨 自定义渐变色 (分段插值) ")
            else:
                self.editor.pack_forget()
    # =================================================================
    # 【新增功能】 试看一帧 (内存处理，不生成文件)
    # =================================================================
    def try_process_one_frame(self):
        # 1. 检查路径
        raw_path = self.input_path.get()
        if not raw_path or not os.path.exists(raw_path):
            messagebox.showwarning("提示", "请先选择有效的源路径 (NPK文件或文件夹)")
            return

        # 2. 收集当前面板的所有参数 (与 run 方法一致)
        mode = self.algo_mode.get()
        overlay = self.get_real_overlay_mode()
        if self.algo_mode.get() == "glitch":
            density = self.var_glitch_power.get()
        else:
            density = self.var_density.get()
        scale = self.var_scale.get()
        eff_rgb = self.star_color_picker.target_rgb
        algo_rgb = self.hue_picker.target_rgb
        lut = None
        target_hue = 0
        
        # 准备调色数据
        if mode == "target_hue":
            target_hue = self.hue_picker.get_hue_value()
        elif mode != "none":
             # 【修改】同样删除了 "and mode != 'line_art'"
             current_preset = self.editor.combo_preset.get()
             if "透明" in current_preset or "Hidden" in current_preset:
                 mode = "transparent"
             else:
                 lut = self.editor.get_lut()
        
        # 准备自定义贴图
        custom_tex = None
        if "custom" in overlay:
            if self.cached_custom_img:
                custom_tex = self.cached_custom_img
            else:
                # 尝试现场加载一下，防止用户没点过 Start 还没缓存
                p = self.custom_tex_path.get()
                if p and os.path.exists(p):
                    try:
                        custom_tex = Image.open(p).convert("RGBA")
                        # 简单的去黑底逻辑复用
                        if self.var_remove_black.get():
                            dt = np.array(custom_tex)
                            mask = (dt[:,:,0]<30) & (dt[:,:,1]<30) & (dt[:,:,2]<30)
                            dt[:,:,3][mask] = 0
                            custom_tex = Image.fromarray(dt)
                    except: pass
        
        # 3. 锁定界面并启动后台线程
        self.btn_try.config(state="disabled", text="处理中...")
        threading.Thread(target=self._worker_try_one, 
                         args=(raw_path, mode, lut, target_hue, overlay, custom_tex, density, scale, eff_rgb, algo_rgb)).start()

    def _worker_try_one(self, raw_path, mode, lut, target_hue, overlay, custom_tex, density, scale, eff_rgb, algo_rgb):
        try:
            target_npk_path = raw_path
            # 如果选的是文件夹，找里面的第一个 NPK
            if os.path.isdir(raw_path):
                files = glob.glob(os.path.join(raw_path, "*.npk"))
                if not files:
                    raise Exception("文件夹内没有找到 .npk 文件")
                target_npk_path = files[0]

            found_original = None
            found_processed = None
            found_info = ""

            # 4. 内存读取 NPK -> IMG -> Frame
            with open(target_npk_path, 'rb') as f:
                npk = NPK.open(f)
                npk.load_all() # 读取目录
                
                # 寻找第一个包含有效图片的 IMG
                for inner_file in npk.files:
                    if not inner_file.name.lower().endswith(".img"): continue
                    
                    try:
                        img_obj = IMGFactory.open(BytesIO(inner_file.data))
                        # 遍历帧，找一个有内容的
                        for i, frame in enumerate(img_obj.images):
                            # 处理 ImageLink 引用 (核心稳定性逻辑)
                            current_frame_data = frame
                            if isinstance(frame, ImageLink):
                                target_idx = -1
                                if hasattr(frame, '_image'): target_idx = frame._image
                                elif hasattr(frame, 'link'): target_idx = frame.link
                                elif hasattr(frame, 'target'): target_idx = frame.target
                                if isinstance(target_idx, int) and 0 <= target_idx < len(img_obj.images):
                                    current_frame_data = img_obj.images[target_idx]
                            
                            # 构建 PIL 图片
                            pil_img = img_obj.build(current_frame_data)
                            
                            # 过滤太小的图或全透明图 (比如只有 1x1 的占位符)
                            if pil_img.width > 20 and pil_img.height > 20:
                                # 检查是否全透明
                                extrema = pil_img.getextrema()
                                if extrema and len(extrema) > 3:
                                    alpha_min, alpha_max = extrema[3]
                                    if alpha_max == 0: continue # 全透明，跳过
                                
                                # === 找到有效图，开始处理 ===
                                found_original = pil_img.copy()
                                found_info = f"{os.path.basename(target_npk_path)}\n{inner_file.name} [帧 {i}]"
                                
                                cos_params = {}
                                if self.algo_mode.get() == "cosine_grad":
                                    # 【新增】收集自定义颜色
                                    custom_colors_data = []
                                    if self.palette_var.get() == "Custom (自定义)":
                                        # self.editor.color_list 存的是 HEX 字符串，需转 RGB
                                        for c_hex in self.editor.color_list:
                                            try:
                                                custom_colors_data.append(hex_to_rgb(c_hex))
                                            except: pass
                                    
                                    cos_params = {
                                        'scan': self.get_real_scan_mode(),
                                        'preset': self.palette_var.get(),
                                        'freq': self.freq_var.get(),
                                        'offset': self.offset_var.get(),
                                        'custom_colors': custom_colors_data # 【新增】
                                    }
                                # 调用核心处理器 (直接复用)
                                found_processed = ImageProcessor.process(
                                    pil_img.copy(),
                                    mode, lut, target_hue,
                                    overlay_type=overlay,
                                    overlay_texture=custom_tex,
                                    overlay_density=density,
                                    overlay_scale=scale,
                                    overlay_rgb=eff_rgb,
                                    target_rgb=algo_rgb,
                                    cosine_params=cos_params
                                )
                                break # 跳出帧循环
                        
                        if found_original: break # 跳出文件循环
                    except: continue

            # 5. 回到主线程显示弹窗
            if found_original and found_processed:
                self.after(0, lambda: self._show_try_popup(found_original, found_processed, found_info))
            else:
                self.after(0, lambda: messagebox.showwarning("失败", "未能在 NPK 中找到有效的图片帧。"))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("错误", f"试看失败: {e}"))
        finally:
            self.after(0, lambda: self.btn_try.config(state="normal", text="👁 试看一帧"))

    def _show_try_popup(self, img_src, img_dst, info_text):
        top = tk.Toplevel(self)
        top.title("效果试看")
        top.geometry("900x600")
        top.config(bg="#222") # 深色背景方便看特效
        
        # 顶部信息
        ttk.Label(top, text=f"来源: {info_text}", background="#222", foreground="#aaa", font=("Consolas", 9)).pack(pady=5)
        
        # 图片容器
        f_imgs = tk.Frame(top, bg="#222")
        f_imgs.pack(expand=True, fill="both", padx=10, pady=10)
        
        # 智能缩放 (如果图片太小，放大一点看细节)
        preview_h = 500
        scale_ratio = 1.0
        if img_src.height < 300: scale_ratio = 2.0 # 小图放大2倍
        if img_src.height > 800: scale_ratio = 500 / img_src.height # 大图缩小
        
        def process_for_view(pil_img):
            w = int(pil_img.width * scale_ratio)
            h = int(pil_img.height * scale_ratio)
            return ImageTk.PhotoImage(pil_img.resize((w, h), Image.Resampling.NEAREST))
            
        tk_src = process_for_view(img_src)
        tk_dst = process_for_view(img_dst)
        
        # 左侧：原图
        f_l = tk.LabelFrame(f_imgs, text=" 修改前 (Original) ", bg="#222", fg="white")
        f_l.pack(side="left", expand=True, fill="both", padx=5)
        lbl_l = tk.Label(f_l, image=tk_src, bg="#333") # 深灰底色
        lbl_l.image = tk_src
        lbl_l.pack(expand=True)
        
        # 中间箭头
        tk.Label(f_imgs, text="▶", bg="#222", fg="#555", font=("Arial", 20)).pack(side="left", padx=5)

        # 右侧：效果图
        f_r = tk.LabelFrame(f_imgs, text=" 修改后 (Preview) ", bg="#222", fg="#00ff00") # 绿色标题
        f_r.pack(side="left", expand=True, fill="both", padx=5)
        lbl_r = tk.Label(f_r, image=tk_dst, bg="#333")
        lbl_r.image = tk_dst
        lbl_r.pack(expand=True) 
    def on_mode_change(self):
        """调色模式切换逻辑"""
        mode = self.algo_mode.get()
        Config.set("prism_mode", mode)
        
        # 隐藏所有颜色配置
        self.editor.pack_forget()
        self.hue_picker.pack_forget()
        self.f_glitch_ctrl.pack_forget()# 【新增】故障滑块
        
        if hasattr(self, 'f_cosine_ctrl'): # 防止未初始化报错
            self.f_cosine_ctrl.pack_forget() # 程序化渐变参数
        if mode == "none":
            pass # 不调色
        elif mode == "glitch":
            # 【增加】显示故障滑块
            self.f_glitch_ctrl.pack(fill="x")
        elif mode == "target_hue":
            # 只有统一色相模式才用单色选择器
            self.hue_picker.pack(fill="x")
            self.hue_picker.config(text=" 🌈 目标颜色选择 ")
        elif mode == "cosine_grad":  # <--- 【修改】幻彩模式逻辑
            self.f_cosine_ctrl.pack(fill="x")
            # 只有当幻彩预设选了“自定义”时，才把下面的颜色编辑器显示出来
            if self.palette_var.get() == "Custom (自定义)":
                self.editor.pack(fill="x")
                self.editor.config(text=" 🎨 自定义渐变色 (分段插值) ")
        else:
            self.editor.pack(fill="x")
            self.editor.update_preview()
            
        # 触发一下特效检查，确保界面布局正确
        self.on_overlay_change()
		
    def get_real_overlay_mode(self):
        # 将显示名称转回代码
        val = self.cb_overlay.get()
        for k, v in self.overlay_map.items():
            if v == val: return k
            if k == val: return k # 如果还没选，可能是默认值
        return "none"
    
    def on_overlay_change(self, event=None):
        mode = self.get_real_overlay_mode()
        Config.set("prism_overlay", mode)

        # 隐藏所有参数控件
        self.f_custom_file.pack_forget()
        self.f_scale.pack_forget()
        self.f_density.pack_forget()
        self.star_color_picker.pack_forget()
      
        # 根据模式显示控件
        if "custom" in mode:
            # 自定义模式：显示文件、大小、密度
            self.f_custom_file.pack(fill="x", pady=2)
            self.f_scale.pack(fill="x", pady=2)
            self.f_density.pack(fill="x", pady=2)
        elif mode in ["lightning", "outer_glow", "starlight", "sakura"]:
            # 内置模式：显示大小(粗细)、密度(数量)、颜色
            self.f_scale.pack(fill="x", pady=2)
            if mode != "outer_glow":
                self.f_density.pack(fill="x", pady=2) 
            
            self.star_color_picker.pack(fill="x", pady=2)
            titles = {
                "lightning": " ⚡ 闪电颜色 ",
                "outer_glow": " 🌟 描边颜色 ",
                "starlight": " ✨ 星光颜色 ",
                "sakura": " 🌸 花瓣颜色 "
            }
            self.star_color_picker.config(text=titles.get(mode, " ✨ 特效颜色 "))

    def sel_custom_tex(self):
        p = filedialog.askopenfilename(filetypes=[("Image", "*.png;*.jpg;*.bmp")])
        if p: self.custom_tex_path.set(p)

    def on_preset_change(self, event):
        val = self.editor.combo_preset.get()
        self.editor.load_preset(val)
        Config.set("prism_preset", val) # 保存预设

    def sel_file(self):
        p = filedialog.askopenfilename(filetypes=[("NPK", "*.npk")])
        if p: self._set_p(p)
    def sel_dir(self):
        p = filedialog.askdirectory()
        if p: self._set_p(p)
    def sel_out_dir(self):
        p = filedialog.askdirectory()
        if p: self.output_dir.set(p); Config.set("prism_output", p)
    def _set_p(self, p):
        self.input_path.set(p); Config.set("prism_input", p)
        base = os.path.dirname(p) if os.path.isfile(p) else p
        if not self.output_dir.get():
            default_out = os.path.join(base, "Output_MOD")
            self.output_dir.set(default_out); Config.set("prism_output", default_out)
    def log(self, s):
        self.log_box.config(state="normal")
        self.log_box.insert("end", s+"\n"); self.log_box.see("end")
        self.log_box.config(state="disabled")

    def open_preview(self):
        raw = self.input_path.get(); out_dir = self.output_dir.get()
        if not raw or not os.path.exists(raw): messagebox.showwarning("提示", "请先选择有效的源路径"); return
        file_pairs = []
        if os.path.isfile(raw):
            fname = os.path.basename(raw)
            if fname.lower().endswith(".npk"): file_pairs.append((raw, os.path.join(out_dir, "%" + fname)))
        else:
            for fname in os.listdir(raw):
                if fname.lower().endswith(".npk") and not fname.startswith("%"):
                    file_pairs.append((os.path.join(raw, fname), os.path.join(out_dir, "%" + fname)))
        if not file_pairs:
            messagebox.showinfo("提示", "没有找到 NPK 文件")
            return
        CompareViewer(self.winfo_toplevel(), file_pairs)

    def start(self):
        # 1. 基础检查
        raw = self.input_path.get()
        out = self.output_dir.get()
        patcher = get_resource_path("NpkPatcher.exe")
        
        if not os.path.exists(raw): return messagebox.showerror("错误", "源路径无效")
        if not os.path.exists(patcher): return messagebox.showerror("错误", "缺少 NpkPatcher.exe")
        
        # 2. 收集参数 (必须在主线程完成，转换为简单数据类型)
        # 这里的变量将打包传给子进程
        params = {}
        params['patcher'] = os.path.abspath(patcher)
        params['raw'] = raw
        params['out'] = out
        
        params['mode'] = self.algo_mode.get()
        params['overlay'] = self.get_real_overlay_mode()
        
        # Glitch 还是 普通密度
        if params['mode'] == "glitch":
             params['density'] = self.var_glitch_power.get()
        else:
             params['density'] = self.var_density.get()
             
        params['scale'] = self.var_scale.get()
        params['eff_rgb'] = self.star_color_picker.target_rgb
        params['algo_rgb'] = self.hue_picker.target_rgb
        params['remove_black'] = self.var_remove_black.get()
        
        # 贴图路径 (只传路径，不传对象)
        params['tex_path'] = self.custom_tex_path.get() if "custom" in params['overlay'] else None
        
        # LUT 数据 (numpy array)
        params['lut'] = None
        params['target_hue'] = 0
        
        if params['mode'] == "target_hue":
            params['target_hue'] = self.hue_picker.get_hue_value()
        elif params['mode'] != "none" and params['mode'] != "glitch" and params['mode'] != "cosine_grad":
            # 检查预设名
            preset = self.editor.combo_preset.get()
            if "透明" in preset or "Hidden" in preset:
                params['mode'] = "transparent"
            else:
                params['lut'] = self.editor.get_lut()
        
        # Cosine 参数
        params['cos'] = {}
        if params['mode'] == "cosine_grad":
            custom_colors_data = []
            if self.palette_var.get() == "Custom (自定义)":
                for c_hex in self.editor.color_list:
                    try: custom_colors_data.append(hex_to_rgb(c_hex))
                    except: pass
            
            params['cos'] = {
                'scan': self.get_real_scan_mode(),
                'preset': self.palette_var.get(),
                'freq': self.freq_var.get(),
                'offset': self.offset_var.get(),
                'custom_colors': custom_colors_data
            }

        # 3. 启动后台线程 (Thread -> ProcessPool)
        self.btn_run.config(state="disabled", text="正在初始化进程...")
        self.log_box.config(state="normal"); self.log_box.delete(1.0, "end"); self.log_box.config(state="disabled")
        
        threading.Thread(target=self.run_multiprocess, args=(params,)).start()

    def run_multiprocess(self, p):
        """后台线程：负责调度多进程"""
        try:
            raw = p['raw']
            out = p['out']
            if not os.path.exists(out): os.makedirs(out)
            
            # 1. 扫描文件
            file_list = []
            if os.path.isfile(raw):
                if raw.lower().endswith(".npk"): file_list = [raw]
            elif os.path.isdir(raw):
                file_list = [os.path.join(raw, f) for f in os.listdir(raw) if f.lower().endswith(".npk")]
            
            total = len(file_list)
            if total == 0:
                self.log("⚠️ 未找到 NPK 文件")
                self.after(0, lambda: self.btn_run.config(state="normal", text="🚀 开始处理"))
                return

            self.log(f"🚀 启动多进程处理 (CPU核心数: {os.cpu_count()})")
            self.log(f"📋 待处理文件数: {total}")
            
            # 2. 组装任务包
            # task_data 格式对应 process_one_npk_task 的解包顺序
            tasks = []
            for fpath in file_list:
                task = (
                    fpath, out, p['patcher'],
                    p['mode'], p['lut'], p['target_hue'],
                    p['overlay'], p['tex_path'], p['density'], p['scale'], p['eff_rgb'], p['algo_rgb'],
                    p['cos'], p['remove_black']
                )
                tasks.append(task)
            
            # 3. 创建进程池
            # max_workers 建议设置为 CPU核数 - 1，留一个核给 UI 和系统
            workers = max(1, os.cpu_count() - 1)
            
            completed_count = 0
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                # 提交所有任务
                futures = {executor.submit(process_one_npk_task, t): t[0] for t in tasks}
                
                # 监听结果
                for future in concurrent.futures.as_completed(futures):
                    filename = os.path.basename(futures[future])
                    try:
                        success, msg = future.result()
                        if success:
                            self.log(f"✅ {msg}")
                        else:
                            self.log(f"❌ {msg}")
                    except Exception as e:
                        self.log(f"❌ {filename} 异常: {e}")
                    
                    completed_count += 1
                    # 更新进度条
                    progress = (completed_count / total) * 100
                    self.after(0, lambda v=progress: self.pbar.config(value=v))

            self.log("-" * 30)
            self.log("🎉 所有任务处理完毕！")
            self.after(0, self.on_processing_done)

        except Exception as e:
            self.log(f"❌ 严重错误: {e}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: self.btn_run.config(state="normal", text="🚀 开始处理"))

    def on_processing_done(self):
        self.pbar.config(value=100)
        self.btn_run.config(state="normal", text="🚀 开始处理")
        if messagebox.askyesno("完成", "处理结束！\n是否立即对比预览效果？"):
            self.open_preview()