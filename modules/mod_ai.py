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
from common import Config, check_path_safety, get_resource_path,is_ascii_path

HAS_AI_DEPS = True 

HAS_PYDNFEX = False
try:
    from pydnfex.npk import NPK
    from pydnfex.img import IMGFactory, ImageLink
    HAS_PYDNFEX = True
except ImportError:
    print("⚠️ 警告: 未检测到库，【调色盘】功能将不可用。")
    
# =========================================================================
# PART 4.6: 【AI 鹰眼模式】深度学习特征搜索 (MobileNetV3 + Faiss + SQLite)
# =========================================================================
# try:
    # import onnxruntime as ort
    # import faiss
    # import sqlite3
    # import numpy as np
    # import concurrent.futures
    # from PIL import ImageGrab
    # import glob 
    # HAS_AI_DEPS = True
# except ImportError:
    # import traceback
    # traceback.print_exc()
    # HAS_AI_DEPS = False
import sqlite3
class AIImagePreprocessor:
    @staticmethod
    def crop_and_resize(pil_img):
        try:
            if pil_img.mode != 'RGBA': pil_img = pil_img.convert('RGBA')
            bbox = pil_img.getbbox()
            if not bbox: return None
            cropped = pil_img.crop(bbox)
            if cropped.width < 1 or cropped.height < 1: return None
            target_size = 224
            w, h = cropped.size
            ratio = min(target_size / w, target_size / h)
            new_w = max(1, int(w * ratio))
            new_h = max(1, int(h * ratio))
            return cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        except: return None
    @staticmethod
    def preprocess_for_onnx(pil_img):
        """将 PIL 图片转换为 ONNX Runtime 需要的标准格式 (Numpy)"""
        try:
            # 1. 调整大小到 224x224 (MobileNet 标准输入)
            img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # 2. 转为 Numpy 数组 (H, W, C)
            img_np = np.array(img).astype(np.float32)
            
            # 3. 归一化 (模拟 PyTorch 的 transforms.Normalize)
            # 像素值从 0-255 变到 0-1
            img_np /= 255.0
            
            # 减均值 (Mean) 除以 标准差 (Std)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_np = (img_np - mean) / std
            
            # 4. 调整维度顺序: (H, W, C) -> (C, H, W)
            img_np = img_np.transpose((2, 0, 1))
            
            # 5. 增加 Batch 维度: (1, C, H, W)
            img_np = np.expand_dims(img_np, axis=0)
            
            return img_np.astype(np.float32)
        except Exception as e:
            print(f"预处理失败: {e}")
            return None
    @staticmethod
    def add_background(rgba_img):
        try:
            target_size = 224
            bg = Image.new("RGBA", (target_size, target_size), (128, 128, 128, 255))
            w, h = rgba_img.size
            paste_x = (target_size - w) // 2
            paste_y = (target_size - h) // 2
            bg.alpha_composite(rgba_img, (paste_x, paste_y))
            return bg.convert('RGB')
        except: return None

    @staticmethod
    def extract_color_histogram(rgba_img):
        try:
            arr = np.array(rgba_img)
            mask = arr[:, :, 3] > 30
            valid = arr[mask]
            if len(valid) == 0: return np.zeros(512, dtype=np.float32)
            rgb = valid[:, :3]
            quantized = rgb // 32 
            bin_indices = quantized[:, 0] * 64 + quantized[:, 1] * 8 + quantized[:, 2]
            hist = np.bincount(bin_indices, minlength=512)
            hist = hist / (hist.sum() + 1e-6)
            return hist.astype(np.float32) * 3.0 
        except: return np.zeros(512, dtype=np.float32)

class FeatureExtractor:
    _instance = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance

    def __init__(self):
        # 1. 确定模型路径
        import onnxruntime as ort 
        
        model_name = "mobilenet_v3_small.onnx" # 【注意】改成你的 onnx 文件名
        if hasattr(sys, '_MEIPASS'):
            model_path = os.path.join(sys._MEIPASS, model_name)
        else:
            model_path = model_name
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"缺失模型文件: {model_path}")

        # 2. 初始化 ONNX Runtime
        # providers=['CPUExecutionProvider'] 强制使用 CPU，兼容性最好
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 获取输入输出节点的名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def extract(self, pil_img):
        # 1. 图片预处理
        import faiss 
        rgba = AIImagePreprocessor.crop_and_resize(pil_img)
        if rgba is None: return None
        
        # 提取颜色直方图 (保持不变)
        color_vec = AIImagePreprocessor.extract_color_histogram(rgba)
        
        # 2. 准备 CNN 输入
        cnn_img = AIImagePreprocessor.add_background(rgba)
        
        # 【核心修改】调用新的 numpy 预处理
        input_data = AIImagePreprocessor.preprocess_for_onnx(cnn_img)
        if input_data is None: return None

        # 3. ONNX 推理
        try:
            # run 返回的是一个列表，取第一个元素即为特征向量
            cnn_vec = self.session.run([self.output_name], {self.input_name: input_data})[0]
            cnn_vec = cnn_vec.flatten() # 展平为一维数组
            
            # 4. 合并特征 + 归一化 (保持不变)
            combined = np.concatenate([cnn_vec, color_vec])
            faiss.normalize_L2(combined.reshape(1, -1))
            return combined.reshape(1, -1)
        except Exception as e:
            print(f"ONNX 推理错误: {e}")
            return None

# --- 数据库 (支持分片搜索) ---
class VectorDB:
    SQL_DB = "dnf_ai_meta.db"
    INDEX_PREFIX = "dnf_ai" # 匹配 dnf_ai_0.index
    
    def __init__(self):
        self.conn = None
        self.dim = 1088 
        self.db_dir = os.getcwd() # 【新增】默认为当前目录

    # 【修改】支持传入路径
    def init_storage(self, mode='read', db_dir=None):
        if db_dir and os.path.exists(db_dir):
            self.db_dir = db_dir
            
        db_path = os.path.join(self.db_dir, self.SQL_DB)
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        
    def get_count(self):
        # 统计所有分片总数
        total = 0
        try:
            # 拼接路径
            pattern = os.path.join(self.db_dir, f"{self.INDEX_PREFIX}_*.index")
            files = glob.glob(pattern)
            cursor = self.conn.execute("SELECT COUNT(*) FROM meta")
            total = cursor.fetchone()[0]
        except: pass
        return total

    def query(self, vector, k=500):
        import faiss
        # 拼接路径
        pattern = os.path.join(self.db_dir, f"{self.INDEX_PREFIX}_*.index")
        index_files = glob.glob(pattern)
        index_files.sort()
        
        all_candidates = []
        vec_arr = vector.astype('float32')
        
        # 2. 串行搜索：搜完一个扔一个
        for fpath in index_files:
            try:
                # 解析 shard_id
                shard_id = int(os.path.splitext(os.path.basename(fpath))[0].split('_')[-1])
                idx = faiss.read_index(fpath)
                
                # 搜索
                D, I = idx.search(vec_arr, k)
                
                # 收集结果
                ids = I[0]
                scores = D[0]
                for local_id, score in zip(ids, scores):
                    if local_id == -1: continue
                    all_candidates.append({
                        "shard_id": shard_id,
                        "local_id": int(local_id),
                        "score": float(score)
                    })
                
                # 释放内存
                del idx
            except Exception as e:
                print(f"Index read error {fpath}: {e}")
                
        # 3. 汇总排序，取前 k 个
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_k = all_candidates[:k]
        
        # 4. 查数据库获取详情
        final_results = []
        for item in top_k:
            cursor = self.conn.execute(
                "SELECT npk, img, frame FROM meta WHERE shard_id=? AND local_id=?", 
                (item['shard_id'], item['local_id'])
            )
            row = cursor.fetchone()
            if row:
                # 【关键】这里只返回文件名(row[0]虽是全路径，但我们取basename)，路径拼接留给UI层处理
                npk_basename = os.path.basename(row[0])
                final_results.append({
                    "score": item['score'],
                    "npk": npk_basename,
                    "img": row[1],
                    "frame": row[2],
                    "orig_path": row[0] # 保留原路径备查
                })
                
        return final_results

# --- 3. 索引构建器 (爬虫) ---
class AIIndexer:
    def build(self, folder_path, progress_cb, done_cb):
        threading.Thread(target=self._worker, args=(folder_path, progress_cb, done_cb)).start()

    def _worker(self, folder_path, progress_cb, done_cb):
        try:
            import faiss
            import onnxruntime
        except ImportError:
             done_cb("❌ 缺少依赖库")
             return
        if not HAS_AI_DEPS:
            done_cb("❌ 缺少 pytorch/faiss 库，无法运行。")
            return

        db = VectorDB()
        db.init_storage(mode='write')
        extractor = FeatureExtractor.get_instance()
        
        all_npks = glob.glob(os.path.join(folder_path, "*.npk"))
        total_npks = len(all_npks)
        
        global_id_counter = 0
        total_indexed = 0
        
        # 缓存队列
        meta_buffer = []
        vec_buffer = []
        
        for i, npk_path in enumerate(all_npks):
            # 更新进度
            progress_cb(i, total_npks, f"正在分析: {os.path.basename(npk_path)}")
            
            try:
                with open(npk_path, 'rb') as f:
                    npk = NPK.open(f)
                    npk.load_all()
                    
                    for inner in npk.files:
                        if not inner.name.lower().endswith('.img'): continue
                        try:
                            img_obj = IMGFactory.open(BytesIO(inner.data))
                            for frame_idx, frame in enumerate(img_obj.images):
                                try:
                                    # 1. 获取图片
                                    pil_img = img_obj.build(frame)
                                    
                                    # 过滤太小的图 (图标/碎片)，这些通常不需要搜，且干扰特征
                                    if pil_img.width < 32 or pil_img.height < 32: continue
                                    if not pil_img.getbbox(): continue # 空白图
                                    
                                    # 2. 提取特征 (核心耗时步)
                                    vec = extractor.extract(pil_img)
                                    
                                    if vec is not None:
                                        meta_buffer.append((npk_path, inner.name, frame_idx))
                                        vec_buffer.append(vec)
                                        
                                        # 批处理写入，防止内存爆炸
                                        if len(vec_buffer) >= 500:
                                            db.add_batch(global_id_counter, meta_buffer, vec_buffer)
                                            global_id_counter += len(vec_buffer)
                                            total_indexed += len(vec_buffer)
                                            meta_buffer = []
                                            vec_buffer = []
                                except: continue
                        except: continue
            except Exception as e:
                print(f"Skipped {os.path.basename(npk_path)}: {e}")
                
        # 处理剩余缓存
        if vec_buffer:
            db.add_batch(global_id_counter, meta_buffer, vec_buffer)
            total_indexed += len(vec_buffer)
            
        # 保存索引
        db.save_index()
        done_cb(f"✅ 索引构建完成！\n共收录 {total_indexed} 个素材特征。\n索引文件已保存，下次直接搜索。")
# =========================================================================
# 【新增】全局图片缓存池 (防止切换窗口时重复加载)
# =========================================================================
class GlobalImageCache:
    _cache = {} # 静态字典，生命周期伴随整个程序

    @classmethod
    def get(cls, key):
        return cls._cache.get(key)

    @classmethod
    def set(cls, key, tk_image):
        cls._cache[key] = tk_image

    @classmethod
    def has(cls, key):
        return key in cls._cache
        
# =========================================================================
# 【新增】极速 NPK 读取 (增强兼容版)
# =========================================================================
def fast_load_frame_from_npk(npk_path, img_name_inner, frame_idx):
    try:
        # 1. 打开文件
        with open(npk_path, 'rb') as f:
            npk = NPK.open(f)
            
            # 2. 查找目标文件 (增强路径匹配)
            target_entry = None
            
            # 将搜索的目标名统一转为小写 + 正斜杠
            search_key = img_name_inner.replace('\\', '/').lower()
            
            for entry in npk.files:
                # 将 NPK 里的名字也统一转换
                entry_key = entry.name.replace('\\', '/').lower()
                
                # 模糊匹配：有时候 NPK 里是 "sprite/map/a.img"，搜索结果是 "map/a.img"
                if entry_key == search_key or entry_key.endswith(search_key) or search_key.endswith(entry_key):
                    target_entry = entry
                    break
            
            if not target_entry:
                print(f"[错误] NPK中找不到文件: {img_name_inner} (在 {os.path.basename(npk_path)})")
                return None

            # 3. 尝试获取偏移量和大小 (适配不同版本的 pydnfex)
            # 尝试顺序: offset/size -> pos/length -> start/size
            offset = getattr(target_entry, 'offset', getattr(target_entry, 'pos', getattr(target_entry, 'start', None)))
            size = getattr(target_entry, 'size', getattr(target_entry, 'length', getattr(target_entry, 'packed_size', None)))

            data = None
            
            # 4. 极速读取模式
            if offset is not None and size is not None:
                f.seek(offset)
                data = f.read(size)
            else:
                # 【兼容模式】如果找不到 offset 属性，说明库不支持直接访问属性
                # 我们尝试调用库的 read 方法 (如果存在)
                #print(f"[警告] 无法获取 offset/size，尝试兼容模式: {img_name_inner}")
                # 这是一个兜底策略，为了防止一直 Loading，如果这里也失败，那确实没办法了
                # 某些库版本可能需要 npk.read_file(target_entry)
                #if hasattr(npk, 'read_file'):
                    #data = npk.read_file(target_entry)
                if hasattr(target_entry, 'data') and target_entry.data:
                     # 有些库 open 的时候就读进去了? (不太可能，但检查一下)
                    data = target_entry.data
                else:
                    print(f"[严重] 无法读取数据，对象属性: {dir(target_entry)}")
                    return None

            if not data:
                print(f"[错误] 读取到的数据为空: {img_name_inner}")
                return None

            # 5. 解析 IMG
            img_obj = IMGFactory.open(BytesIO(data))
            
            # 6. 获取帧 (防止越界)
            total_frames = len(img_obj.images)
            if total_frames == 0: return None
            
            actual_frame_idx = frame_idx
            if actual_frame_idx >= total_frames:
                actual_frame_idx = 0 # 如果帧号超了，就显示第0帧，总比不显示好
            
            frame = img_obj.images[actual_frame_idx]
            
            # 处理引用帧 (Link)
            if isinstance(frame, ImageLink):
                target_idx = -1
                if hasattr(frame, '_image'): target_idx = frame._image
                elif hasattr(frame, 'link'): target_idx = frame.link
                elif hasattr(frame, 'target'): target_idx = frame.target
                
                if isinstance(target_idx, int) and 0 <= target_idx < total_frames:
                    frame = img_obj.images[target_idx]
                else:
                    # 引用失效，尝试找第一个非引用的
                    frame = img_obj.images[0]

            # 7. 生成 PIL 图片
            pil_img = img_obj.build(frame)
            return pil_img

    except Exception as e:
        print(f"[异常] 读图失败 {os.path.basename(npk_path)} -> {img_name_inner}: {e}")
        # import traceback
        # traceback.print_exc()
        return None
# --- 4. 搜索页面 (UI) ---
# --- 4. 搜索页面 (适配分片索引) ---
class GoogleSearchPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.indexer = AIIndexer() # 这里的 Indexer 还是旧的引用，但不影响搜图，只要不点构建就行
        self.db = VectorDB()
        self.index_lib_path = tk.StringVar(value=Config.get("ai_index_lib", os.getcwd()))
        self.repo_path = tk.StringVar(value=Config.get("ai_repo", ""))
        self.target_img_path = tk.StringVar(value="")
        self.pasted_image = None
        self.tk_preview = None
        self.is_ready = False
        
        # --- 【新增】K值与筛选变量 ---
        self.var_k = tk.IntVar(value=500)       # 搜索数量 (默认500)
        self.var_filter = tk.StringVar(value="") # 筛选关键字
        self.current_page = 0   # 当前页码 (从0开始计数)
        self.var_page_size = tk.IntVar(value=50)
        self.var_jump_page = tk.StringVar() 
        self.page_size = 50 
        self.preview_win = None     
        self.preview_tk_img = None  
        self.last_preview_id = None 
        
        if not HAS_AI_DEPS:
            tk.Label(self, text="❌ 缺少深度学习依赖库\n请运行: pip install torch torchvision faiss-cpu", fg="red", font=("微软雅黑", 12)).pack(pady=50)
            return

        self.create_ui()
        
        # 【核心修复】检查分片索引是否存在
        # 旧代码: checking VectorDB.FAISS_IDX (已删除)
        # 新代码: 检查第一个分片 dnf_ai_0.index 和数据库
        if os.path.exists(os.path.join(self.index_lib_path.get(), "dnf_ai_meta.db")):
             self.load_index_bg()

        # 记录当前独立窗口对象，用于判断状态
        self.detached_win = None
        self.bind_all("<Control-v>", self._handle_ctrl_v)

    def create_ui(self):
        tk.Label(self, text="AI 搜图", font=("微软雅黑", 16, "bold"), fg="#8e44ad").pack(pady=10)

        # 索引区
        f_idx = ttk.LabelFrame(self, text=" 1. 路径设置  请使用英文路径", padding=10)
        f_idx.pack(fill="x", padx=10)
        
        # 行1：索引数据位置
        f_r1 = ttk.Frame(f_idx); f_r1.pack(fill="x", pady=2)
        ttk.Label(f_r1, text="索引数据(db/index):", width=18).pack(side="left")
        ttk.Entry(f_r1, textvariable=self.index_lib_path).pack(side="left", fill="x", expand=True)
        ttk.Button(f_r1, text="📂 选择", command=self.sel_index_lib).pack(side="left", padx=5)
        # 加载按钮
        ttk.Button(f_r1, text="🔄 加载/刷新索引", command=self.load_index_bg).pack(side="left")

        # 行2：素材库位置
        f_r2 = ttk.Frame(f_idx); f_r2.pack(fill="x", pady=2)
        ttk.Label(f_r2, text="素材仓库(NPK源):", width=18).pack(side="left")
        ttk.Entry(f_r2, textvariable=self.repo_path).pack(side="left", fill="x", expand=True)
        ttk.Button(f_r2, text="📂 选择", command=self.sel_repo).pack(side="left", padx=5)

        self.lbl_status = ttk.Label(f_idx, text="索引未加载", foreground="gray")
        self.lbl_status.pack(anchor="w", pady=(5,0))


        # 搜索区
        f_sch = ttk.LabelFrame(self, text=" 2. 图像识别 ", padding=10)
        f_sch.pack(fill="x", padx=10, pady=10) # 注意：这里去掉了 expand=True, fill="both" 改为 "x"，不再占用底部空间
        f_s = ttk.Frame(f_sch); f_s.pack(fill="x")
        ttk.Entry(f_s, textvariable=self.target_img_path).pack(side="left", fill="x", expand=True)
        ttk.Button(f_s, text="📂 选图", command=self.sel_target).pack(side="left", padx=2)
        ttk.Button(f_s, text="📋 粘贴(Ctrl+V)", command=self.on_paste).pack(side="left", padx=2)
        
        f_preview_container = tk.Frame(f_sch, bg="#e0e0e0", bd=1, relief="sunken", height=300)
        f_preview_container.pack(fill="x", padx=5, pady=5)
        
        # 【核心修改】禁止子控件改变容器大小，确保它永远是300高
        f_preview_container.pack_propagate(False) 
        
        self.lbl_preview = tk.Label(
            f_preview_container, 
            text="\n⬇️  暂无图片  ⬇️\n\n(粘贴图片 或 点击上方选图)", 
            bg="#e0e0e0", fg="#888",
            font=("微软雅黑", 12)
        )
        # expand=True 会让 Label 在固定的 300px 高度里垂直居中
        self.lbl_preview.pack(fill="both", expand=True)
        
        # 2. 底部参数与按钮行 (居中显示)
        f_ctrl = ttk.Frame(f_sch)
        f_ctrl.pack(fill="x", pady=(15, 5)) # 增加上方间距
        
        # 使用一个内部 Frame 来实现居中
        f_center_btns = ttk.Frame(f_ctrl)
        f_center_btns.pack(anchor="center")
        
        # Top K 设置
        ttk.Label(f_center_btns, text="结果个数:").pack(side="left")
        ttk.Spinbox(f_center_btns, from_=1, to=10000, textvariable=self.var_k, width=5).pack(side="left", padx=5)
        
        # 搜索按钮 (加大尺寸)
        self.btn_search = ttk.Button(f_center_btns, text="🔍 开始 AI 识别", command=self.do_search)
        self.btn_search.pack(side="left", padx=20, ipadx=20, ipady=5)
        
        # 历史按钮
        ttk.Button(f_center_btns, text="🕒 查看上次结果", command=self.check_last_result).pack(side="left")

        # --- 原有的 3.结果视图控制区 和 4.结果展示容器 已移除 ---
        # 界面到底部截止，不再显示列表和网格

        # 初始化视图模式变量 (虽然主界面不显示控件，但逻辑需要)
        self.view_mode = tk.StringVar(value="list")


    # --- 【修改】筛选逻辑 ---
    def on_filter_trigger(self, event=None):
        """
        当用户按下回车或点击筛选按钮时触发
        """
        # 如果没有原始数据，直接返回
        if not hasattr(self, 'all_raw_results') or not self.all_raw_results:
            return 

        keyword = self.var_filter.get().strip().lower()
        
        if not keyword:
            # 关键字为空，恢复全量数据
            self.search_results = list(self.all_raw_results)
        else:
            # 过滤：检查 NPK名 或 IMG名 是否包含关键字
            self.search_results = [
                item for item in self.all_raw_results
                if keyword in item['npk'].lower() or keyword in item['img'].lower()
            ]
        
        # 刷新界面显示
        self.refresh_data_display()
        
        # (可选) 如果是通过按钮触发的，可以将焦点设置回输入框，方便继续修改
        # if event is None: 
        #     self.focus_set()

    def clear_filter(self):
        """清除筛选"""
        self.var_filter.set("")
        self.on_filter_trigger() # 清除后立即刷新

    # --- 修改后的搜索逻辑 ---
    def do_search(self):
        if not self.is_ready:
            messagebox.showwarning("未就绪", "正在初始化 AI 引擎，请稍候...")
            return
            
        target = self.pasted_image if self.pasted_image else self.target_img_path.get()
        if not target: return
        
        try:
            img = None
            if isinstance(target, str):
                if os.path.exists(target):
                    img = Image.open(target).convert("RGBA")
            else:
                img = target
            
            if not img: return

            self.lbl_status.config(text="正在提取特征并遍历分片搜索...", foreground="blue")
            self.update()
            
            extractor = FeatureExtractor.get_instance()
            vec = extractor.extract(img)
            
            if vec is None:
                messagebox.showerror("错误", "无法提取特征 (可能是空白图片)")
                self.lbl_status.config(text="特征提取失败", foreground="red")
                return

            # --- 【修改】获取用户设定的 K 值 ---
            try:
                k_val = int(self.var_k.get())
                if k_val < 1: k_val = 500
            except:
                k_val = 500
            
            # 执行搜索
            results = self.db.query(vec, k=k_val) 
            
            # 1. 备份原始结果 (用于筛选还原)
            self.all_raw_results = list(results) 
            
            # 2. 重置筛选框 (新搜索开始，清空旧筛选)
            self.var_filter.set("")
            
            # 3. 设置当前显示数据并刷新
            self.search_results = results

            # 4. 【核心修改】直接打开结果弹窗，不再在主界面刷新
            self.show_result_window()
                
            self.lbl_status.config(text=f"搜索完成，找到 {len(results)} 个结果 (Top {k_val})")
            
        except Exception as e:
            print(f"Search error: {e}")
            self.lbl_status.config(text=f"搜索出错: {e}", foreground="red")

    # --- 新增：查看上次结果 ---
    def check_last_result(self):
        if not hasattr(self, 'search_results') or self.search_results is None:
            messagebox.showinfo("提示", "暂无历史记录，请先进行搜索。")
            return
        self.show_result_window()

    def _handle_ctrl_v(self, event):
        """
        处理 Ctrl+V 快捷键
        """
        # 1. 只有当前页面显示时才触发（防止在其他页面按粘贴时误触发这里的搜图）
        if not self.winfo_ismapped():
            return
            
        # 2. 直接调用粘贴逻辑
        # 如果剪贴板里是图片，on_paste 会自动识别并显示预览
        # 如果剪贴板里是普通文本，on_paste 里的 ImageGrab 会返回 None，不会有副作用
        self.on_paste()
    def init_result_view(self, parent_widget):
        """
        在指定的 parent_widget 中构建 列表+网格 视图
        """
        # 清空父容器（防止重复添加）
        for child in parent_widget.winfo_children():
            child.destroy()
        parent_widget.grid_rowconfigure(0, weight=1)
        parent_widget.grid_columnconfigure(0, weight=1)
        # === A. 列表视图 Frame ===
        self.f_tree_frame = ttk.Frame(parent_widget)
        self.f_tree_frame.grid(row=0, column=0, sticky="nsew")

        cols = ("score", "npk", "loc")
        self.tree = ttk.Treeview(self.f_tree_frame, columns=cols, show="headings")
        self.tree.heading("score", text="相似度"); self.tree.column("score", width=80)
        self.tree.heading("npk", text="NPK来源"); self.tree.column("npk", width=200)
        self.tree.heading("loc", text="位置"); self.tree.column("loc", width=200)
        
        vsb = ttk.Scrollbar(self.f_tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        
        # 重新绑定 Treeview 事件 (因为对象重建了，必须重绑)
        self.tree.bind("<Double-1>", self.on_preview_double_click)
        self.tree.bind("<Motion>", self.on_tree_hover)
        self.tree.bind("<Leave>", self.on_tree_leave)
        self.tree.bind("<Button-3>", self.on_right_click_menu)

        # === B. 网格视图 Frame ===
        self.f_grid_frame = ttk.Frame(parent_widget)
        # 默认隐藏，由 switch_view 控制显示
        self.f_grid_frame.grid(row=0, column=0, sticky="nsew")
        self.grid_canvas = tk.Canvas(self.f_grid_frame, bg="#e0e0e0")
        self.grid_vsb = ttk.Scrollbar(self.f_grid_frame, orient="vertical", command=self.grid_canvas.yview)
        self.grid_canvas.configure(yscrollcommand=self.grid_vsb.set)
        
        # 滚轮绑定 (必须在此重新绑定，因为 Canvas 是新的)
        self.grid_canvas.bind('<Enter>', lambda e: self.grid_canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.grid_canvas.bind('<Leave>', lambda e: self.grid_canvas.unbind_all("<MouseWheel>"))

        self.grid_canvas.pack(side="left", fill="both", expand=True)
        self.grid_vsb.pack(side="right", fill="y")
        
        self.grid_inner = tk.Frame(self.grid_canvas, bg="#e0e0e0")
        self.grid_canvas.create_window((0,0), window=self.grid_inner, anchor="nw")
        self.grid_inner.bind("<Configure>", lambda e: self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all")))
         # 记录当前列数，默认5列
        self.current_cols = 5 
        # 当画布大小改变时，触发 on_grid_resize
        self.grid_canvas.bind('<Configure>', self.on_grid_resize)
        # 根据当前的模式 (List/Grid) 决定显示谁
        self.switch_view()
    
    def on_grid_resize(self, event):
        """当窗口大小改变时，计算能容纳多少列"""
        # 如果当前不是网格视图，或者是空的，就不算
        if self.view_mode.get() != "grid" or not self.search_results:
            return
            
        # 145 是估算的单个格子宽度: 
        # 130(图片框宽度) + 10(左右padding各5) + 5(滚动条预留余量)
        cell_width = 145
        
        # 计算新列数 (至少显示 1 列)
        available_width = event.width
        new_cols = max(1, available_width // cell_width)
        
        # 【核心优化】只有当列数真的变了才刷新，避免拖动窗口时界面闪烁/卡死
        if new_cols != self.current_cols:
            self.current_cols = new_cols
            # 重新排列网格
            self.populate_grid()
            
    def remove_duplicate_imgs(self):
        """
        逻辑去重：移除同一个 NPK 中同一个 IMG 的重复条目。
        只保留该 IMG 中相似度最高（排名最靠前）的那一帧。
        """
        if not self.search_results:
            messagebox.showinfo("提示", "列表中没有数据。")
            return

        seen_keys = set()
        unique_data = []
        
        # 1. 数据清洗
        # 列表本身已经是按相似度从高到低排序的
        # 所以我们遇到的第一个 (NPK, IMG) 组合，一定就是该 IMG 里得分最高的
        for item in self.search_results:
            # 唯一标识符 = (NPK文件名, 内部IMG路径)
            # 只有这两个都相同，才算是"同一个文件"
            key = (item['npk'], item['img'])
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_data.append(item)
        
        removed_count = len(self.search_results) - len(unique_data)
        
        if removed_count == 0:
            messagebox.showinfo("提示", "没有发现同一个 IMG 的重复帧。")
            return

        # 2. 更新当前展示数据
        self.search_results = unique_data
        
        # 3. 刷新视图 (列表 + 网格)
        self.refresh_data_display()
        
        # 4. 更新提示
        msg = f"已清理 {removed_count} 条重复帧。剩余 {len(unique_data)} 个唯一 IMG 文件。"
        self.lbl_status.config(text=msg)
        
    def show_result_window(self):
        """
        打开或激活独立结果窗口。
        原 open_detached_window 改造而来，现在是唯一的结果展示入口。
        """
        if self.detached_win is not None and self.detached_win.winfo_exists():
            self.detached_win.lift() # 如果已经开了，就置顶
            # 即使窗口已存在，也要刷新数据，确保是最新搜索结果
            self.refresh_data_display()
            return

        # 创建新窗口
        self.detached_win = tk.Toplevel(self)
        self.detached_win.title("AI 搜索结果")
        self.detached_win.geometry("1000x700") 
        self.detached_win.resizable(True, True) 
        self.detached_win.bind("<Left>", lambda e: self.on_key_page_change(-1, e))
        self.detached_win.bind("<Right>", lambda e: self.on_key_page_change(1, e))
        # =========================================================
        # 【新增】 在独立窗口顶部添加 视图切换按钮
        # =========================================================
        f_ctrl = ttk.Frame(self.detached_win)
        f_ctrl.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(f_ctrl, text="视图模式: ").pack(side="left")
        # 直接复用 self.view_mode 和 self.switch_view
        # 这样无论在主界面还是新窗口，逻辑都是通用的
        ttk.Radiobutton(f_ctrl, text="📄 列表视图", variable=self.view_mode, value="list", 
                        command=self.switch_view).pack(side="left")
        ttk.Radiobutton(f_ctrl, text="▦ 网格视图", variable=self.view_mode, value="grid", 
                        command=self.switch_view).pack(side="left", padx=10)
        
        # 【修改】独立窗口也加上筛选框，逻辑同步
        ttk.Label(f_ctrl, text="   ⚡ 筛选: ").pack(side="left")
        
        e_fil = ttk.Entry(f_ctrl, textvariable=self.var_filter, width=20)
        e_fil.pack(side="left", padx=5)
        
        # 1. 绑定回车
        e_fil.bind("<Return>", self.on_filter_trigger)
        
        # 2. 新增按钮
        ttk.Button(f_ctrl, text="🔎", width=3, command=self.on_filter_trigger).pack(side="left")
        ttk.Button(f_ctrl, text="✖", width=3, command=self.clear_filter).pack(side="left")
        self.lbl_result_count = ttk.Label(f_ctrl, text="", foreground="blue", font=("微软雅黑", 10))
        self.lbl_result_count.pack(side="left", padx=20) 
        f_pager = ttk.Frame(f_ctrl)
        f_pager.pack(side="right") # 靠右
        ttk.Label(f_pager, text="每页:").pack(side="left")
        sp_size = ttk.Spinbox(f_pager, from_=10, to=500, increment=10, 
                              textvariable=self.var_page_size, width=4)
        sp_size.pack(side="left", padx=(0, 10))
        # 绑定回车键：输入数字后按回车生效
        sp_size.bind("<Return>", self.on_page_size_change)
        # 绑定焦点离开：点别的地方也生效
        sp_size.bind("<FocusOut>", self.on_page_size_change)
        ttk.Button(f_pager, text="< 上一页", command=lambda: self.change_page(-1)).pack(side="left", padx=2)
        
        # 页码标签 (居中)
        self.lbl_page_info = ttk.Label(f_pager, text="1 / 1", width=10, anchor="center", font=("Arial", 9, "bold"))
        self.lbl_page_info.pack(side="left", padx=5)
        
        ttk.Button(f_pager, text="下一页 >", command=lambda: self.change_page(1)).pack(side="left", padx=2)
        ttk.Label(f_pager, text="前往:").pack(side="left", padx=(10, 2))
        
        # 跳转输入框 (宽度设小一点)
        entry_jump = ttk.Entry(f_pager, textvariable=self.var_jump_page, width=4)
        entry_jump.pack(side="left")
        
        # 绑定回车键跳转
        entry_jump.bind("<Return>", self.jump_to_page)
        # 3. 创建容器 (用于挂载列表/网格)
        # 注意：这里创建一个新的 frame 作为容器，填满剩余空间
        detached_container = ttk.Frame(self.detached_win)
        detached_container.pack(fill="both", expand=True, padx=10, pady=5)

        # 4. 在新窗口里生成视图
        # 将视图挂载到刚才创建的 detached_container 里
        self.init_result_view(detached_container)

        # 5. 重新渲染数据
        self.refresh_data_display()

        # 6. 监听关闭事件 (关闭时要恢复到主界面)
        self.detached_win.protocol("WM_DELETE_WINDOW", self.on_detached_close)
    
    def jump_to_page(self, event=None):
        """跳转到指定页码"""
        if not self.search_results: return
        
        try:
            # 获取用户输入
            target_page_str = self.var_jump_page.get().strip()
            if not target_page_str: return
            
            target_page = int(target_page_str)
            
            # 计算总页数
            total_items = len(self.search_results)
            total_pages = (total_items + self.page_size - 1) // self.page_size
            
            # 校验范围 (用户输入1代表第1页，程序内部是0)
            if 1 <= target_page <= total_pages:
                # 核心跳转逻辑
                self.current_page = target_page - 1 
                self.populate_grid()
                self.grid_canvas.yview_moveto(0) # 滚回顶部
                
                # (可选) 跳转成功后清空输入框，或者保留显示
                # self.var_jump_page.set("") 
                
                # 让输入框失去焦点，避免误触
                self.detached_win.focus_set()
            else:
                # 输入超出范围，可以不做反应，或者重置为当前页
                self.var_jump_page.set(str(self.current_page + 1))
                
        except ValueError:
            # 输入了非数字
            self.var_jump_page.set("")
            
    def on_key_page_change(self, delta, event):
        """键盘翻页处理"""
        # 如果焦点在输入框(Entry/Spinbox)内，不触发翻页，否则无法移动光标修改文字
        if event.widget.winfo_class() in ['Entry', 'TEntry', 'Spinbox', 'TSpinbox']:
            return
        self.change_page(delta)

    def on_page_size_change(self, event=None):
        """当用户修改每页数量时触发"""
        try:
            new_size = self.var_page_size.get()
            if new_size < 1: new_size = 50 # 最小保护
            
            # 只有数值真的变了才刷新
            if new_size != self.page_size:
                self.page_size = new_size
                self.current_page = 0 # 重置回第一页
                self.populate_grid()  # 重新渲染
                # 让输入框失去焦点，避免一直占着
                self.detached_win.focus_set()
        except:
            pass
            
    def on_detached_close(self):
        """独立窗口关闭时，恢复主界面视图"""
        # 1. 销毁独立窗口
        if self.detached_win:
            self.detached_win.destroy()
            self.detached_win = None
        # 注意：这里不再需要恢复主界面视图，因为主界面已经没有视图了

    def _on_mousewheel(self, event):
        """处理鼠标滚轮滚动 + 触底自动翻页"""
        # 1. 计算滚动方向
        # Windows下: event.delta < 0 代表向下滚，direction > 0
        scroll_units = int(-1 * (event.delta / 120))
        
        # 2. 执行滚动
        self.grid_canvas.yview_scroll(scroll_units, "units")
        
        # 3. 获取当前视图位置
        # yview() 返回一个元组 (top, bottom)，范围是 0.0 到 1.0
        # top: 当前可见区域顶部在总长度的百分比
        # bottom: 当前可见区域底部在总长度的百分比
        top, bottom = self.grid_canvas.yview()
        
        # 4. 判断是否触发翻页
        # 条件A: scroll_units > 0 (用户正在试图向下滑动)
        # 条件B: bottom >= 1.0 (滚动条已经到底了)
        # 条件C: 确实有数据 (防止空列表乱翻)
        if scroll_units > 0 and bottom >= 1.0 and self.search_results:
            # 触发下一页
            self.change_page(1)
            
        # (可选) 如果滑到顶端想去上一页，可以把下面注释解开
        elif scroll_units < 0 and top <= 0.0 and self.search_results:
             self.change_page(-1)
    def switch_view(self):
        mode = self.view_mode.get()
        # 必须检查控件是否存在（因为可能在窗口创建前调用，或窗口已关闭）
        if not hasattr(self, 'f_tree_frame') or not self.f_tree_frame.winfo_exists():
            return

        if mode == "list":
            # 将列表视图提升到最顶层
            self.f_tree_frame.tkraise()
        else:
            # 将网格视图提升到最顶层
            self.f_grid_frame.tkraise()
            if hasattr(self, 'grid_inner') and not self.grid_inner.winfo_children() and hasattr(self, 'search_results') and self.search_results:
                self.populate_grid()
    def sel_index_lib(self):
        p = filedialog.askdirectory()
        if p:
            self.index_lib_path.set(p)
            Config.set("ai_index_lib", p)
            self.load_index_bg() # 选完自动重新加载
    def sel_repo(self):
        p = filedialog.askdirectory()
        if p: self.repo_path.set(p); Config.set("ai_repo", p)

    def sel_target(self):
        p = filedialog.askopenfilename(filetypes=[("Image", "*.png;*.jpg;*.bmp")])
        if p:
            self.target_img_path.set(p)
            self.pasted_image = None
            
            try:
                # --- [修改] ---
                img = Image.open(p)
                # 尺寸设为 500x300 或适应你屏幕的大小
                img.thumbnail((500, 300)) 
                self.tk_preview = ImageTk.PhotoImage(img)
                self.lbl_preview.config(image=self.tk_preview, text="", bg="#e0e0e0")
                # --- [修改结束] ---
            except Exception as e:
                self.lbl_preview.config(image="", text=f"预览失败: {e}")

    def on_paste(self):
        try:
            content = ImageGrab.grabclipboard()
            if isinstance(content, Image.Image):
                safe_image = content.convert("RGB")
                # 简单裁切空白
                try:
                    arr = np.array(safe_image)
                    bg_color = arr[0, 0]
                    diff = np.sum(np.abs(arr - bg_color), axis=2)
                    rows = np.any(diff > 30, axis=1)
                    cols = np.any(diff > 30, axis=0)
                    if np.any(rows) and np.any(cols):
                        y1, y2 = np.where(rows)[0][[0, -1]]
                        x1, x2 = np.where(cols)[0][[0, -1]]
                        safe_image = safe_image.crop((max(0, x1-2), max(0, y1-2), min(safe_image.width, x2+2), min(safe_image.height, y2+2)))
                except: pass

                self.pasted_image = safe_image
                self.target_img_path.set("[剪贴板内容]")
                
                pv = safe_image.copy()
                # 尺寸设为 500x300
                pv.thumbnail((500, 300))
                self.tk_preview = ImageTk.PhotoImage(pv)
                # 去掉 height=100 这种固定高度，让它自适应
                self.lbl_preview.config(image=self.tk_preview, text="", bg="#e0e0e0")
                
            elif isinstance(content, list) and content:
                if os.path.isfile(content[0]):
                    self.target_img_path.set(content[0])
                    self.pasted_image = None
                    self.lbl_preview.config(image="", text=f"[路径]: {os.path.basename(content[0])}", height=2)
        except Exception as e:
            print(f"Paste error: {e}")
    def on_tree_hover(self, event):
        """鼠标在列表上移动时触发"""
        # 1. 识别鼠标下方的行
        item_id = self.tree.identify_row(event.y)
        
        # 情况 A: 鼠标在空白处（比如列表底部空白，或者表头）
        if not item_id:
            self.close_preview_window()
            self.last_preview_id = None
            return

        # 情况 B: 鼠标还在同一行上
        if item_id == self.last_preview_id:
            # 可选：让浮窗跟随鼠标移动（如果觉得闪烁可以注释掉下面两行）
            # if self.preview_win:
            #     self.preview_win.geometry(f"+{event.x_root + 20}+{event.y_root + 20}")
            return
            
        # 情况 C: 鼠标移到了新的一行 -> 开始加载
        self.last_preview_id = item_id
        
        # 先关闭旧的（或者显示一个加载中的状态）
        # 这里我们选择先不关闭旧的直接覆盖，或者显示Loading，体验更流畅
        # 但为了防止旧图残留，建议先清理内容
        
        # 获取数据
        try:
            full_npk_path = self.tree.item(item_id, "tags")[0]
            values = self.tree.item(item_id, "values")
            if len(values) < 3: return
            loc_str = values[2] # "xxx.img # 10"
            parts = loc_str.rsplit(" # ", 1)
            img_path_inner = parts[0]
            frame_idx = int(parts[1])
        except:
            return

        # 显示“加载中”提示 (跟随当前鼠标位置)
        self.show_floating_window(event.x_root, event.y_root, loading=True)
        
        # 后台加载 (传入当前的 item_id 用于校验，防止鼠标动太快导致图片错乱)
        threading.Thread(target=self.load_image_for_hover, 
                         args=(full_npk_path, img_path_inner, frame_idx, item_id)).start()

    def on_tree_leave(self, event):
        """鼠标离开列表控件时触发"""
        self.close_preview_window()
        self.last_preview_id = None

    # ---------------------------------------------------------------------
    # 加载与显示逻辑 (适配悬停模式)
    # ---------------------------------------------------------------------
    def load_image_for_hover(self, npk_path, img_path, frame_idx, request_id):
        try:
            if not os.path.exists(npk_path): return
            # 使用 fast_load_frame_from_npk 提高悬停加载速度
            pil_img = fast_load_frame_from_npk(npk_path, img_path, frame_idx)
            if pil_img and self.last_preview_id == request_id:
                self.after(0, lambda: self.update_preview_content_hover(pil_img))
        except Exception as e: pass

    def update_preview_content_hover(self, pil_img):
        if not self.preview_win: return
        mx = self.winfo_pointerx()
        my = self.winfo_pointery()
        self.show_floating_window(mx, my, loading=False, pil_img=pil_img)
    def start_build(self):
        messagebox.showinfo("提示", "由于数据量巨大，请运行独立的 RunTrain.py 脚本进行构建。\n。")

    def load_index_bg(self):
        target_dir = self.index_lib_path.get()
        if not os.path.exists(target_dir):
            self.lbl_status.config(text="❌ 索引路径不存在", foreground="red")
            return
        # 【新增】检查加载路径是否含中文
        if not is_ascii_path(target_dir):
            self.lbl_status.config(text="❌ 路径含中文，无法读取，请移动索引到纯英文路径！", foreground="red")
            return
        self.lbl_status.config(text="⏳ 正在加载索引...", foreground="blue")
        def _run():
            try:
                FeatureExtractor.get_instance()
                # 【修改】传入路径
                self.db.init_storage(mode='read', db_dir=target_dir)
                self.is_ready = True
                count = self.db.get_count()
                self.after(0, lambda: self.lbl_status.config(text=f"✅ AI 引擎就绪 | 索引库: {os.path.basename(target_dir)} (约 {count} 条数据)", foreground="green"))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda: self.lbl_status.config(text=f"❌ 加载失败: {err_msg}", foreground="red"))
        threading.Thread(target=_run).start()

    def restore_results(self):
        if not hasattr(self, 'all_raw_results') or not self.all_raw_results:
            messagebox.showinfo("提示", "没有可还原的历史记录。")
            return
        self.search_results = list(self.all_raw_results)
        self.var_filter.set("") # 清除筛选框
        self.refresh_data_display()
        
        self.lbl_status.config(text=f"已还原所有结果，共 {len(self.search_results)} 条。")
        
    def refresh_data_display(self):
        """将 self.search_results 的数据渲染到当前的 Treeview 和 Grid 中"""
        # 1. 列表视图刷新逻辑
        self.current_page = 0 
        if hasattr(self, 'tree') and self.tree.winfo_exists():
            for i in self.tree.get_children(): self.tree.delete(i)
            repo_root = self.repo_path.get()
            
            # 重新填充列表
            for idx, r in enumerate(self.search_results):
                score_display = f"{int(r['score']*100)}%"
                if repo_root and os.path.exists(repo_root):
                    real_full_path = os.path.join(repo_root, r['npk'])
                else:
                    real_full_path = r['orig_path']
                
                self.tree.insert("", "end", values=(score_display, r['npk'], f"{r['img']} # {r['frame']}"), 
                                 tags=(real_full_path, str(idx)))

        # 2. 刷新网格 (如果当前是网格视图)
        if hasattr(self, 'grid_inner') and self.grid_inner.winfo_exists():
            for w in self.grid_inner.winfo_children(): w.destroy()
            self.thumbnail_cache = {}
            if self.view_mode.get() == "grid" and self.search_results:
                self.populate_grid() # 这里会调用分页逻辑
        
        # 3. 更新状态文本 (带筛选提示)
        count = len(self.search_results)
        total = len(getattr(self, 'all_raw_results', []))
        filter_txt = self.var_filter.get()
        
        # 构造提示语
        if filter_txt:
            msg = f"📊 筛选结果: {count} / {total} 条"
        else:
            msg = f"📊 共找到 {count} 条结果"

        # 1. 更新主界面底部的状态栏 (可选)
        if hasattr(self, 'lbl_status') and self.lbl_status.winfo_exists():
            self.lbl_status.config(text=msg)

        # 2. 更新独立窗口顶部的数量 Label
        if hasattr(self, 'lbl_result_count') and self.lbl_result_count.winfo_exists():
            self.lbl_result_count.config(text=msg)
            
    def change_page(self, delta):
        """翻页处理函数: delta 为 -1 或 1"""
        if not self.search_results: return
        
        # 计算总页数
        total_items = len(self.search_results)
        total_pages = (total_items + self.page_size - 1) // self.page_size
        
        # 计算新页码
        new_page = self.current_page + delta
        
        # 边界检查
        if 0 <= new_page < total_pages:
            self.current_page = new_page
            # 翻页后重新填充网格
            self.populate_grid()
            # 翻页后滚轮回到顶部
            self.grid_canvas.yview_moveto(0)
            
    def populate_grid(self):
        # 1. 清空现有网格
        self.hide_tooltip(None) # <--- 【新增】强制清除残留的提示框
        try:
            val = self.var_page_size.get()
            if val > 0: self.page_size = val
        except: pass
        
        for w in self.grid_inner.winfo_children(): w.destroy()
        self.thumbnail_cache = {} # 清理缓存引用(可选)
        
        if not self.search_results: 
            if hasattr(self, 'lbl_page_info'): self.lbl_page_info.config(text="0 / 0")
            return

        # =========================================================
        # --- [核心修改] 分页切片逻辑 ---
        # =========================================================
        total_items = len(self.search_results)
        # 计算总页数
        total_pages = (total_items + self.page_size - 1) // self.page_size
        
        # 确保当前页码不越界 (防止筛选后页码溢出)
        if self.current_page >= total_pages: self.current_page = 0
        
        # 计算切片索引
        start_idx = self.current_page * self.page_size
        end_idx = start_idx + self.page_size
        # 获取当前页的数据
        current_page_data = self.search_results[start_idx:end_idx]
        
        # 更新页码显示
        if hasattr(self, 'lbl_page_info'):
            self.lbl_page_info.config(text=f"{self.current_page + 1} / {total_pages}")
        # =========================================================

        cols = self.current_cols # 使用 resize 事件里计算出的列数
        repo_root = self.repo_path.get()
        self.grid_widgets = [] 
        
        # 注意：这里循环的是 current_page_data (只有50条)，不是整个结果集
        # i 是当前页的索引，我们需要计算出它在全局的真实索引用于定位
        for i, r in enumerate(current_page_data):
            row = i // cols
            col = i % cols
            
            cell = tk.Frame(self.grid_inner, bg="white", bd=1, relief="solid", width=130, height=130)
            cell.grid(row=row, column=col, padx=5, pady=5)
            cell.pack_propagate(False) 
            
            lbl_img = tk.Label(cell, text="Loading...", bg="#eee")
            lbl_img.pack(fill="both", expand=True)
            
            if repo_root and os.path.exists(repo_root):
                full_path = os.path.join(repo_root, r['npk'])
            else:
                full_path = r['orig_path']

            # 计算真实索引：用于"定位到列表"等功能
            real_index = start_idx + i 

            lbl_img.bind("<Double-1>", lambda e, p=full_path, im=r['img'], fr=r['frame']: 
                         self.open_compare_viewer(p, im, fr))
            # 传入 real_index 确保定位准确
            lbl_img.bind("<Button-3>", lambda e, idx=real_index: self.on_grid_right_click(e, idx))
            lbl_img.bind("<Enter>", lambda e, name=r['npk']: self.show_tooltip(e, name))
            lbl_img.bind("<Leave>", self.hide_tooltip)
            
            self.grid_widgets.append((lbl_img, full_path, r['img'], r['frame']))

        # 启动后台加载图片 (只加载这50张)
        threading.Thread(target=self.load_grid_thumbnails, daemon=True).start()

    def load_grid_thumbnails(self):
        # 遍历所有格子，加载图片
        for i, (lbl, full_path, img_inner, frame_idx) in enumerate(self.grid_widgets):
            try:
                # 0. 只有当控件还存在时才加载
                if not lbl.winfo_exists(): break
                
                # 1. 生成缓存唯一 Key (路径 + 内部文件名 + 帧号)
                cache_key = f"{full_path}::{img_inner}::{frame_idx}"
                
                # 2. 【第一道防线】检查全局缓存
                cached_tk_img = GlobalImageCache.get(cache_key)
                
                if cached_tk_img:
                    # 命中缓存！直接更新UI，耗时 0ms
                    self.after(0, lambda l=lbl, img=cached_tk_img: self.apply_grid_image(l, img))
                    continue

                # 3. 【第二道防线】缓存未命中，执行极速读取
                pil_img = fast_load_frame_from_npk(full_path, img_inner, frame_idx)
                
                if pil_img:
                    # 缩放成缩略图 (减少内存占用)
                    pil_img.thumbnail((120, 120))
                    
                    # 转为 Tkinter 对象
                    tk_img = ImageTk.PhotoImage(pil_img)
                    
                    # 存入全局缓存
                    GlobalImageCache.set(cache_key, tk_img)
                    
                    # 回到主线程更新
                    self.after(0, lambda l=lbl, img=tk_img: self.apply_grid_image(l, img))
                else:
                    # 【新增】如果读不到图，显示一个"错误"文本，不再显示Loading
                    # 这样你就知道是读取失败了，而不是还在读
                    self.after(0, lambda l=lbl: l.config(text="❌", fg="red"))
            except Exception as e:
                pass 

    # 辅助方法：更新 UI (把它单独提出来是为了代码整洁)
    def apply_grid_image(self, label, tk_img):
        if label.winfo_exists():
            label.config(image=tk_img, text="")

    def show_tooltip(self, event, text):
        self.hide_tooltip(None)
        self.tooltip = tk.Toplevel(self)
        self.tooltip.wm_overrideredirect(True) # 无边框
        self.tooltip.wm_geometry(f"+{event.x_root+15}+{event.y_root+15}")
        self.tooltip.attributes("-topmost", True)
        
        tk.Label(self.tooltip, text=text, bg="#FFFFE0", fg="black", 
                 relief="solid", bd=1, padx=5, pady=2).pack()

    def hide_tooltip(self, event):
        if hasattr(self, 'tooltip') and self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None # <--- 【建议】销毁后置空，防止野指针

    # --- 右键菜单 & 定位 ---
    def on_grid_right_click(self, event, index):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="🎯 定位到列表", command=lambda: self.locate_in_list(index))
        # 也可以加复制名字等功能
        menu.post(event.x_root, event.y_root)

    def locate_in_list(self, index):
        # 1. 切换回列表视图
        self.view_mode.set("list")
        self.switch_view()
        
        # 2. 在 Treeview 中查找
        # 因为我们插入时是按顺序的，所以 get_children()[index] 通常就是对应的项
        children = self.tree.get_children()
        if index < len(children):
            item_id = children[index]
            
            # 3. 选中并滚动
            self.tree.selection_set(item_id)
            self.tree.focus(item_id)
            self.tree.see(item_id) # 滚动到可见区域

    # --- 辅助：打开对比窗口 (提取公用逻辑) ---
    def open_compare_viewer(self, full_path, inner_img, frame_idx):
        CompareViewer(self.winfo_toplevel(), file_pairs=[(full_path, None)], 
                      jump_to_img=inner_img, jump_to_frame=frame_idx)
    # ---------------------------------------------------------
    # 【新增功能 1】 右键菜单复制
    # ---------------------------------------------------------
    def on_right_click_menu(self, event):
        # 1. 识别右键点击的是哪一行
        item_id = self.tree.identify_row(event.y)
        if not item_id: 
            # 如果没点中行，创建一个只包含去重功能的菜单
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="🧹 移除重复 NPK (保留最高分)", 
                             command=self.remove_duplicate_npks)
            menu.add_command(label="🎞️ 移除重复 IMG (同文件只留一帧)", 
                             command=self.remove_duplicate_imgs)
            # 【新增】还原按钮
            menu.add_command(label="↩️ 还原所有结果", command=self.restore_results)
            menu.post(event.x_root, event.y_root)
            return
        # 2. 选中该行
        self.tree.selection_set(item_id)
        
        # 3. 获取数据
        values = self.tree.item(item_id, "values") # (score, npk_name, location)
        full_path = self.tree.item(item_id, "tags")[0] # NPK完整路径
        
        npk_name = values[1]
        location = values[2]
        
        # 4. 创建弹出菜单
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label=f"📄 复制 NPK 名称 ({npk_name})", 
                         command=lambda: self.copy_to_clipboard(npk_name))
        menu.add_command(label=f"📍 复制 位置信息 ({location})", 
                         command=lambda: self.copy_to_clipboard(location))
        menu.add_separator()
        menu.add_command(label="💾 复制 完整路径 (包含文件名)", 
                         command=lambda: self.copy_to_clipboard(f"{npk_name}\t{location}"))
        # --- 【新增】去重按钮 ---
        menu.add_separator()
        menu.add_command(label="🧹 移除重复 NPK (保留最高分)", 
                         command=self.remove_duplicate_npks)
        menu.add_command(label="🎞 移除重复 IMG (同文件只留一帧)", 
                         command=self.remove_duplicate_imgs)
        menu.add_command(label="↩ 还原所有结果", command=self.restore_results)
        menu.post(event.x_root, event.y_root)
    def remove_duplicate_npks(self):
        """逻辑去重：操作底层数据并同步视图"""
        if not self.search_results:
            messagebox.showinfo("提示", "列表中没有数据。")
            return

        seen_npks = set()
        unique_data = []
        
        # 1. 数据清洗
        # 列表已经按相似度排序，所以保留遇到的第一个即可
        for item in self.search_results:
            npk_name = item['npk']
            if npk_name not in seen_npks:
                seen_npks.add(npk_name)
                unique_data.append(item)
        
        removed_count = len(self.search_results) - len(unique_data)
        
        if removed_count == 0:
            messagebox.showinfo("提示", "没有发现重复的 NPK 文件。")
            return

        # 2. 更新当前展示数据
        self.search_results = unique_data
        
        # 3. 【核心】调用总刷新，列表和网格会同时更新
        self.refresh_data_display()
        
        # 4. 更新提示
        msg = f"已清理 {removed_count} 条重复数据。剩余 {len(unique_data)} 个唯一 NPK 文件。"
        self.lbl_status.config(text=msg)
        
    def copy_to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update() # 必须调用以保持剪贴板内容

    # ---------------------------------------------------------
    # 【新增功能 2】 单击浮窗预览
    # ---------------------------------------------------------
    def on_middle_click_preview(self, event):
        # 1. 获取选中的条目
        item_id = self.tree.identify_row(event.y)
        if not item_id: return
        
        # 如果点击的是同一个条目，就不重新加载（避免闪烁），但要确保窗口还在
        if item_id == self.last_preview_id and self.preview_win and self.preview_win.winfo_exists():
            return
            
        self.last_preview_id = item_id
        
        # 2. 获取数据
        full_npk_path = self.tree.item(item_id, "tags")[0]
        loc_str = self.tree.item(item_id, "values")[2] # "sprite/map/xxx.img # 10"
        
        # 解析 img 和 frame
        try:
            parts = loc_str.rsplit(" # ", 1)
            img_path_inner = parts[0]
            frame_idx = int(parts[1])
        except:
            return

        # 3. 后台加载图片 (因为解压NPK可能微卡)
        # 为了响应快，这里做个简单的加载提示
        self.show_floating_window(event.x_root, event.y_root, loading=True)
        
        # 开启线程去读取图片
        threading.Thread(target=self.load_image_for_preview, 
                         args=(full_npk_path, img_path_inner, frame_idx)).start()

    def show_floating_window(self, x, y, loading=False, pil_img=None):
        # 如果窗口不存在，创建它
        if not self.preview_win:
            top = tk.Toplevel(self)
            top.overrideredirect(True) # 无边框
            top.attributes("-topmost", True) # 置顶
            # 设置背景色为深色，看起来更像 Tooltip
            top.config(bg="#333", bd=1, relief="solid")
            
            # 鼠标穿透 (可选：如果你希望鼠标能穿过预览窗点击下面的东西，但在Tkinter里实现比较复杂，这里先不加)
            
            lbl = tk.Label(top, bg="#333", fg="white")
            lbl.pack(padx=2, pady=2)
            self.preview_win = top
        
        # 获取 Label 控件
        label_widget = self.preview_win.winfo_children()[0]
        
        # 设置偏移量，让窗口显示在鼠标右下角，避免遮挡鼠标
        offset_x = 20
        offset_y = 20

        if loading:
            label_widget.config(image="", text="Loading...", font=("Arial", 9))
            # 简单定个位置
            self.preview_win.geometry(f"+{x+offset_x}+{y+offset_y}")
            
        elif pil_img:
            # 缩放逻辑
            max_size = 350
            w, h = pil_img.size
            ratio = min(max_size/w, max_size/h)
            if ratio < 1:
                new_w, new_h = int(w*ratio), int(h*ratio)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            self.preview_tk_img = ImageTk.PhotoImage(pil_img)
            label_widget.config(image=self.preview_tk_img, text="")
            
            # 智能位置调整：防止超出屏幕底部
            screen_h = self.winfo_screenheight()
            screen_w = self.winfo_screenwidth()
            
            final_x = x + offset_x
            final_y = y + offset_y
            
            win_h = pil_img.height + 10
            win_w = pil_img.width + 10
            
            # 如果超出底部，往上翻
            if final_y + win_h > screen_h:
                final_y = y - win_h - 10
            
            # 如果超出右边，往左翻
            if final_x + win_w > screen_w:
                final_x = x - win_w - 10
                
            self.preview_win.geometry(f"+{final_x}+{final_y}")
            # 确保窗口显示出来
            self.preview_win.deiconify()

    def close_preview_window(self, event=None):
        if self.preview_win:
            self.preview_win.destroy()
            self.preview_win = None

    def load_image_for_preview(self, npk_path, img_path, frame_idx):
        try:
            pil_img = fast_load_frame_from_npk(npk_path, img_path, frame_idx)
            if pil_img:
                self.after(0, lambda: self.update_preview_content(pil_img))
        except Exception as e: pass

    def update_preview_content(self, pil_img):
        # 获取当前鼠标位置，确保浮窗跟随
        mx = self.winfo_pointerx()
        my = self.winfo_pointery()
        self.show_floating_window(mx, my, loading=False, pil_img=pil_img) 

    def on_preview_double_click(self, e):
        # 原有的双击进入对比模式
        self.close_preview_window(None) # 双击时关闭浮窗
        sel = self.tree.selection()
        if not sel: return
        full_path = self.tree.item(sel[0], 'tags')[0]
        values = self.tree.item(sel[0], 'values')
        loc_str = values[2]
        t_img, t_frame = None, 0
        if " # " in loc_str:
            try:
                parts = loc_str.rsplit(" # ", 1)
                t_img = parts[0]
                t_frame = int(parts[1])
            except: pass
        CompareViewer(self.winfo_toplevel(), file_pairs=[(full_path, None)], jump_to_img=t_img, jump_to_frame=t_frame)
        
        
# =========================================================================
# PART 4.7: 【AI 训练】构建特征索引库 (单线程 GUI 版)
# =========================================================================

class TrainPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.npk_dir = tk.StringVar(value=Config.get("train_npk_dir", ""))
        self.save_dir = tk.StringVar(value=Config.get("train_save_dir", os.getcwd()))
        self.is_running = False
        
        # 训练参数
        self.batch_size = 256 # 显存小的话调小
        self.shard_size = 1000000 
        
        if not HAS_AI_DEPS:
            tk.Label(self, text="❌ 缺少深度学习依赖库 (torch/torchvision/faiss)", fg="red", font=("微软雅黑", 14)).pack(pady=50)
            return

        self.create_ui()

    def create_ui(self):
        tk.Label(self, text="AI 特征库构建 (训练)", font=("微软雅黑", 16, "bold"), fg="#333").pack(pady=10)
        
        # 设置区
        f_set = ttk.LabelFrame(self, text=" 1. 路径设置  请使用英文路径", padding=10)
        f_set.pack(fill="x", padx=10)
        
        # NPK 源目录
        f_r1 = ttk.Frame(f_set); f_r1.pack(fill="x", pady=2)
        ttk.Label(f_r1, text="NPK素材文件夹:", width=15).pack(side="left")
        ttk.Entry(f_r1, textvariable=self.npk_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(f_r1, text="📂 选择", command=self.sel_npk_dir).pack(side="left", padx=5)

        # 索引保存目录
        f_r2 = ttk.Frame(f_set); f_r2.pack(fill="x", pady=2)
        ttk.Label(f_r2, text="索引保存位置:", width=15).pack(side="left")
        ttk.Entry(f_r2, textvariable=self.save_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(f_r2, text="📂 选择", command=self.sel_save_dir).pack(side="left", padx=5)

        # 操作区
        f_act = ttk.LabelFrame(self, text=" 2. 操作 ", padding=10)
        f_act.pack(fill="x", padx=10, pady=10)
        
        self.btn_start = ttk.Button(f_act, text="🚀 开始构建索引", command=self.start_training)
        self.btn_start.pack(fill="x", pady=5)
        self.progress_val = tk.DoubleVar(value=0)
        self.pbar = ttk.Progressbar(f_act, mode='determinate', variable=self.progress_val)
        #self.pbar.pack(fill="x", pady=5)
        self.lbl_progress = ttk.Label(f_act, text="准备就绪", anchor="center")
        self.lbl_progress.pack(fill="x")
        # 日志区
        self.log_box = tk.Text(self, bg="#2c3e50", fg="#ecf0f1", font=("Consolas", 9), state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)

    def log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", str(msg) + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")
        self.update_idletasks() # 强制刷新UI

    def sel_npk_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.npk_dir.set(p)
            Config.set("train_npk_dir", p)

    def sel_save_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.save_dir.set(p)
            Config.set("train_save_dir", p)

    def start_training(self):
        if self.is_running: return
        
        npk_path = self.npk_dir.get()
        save_path = self.save_dir.get()
        
        if not npk_path or not os.path.exists(npk_path):
            messagebox.showerror("错误", "NPK 文件夹路径无效")
            return
            
        if not save_path or not os.path.exists(save_path):
            messagebox.showerror("错误", "保存路径无效")
            return
        # 【新增】检查保存路径是否含中文
        if not is_ascii_path(save_path):
            messagebox.showerror("路径错误", 
                f"索引保存路径不能包含中文或特殊字符！\n\n错误路径:\n{save_path}\n\n请选择纯英文路径 (例如 D:\\AI_Index)。")
            return
        self.is_running = True
        self.btn_start.config(state="disabled")
        self.progress_val.set(0)       # 归零
        self.pbar.pack(fill="x", pady=5) # 显示进度条
        self.lbl_progress.config(text="正在扫描文件...") # 更新文字
        
        # 在新线程中运行，防止卡死界面
        threading.Thread(target=self.run_logic, args=(npk_path, save_path)).start()

    def run_logic(self, npk_dir, save_dir):
        try:
            # 1. 初始化模型 (ONNX)
            import onnxruntime as ort
            import faiss
            
            model_name = "mobilenet_v3_small.onnx"
            if hasattr(sys, '_MEIPASS'):
                model_path = os.path.join(sys._MEIPASS, model_name)
            else:
                model_path = model_name

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"找不到模型文件: {model_path}")

            self.log(f"🧠 加载 ONNX 模型: {model_path}")
            
            # 创建会话
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            
            # 2. 初始化数据库 (升级版结构)
            db_path = os.path.join(save_dir, "dnf_ai_meta.db")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL")
            
            # 2.1 创建主表 (如果不存在)
            # 注意：新增 packed_size 字段用于去重
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    id INTEGER PRIMARY KEY, 
                    npk TEXT, 
                    img TEXT, 
                    frame INTEGER, 
                    shard_id INTEGER, 
                    local_id INTEGER,
                    packed_size INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_loc ON meta (shard_id, local_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_npk ON meta (npk)") # 加速查询
            
            # 2.2 自动迁移：如果旧数据库没有 packed_size 字段，添加它
            try:
                conn.execute("ALTER TABLE meta ADD COLUMN packed_size INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass # 字段已存在，忽略错误

            # 2.3 创建文件追踪表 (NPK 文件级指纹)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_tracking (
                    npk_path TEXT PRIMARY KEY,
                    mtime REAL,
                    file_size INTEGER
                )
            """)
            
            # 3. 初始化 Faiss
            current_shard_id = 0
            while os.path.exists(os.path.join(save_dir, f"dnf_ai_{current_shard_id}.index")):
                current_shard_id += 1
            
            # 维度 = 576 (MobileNet) + 512 (Color) = 1088
            index = faiss.IndexFlatIP(1088) 
            current_local_id = 0
            
            # 4. 预加载文件追踪信息 (第一层防线：内存缓存)
            self.log("📋 正在加载增量更新记录...")
            cursor = conn.execute("SELECT npk_path, mtime, file_size FROM file_tracking")
            # 格式: { 'D:/Game/a.npk': (167888.0, 102400) }
            tracking_cache = {row[0]: (row[1], row[2]) for row in cursor}
            
            # 5. 扫描文件
            all_npks = glob.glob(os.path.join(npk_dir, "*.npk"))
            total_files = len(all_npks)
            self.pbar.config(maximum=total_files)

            batch_imgs = []
            batch_colors = []
            batch_meta = []
            
            skipped_files_count = 0

            for i, npk_file in enumerate(all_npks):
                current_count = i + 1
                self.progress_val.set(current_count)
                percent = int((current_count / total_files) * 100)
                fname = os.path.basename(npk_file)
                self.lbl_progress.config(text=f"[{percent}%] {current_count}/{total_files}: {fname}")
                
                # --- 第一层漏斗：NPK 文件级过滤 ---
                try:
                    stat_info = os.stat(npk_file)
                    curr_mtime = stat_info.st_mtime
                    curr_size = stat_info.st_size
                    
                    # 检查缓存：如果路径存在，且修改时间、大小都一致，则完全跳过
                    if npk_file in tracking_cache:
                        cached_mtime, cached_size = tracking_cache[npk_file]
                        # 允许 1秒内的时间误差
                        if abs(curr_mtime - cached_mtime) < 1.0 and curr_size == cached_size:
                            skipped_files_count += 1
                            # 只有在日志框开启时才打印，避免刷屏
                            # self.log(f"⏩ 跳过未修改文件: {fname}")
                            continue 
                except Exception as e:
                    print(f"File stat error: {e}")

                # --- 第二层漏斗：IMG 内容级过滤 ---
                # 如果文件变了（或者新文件），进入内部检查
                self.log(f"🔍 扫描变动文件: {fname}")
                
                # 1. 查库：获取该 NPK 已有的图片指纹
                # 我们用 (img路径, packed_size) 作为唯一标识
                existing_imgs = set()
                try:
                    cur = conn.execute("SELECT img, packed_size FROM meta WHERE npk=?", (fname,))
                    for row in cur:
                        # row[0] = sprite/map/a.img, row[1] = 1024
                        p_size = row[1] if row[1] is not None else 0
                        existing_imgs.add((row[0], p_size))
                except: pass

                try:
                    with open(npk_file, 'rb') as f:
                        npk = NPK.open(f)
                        # 这里我们只读目录，不解压图片，速度极快
                        npk.load_all() 
                        
                        img_processed_count = 0
                        
                        for inner in npk.files:
                            if not inner.name.lower().endswith('.img'): continue
                            
                            # 获取 NPK 内部文件的压缩大小 (指纹)
                            # pydnfex 的 NPKFileEntry 通常有 .size (压缩大小) 或 .length
                            inner_packed_size = getattr(inner, 'size', getattr(inner, 'length', 0))
                            
                            # --- 核心去重逻辑 ---
                            # 如果 (文件名, 大小) 都在集合里，说明完全没变 -> 跳过
                            if (inner.name, inner_packed_size) in existing_imgs:
                                continue
                                
                            # 否则：这是新增的，或者大小变了的 -> 处理它！
                            try:
                                img_obj = IMGFactory.open(BytesIO(inner.data))
                                for frame_idx, frame in enumerate(img_obj.images):
                                    try:
                                        pil_img = img_obj.build(frame)
                                        rgba = AIImagePreprocessor.crop_and_resize(pil_img)
                                        if rgba is None: continue
                                        
                                        col_vec = AIImagePreprocessor.extract_color_histogram(rgba)
                                        cnn_pil = AIImagePreprocessor.add_background(rgba)
                                        onnx_input = AIImagePreprocessor.preprocess_for_onnx(cnn_pil)
                                        
                                        if onnx_input is not None:
                                            batch_imgs.append(onnx_input.squeeze(0)) 
                                            batch_colors.append(col_vec)
                                            # 元数据加入 packed_size
                                            batch_meta.append((fname, inner.name, frame_idx, inner_packed_size))
                                            img_processed_count += 1
                                        
                                        # --- 批处理推断 ---
                                        if len(batch_imgs) >= self.batch_size:
                                            input_batch = np.stack(batch_imgs)
                                            cnn_feats = session.run(None, {input_name: input_batch})[0]
                                            col_feats = np.stack(batch_colors)
                                            combined = np.concatenate([cnn_feats, col_feats], axis=1)
                                            faiss.normalize_L2(combined)
                                            index.add(combined)
                                            
                                            db_rows = []
                                            for k in range(len(combined)):
                                                m = batch_meta[k]
                                                # 插入数据，包括 packed_size
                                                db_rows.append((m[0], m[1], m[2], current_shard_id, current_local_id + k, m[3]))
                                            
                                            # SQL 增加 packed_size 字段
                                            conn.executemany("INSERT INTO meta (npk, img, frame, shard_id, local_id, packed_size) VALUES (?,?,?,?,?,?)", db_rows)
                                            conn.commit()
                                            
                                            current_local_id += len(combined)
                                            batch_imgs = []
                                            batch_colors = []
                                            batch_meta = []
                                            
                                            if index.ntotal >= self.shard_size:
                                                idx_path = os.path.join(save_dir, f"dnf_ai_{current_shard_id}.index")
                                                faiss.write_index(index, idx_path)
                                                current_shard_id += 1
                                                index.reset()
                                                current_local_id = 0
                                    except: continue
                            except: continue
                        
                        # 如果有处理图片，或者虽然没处理图片但检查通过了
                        # 更新文件追踪表 (标记该文件已处理为最新状态)
                        conn.execute("INSERT OR REPLACE INTO file_tracking (npk_path, mtime, file_size) VALUES (?, ?, ?)", 
                                     (npk_file, curr_mtime, curr_size))
                        conn.commit()
                        
                        if img_processed_count > 0:
                            self.log(f"  -> 更新/新增了 {img_processed_count} 张图片")
                        else:
                            # 这种情况是：文件时间变了，但里面所有IMG的大小和名字都没变（可能是二进制微调），也视为处理完成
                            pass

                except Exception as e:
                    self.log(f"⚠️ 处理出错: {fname} -> {e}")

            # 处理剩余数据
            if batch_imgs:
                input_batch = np.stack(batch_imgs)
                cnn_feats = session.run(None, {input_name: input_batch})[0]
                col_feats = np.stack(batch_colors)
                combined = np.concatenate([cnn_feats, col_feats], axis=1)
                faiss.normalize_L2(combined)
                index.add(combined)
                
                db_rows = []
                for k in range(len(combined)):
                    m = batch_meta[k]
                    db_rows.append((m[0], m[1], m[2], current_shard_id, current_local_id + k, m[3]))
                conn.executemany("INSERT INTO meta (npk, img, frame, shard_id, local_id, packed_size) VALUES (?,?,?,?,?,?)", db_rows)
                conn.commit()

            # 保存最后索引
            if index.ntotal > 0:
                idx_path = os.path.join(save_dir, f"dnf_ai_{current_shard_id}.index")
                faiss.write_index(index, idx_path)

            conn.close()
            
            final_msg = f"✅ 索引构建完成！\n本次共跳过 {skipped_files_count} 个未修改文件。"
            self.log(final_msg)
            messagebox.showinfo("完成", final_msg)

        except Exception as e:
            self.log(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"训练中断: {e}")
        finally:
            self.is_running = False
            self.pbar.pack_forget()
            self.btn_start.config(state="normal")