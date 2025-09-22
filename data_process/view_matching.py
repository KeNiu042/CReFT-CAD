import json, os, random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# # ---------- 可调参数 ----------
# TOTAL_IMAGES   = 20000          # 数据集中实际图片数量 (id: 1~20000)
# TARGET_SAMPLES = 500   # 需要生成的 QA 数
# MAX_WORKERS    = 8               # 线程数（按机器调整）
# # --------------------------------

# VIEW_DICT = {                     # 视角到子文件夹名映射
#     "Top view":   "view1",        # 俯视
#     "Front view": "view2",        # 主视
#     "Side view":  "view3"         # 侧视
# }
# VIEWS = list(VIEW_DICT.items())   # [(name, folder), ...]

# def add_path(view_folder: str, img_id: int, root: str) -> str:
#     """拼接完整路径"""
#     return os.path.join(root, view_folder, f"{img_id}.png")

# def distractors(correct_id: int, view_folder: str, root: str, k: int = 3) -> list:
#     """在同一 view 文件夹下随机挑选 k 张干扰图，避免与正确答案重复"""
#     paths = set()
#     while len(paths) < k:
#         rid = random.randint(1, TOTAL_IMAGES)
#         if rid != correct_id:
#             paths.add(add_path(view_folder, rid, root))
#     return list(paths)

# def build_task(img_id: int, root: str) -> dict:
#     """生成单条 QA 任务（messages+images）"""
#     # 为该 id 生成三视图路径
#     paths = {name: add_path(folder, img_id, root) for name, folder in VIEW_DICT.items()}

#     # 随机决定哪两张给模型，哪一张做答案
#     view_a, view_b, view_c = random.sample(VIEWS, 3)
#     name_a, folder_a = view_a
#     name_b, folder_b = view_b
#     name_c, folder_c = view_c

#     # 路径
#     path_a = paths[name_a]
#     path_b = paths[name_b]
#     correct_path = paths[name_c]

#     # 生成同一视角下的 3 个干扰路径
#     distractor_paths = distractors(img_id, folder_c, root, 3)

#     # 4 个选项：正确 + 3 干扰，顺序随机
#     option_paths = [correct_path] + distractor_paths
#     random.shuffle(option_paths)

#     # 把 ABCD 标号与 option 绑定
#     abcd = ["A", "B", "C", "D"]
#     correct_letter = abcd[option_paths.index(correct_path)]

#     # messages
#     question = (
#         f"Given the {name_a} <image> and {name_b} <image> , "
#         f"what is the {name_c} ?\n" +
#         "\n".join(f"{abcd[i]}. <image>" for i in range(4))
#     )
#     messages = [
#         {"role": "user",      "content": question},
#         {"role": "assistant", "content": correct_letter}
#     ]

#     # images：2 张已知视图 + 4 个选项，顺序固定
#     images = [path_a, path_b] + option_paths      # 共 6 条

#     return {"messages": messages, "images": images}

# def generate_jsonl(root_folder: str, out_path: str):
#     with open(out_path, "w", encoding="utf-8") as fout:
#         with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
#             pbar = tqdm(total=TARGET_SAMPLES, desc="Generating")
#             generated = 0
#             while generated < TARGET_SAMPLES:
#                 # 批量随机 id（带放回抽样）
#                 batch_ids = [random.randint(1, TOTAL_IMAGES)
#                              for _ in range(min(10_000, TARGET_SAMPLES - generated))]
#                 for task in pool.map(lambda i: build_task(i, root_folder), batch_ids):
#                     fout.write(json.dumps(task, ensure_ascii=False) + "\n")
#                 generated += len(batch_ids)
#                 pbar.update(len(batch_ids))
#             pbar.close()

# # ---------- 调用 ----------
# generate_jsonl(
#     root_folder="/home/chenzhuofan/czf/tuzhi_extract/data/train_3views/img_files",          # 含 view1/view2/view3 的根目录
#     out_path   ="/home/chenzhuofan/czf/tuzhi_extract/data/jsonl/view_matching_test.jsonl"
# )

# import json, os, random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ---------- 可调参数 ----------
TOTAL_IMAGES   = 20000          # 数据集中实际图片数量 (id: 1~20000)
TARGET_SAMPLES = 50            # 需要生成的 QA 数
MAX_WORKERS    = 8              # 线程数（按机器调整）
# --------------------------------

VIEW_DICT = {                     # 视角到子文件夹名映射
    "Top view":   "view1",    # 俯视
    "Front view": "view2",    # 主视
    "Side view":  "view3"     # 侧视
}
VIEWS = list(VIEW_DICT.items())   # [(name, folder), ...]

# 生成选项标记：A 到 Q，共17个
LETTERS = [chr(ord('A') + i) for i in range(17)]


def add_path(view_folder: str, img_id: int, root: str) -> str:
    """拼接完整路径"""
    return os.path.join(root, view_folder, f"{img_id}.png")


def distractors(correct_id: int, view_folder: str, root: str, k: int = 16) -> list:
    """在同一 view 文件夹下随机挑选 k 张干扰图，避免与正确答案重复"""
    paths = set()
    while len(paths) < k:
        rid = random.randint(1, TOTAL_IMAGES)
        if rid != correct_id:
            paths.add(add_path(view_folder, rid, root))
    return list(paths)


def build_task(img_id: int, root: str) -> dict:
    """生成单条 QA 任务（messages+images），并返回 17 个选项"""
    # 为该 id 生成三视图路径
    paths = {name: add_path(folder, img_id, root) for name, folder in VIEW_DICT.items()}

    # 随机决定哪两张给模型，哪一张做答案
    view_a, view_b, view_c = random.sample(VIEWS, 3)
    name_a, folder_a = view_a
    name_b, folder_b = view_b
    name_c, folder_c = view_c

    # 正确答案路径
    correct_path = paths[name_c]
    # 生成 16 张干扰
    distractor_paths = distractors(img_id, folder_c, root, k=16)

    # 合并选项并随机打乱
    option_paths = [correct_path] + distractor_paths
    random.shuffle(option_paths)

    # 找到正确选项对应的标记
    correct_index = option_paths.index(correct_path)
    correct_letter = LETTERS[correct_index]

    # 构建问题文本，列出 17 个选项
    question = (
        f"Given the {name_a} <image> and {name_b} <image>, what is the {name_c}?\n" +
        "\n".join(f"{LETTERS[i]}. <image>" for i in range(len(LETTERS)))
    )
    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": correct_letter}
    ]

    # images：2 张已知视图 + 17 个选项，顺序固定
    images = [paths[name_a], paths[name_b]] + option_paths

    return {"messages": messages, "images": images}


def generate_jsonl(root_folder: str, out_path: str):
    with open(out_path, "w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            pbar = tqdm(total=TARGET_SAMPLES, desc="Generating")
            generated = 0
            while generated < TARGET_SAMPLES:
                batch_size = min(10000, TARGET_SAMPLES - generated)
                batch_ids = [random.randint(1, TOTAL_IMAGES) for _ in range(batch_size)]
                for task in pool.map(lambda i: build_task(i, root_folder), batch_ids):
                    fout.write(json.dumps(task, ensure_ascii=False) + "\n")
                generated += batch_size
                pbar.update(batch_size)
            pbar.close()

# ---------- 调用 ----------
generate_jsonl(
    root_folder="/home/chenzhuofan/czf/tuzhi_extract/data/train_3views/img_files",
    out_path  ="/home/chenzhuofan/czf/tuzhi_extract/data/jsonl/view_matching_test_17opts.jsonl"
)
