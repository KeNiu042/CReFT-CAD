import json
import os
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------- 路径配置 ----------
INPUT_JSON   = '/home/chenzhuofan/czf/tuzhi_extract/data/train2/metadata.json'
IMG_DIR      = '/home/chenzhuofan/czf/tuzhi_extract/data/train2/img_files'
OUTPUT_JSONL = '/home/chenzhuofan/czf/tuzhi_extract/data/sft/judge.jsonl'
# -----------------------------

# 字段映射
FIELD_MAP = {
    "Cap Beam Cross-Bridge Dimension":     "rect_width",
    "Cap Beam Along-Bridge Dimension":     "rect_height",
    "Cross-Bridge Pier Column Count":      "rounded_rect_horizontal_count",
    "Cross-Bridge Pier Spacing":           "rounded_rect_horizontal_distance",
    "Pier Column Cross-Bridge Dimension":  "rounded_rect_width",
    "Pier Column Along-Bridge Dimension":  "rounded_rect_height",
    "Chamfer Radius":                      "rounded_rect_radius",
    "Cross-Bridge Pile Base Count":        "circle_horizontal_count",
    "Along-Bridge Pile Base Count":        "circle_vertical_count",
    "Cross-Bridge Pile Spacing":           "circle_horizontal_distance",
    "Along-Bridge Pile Spacing":           "circle_vertical_distance",
    "Pile Base Radius":                    "circle_radius",
    "Pier Column Height":                  "dunzhu_height",
    "Cap Beam Height":                     "chengtai_height",
    "Pile Base Height":                    "zhuangji_height"
}

FIELD_NAMES = list(FIELD_MAP.keys())
COUNT_FIELDS = {
    "rounded_rect_horizontal_count",
    "circle_horizontal_count",
    "circle_vertical_count"
}

# 构造错误参数：从同一样本中其他字段中采样替换
def get_modified_params(true_params):
    keys = list(true_params.keys())
    max_changes = min(7, len(keys) - 1)
    k = random.randint(1, max_changes)
    keys_to_modify = random.sample(keys, k)

    new_params = true_params.copy()
    for key in keys_to_modify:
        current_value = true_params[key]
        # 候选值为当前样本中其他字段值且不等于当前字段值
        candidates = [v for k2, v in true_params.items() if k2 != key and v != current_value]
        if candidates:
            new_params[key] = random.choice(candidates)
    return new_params

# 构造单个样本
def build_wrapper(item):
    raw_params = item["params"]
    true_params = {k: raw_params[FIELD_MAP[k]] for k in FIELD_NAMES}
    is_correct = random.random() < 0.5

    check_params = true_params if is_correct else get_modified_params(true_params)
    label = "Yes" if is_correct else "No"
    img_path = os.path.join(IMG_DIR, f"{item['id']}.png")

    # 构造字段描述
    field_desc = "\n".join(f"{i+1}. {name}" for i, name in enumerate(FIELD_NAMES)) + "."
    param_values = ", ".join(str(check_params[name]) for name in FIELD_NAMES)

    prompt = (
        "Given the front, top, and side views <image>,\n"
        "please refer to the required field names below, and determine whether the given parameter values match the image.\n\n"
        "Required field names:\n" + field_desc + "\n\n" +
        "Parameter values:\n" + param_values + "\n\n" +
        "Please provide your final answer (Yes or No)."
    )

    return {
        "images": [img_path],
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": label}
                    ],
        "para": list(true_params.values()),  # 仅保留键
    }

def main():
    with open(INPUT_JSON, encoding='utf-8') as f:
        data = json.load(f)

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(build_wrapper, data), total=len(data), desc="构造判断题样本"))

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
