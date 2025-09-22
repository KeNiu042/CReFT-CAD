import json
import os
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------- 路径配置 ----------
INPUT_JSON   = '/home/chenzhuofan/czf/tuzhi_extract/data/train2/metadata.json'
IMG_DIR      = '/home/chenzhuofan/czf/tuzhi_extract/data/train2/img_files'
OUTPUT_JSONL = '/home/chenzhuofan/czf/tuzhi_extract/data/sft/choice.jsonl'
# -----------------------------

# 字段配置
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

# 随机替换当前字段为本行其他字段值
def get_modified_params(true_params):
    keys = list(true_params.keys())
    k = random.randint(1, min(7, len(keys) - 1))
    to_modify = random.sample(keys, k)
    new_params = true_params.copy()
    for key in to_modify:
        value = true_params[key]
        candidates = [v for k2, v in true_params.items() if k2 != key and v != value]
        if candidates:
            new_params[key] = random.choice(candidates)
    return new_params

# 随机遮盖当前字段（原始）
def apply_mask(params):
    indices = list(range(len(FIELD_NAMES)))
    num = random.randint(0, min(7, len(indices)))
    mask_indices = set(random.sample(indices, num))
    return ", ".join("<mask>" if i in mask_indices else str(params[name]) for i, name in enumerate(FIELD_NAMES))

# 错误选项中仅遮掉与原值一致的字段（防止遮掉错误字段使其变对）
def apply_mask_safe(true_params, candidate_params):
    maskable = [i for i, name in enumerate(FIELD_NAMES) if str(true_params[name]) == str(candidate_params[name])]
    num = random.randint(0, min(7, len(maskable)))
    mask_indices = set(random.sample(maskable, num))
    return ", ".join("<mask>" if i in mask_indices else str(candidate_params[name]) for i, name in enumerate(FIELD_NAMES))

# 构造一个选择题样本
def build_choice_item(item):
    raw_params = item["params"]
    true_params = {k: raw_params[FIELD_MAP[k]] for k in FIELD_NAMES}
    img_path = os.path.join(IMG_DIR, f"{item['id']}.png")

    # 正确选项数量：1~2个
    num_correct = random.choice([1, 2])
    correct_opts = [apply_mask(true_params) for _ in range(num_correct)]

    wrong_opts = []
    for _ in range(4 - num_correct):
        wrong_p = get_modified_params(true_params)
        wrong_opts.append(apply_mask_safe(true_params, wrong_p))

    all_opts = correct_opts + wrong_opts
    random.shuffle(all_opts)

    prefix = ['A', 'B', 'C', 'D']
    answer_label = "".join(prefix[i] for i, opt in enumerate(all_opts) if opt in correct_opts)
    option_text = "\n".join(f"{p}. {v}" for p, v in zip(prefix, all_opts))

    field_desc = "\n".join(f"{i+1}. {name}" for i, name in enumerate(FIELD_NAMES)) + "."

    prompt = (
        "Given the front, top, and side views <image>,\n"
        "please refer to the required field names below, and choose all options that match the image.\n"
        "Some parameter values are replaced with <mask>.\n\n"
        "Required field names:\n" + field_desc + "\n\n" +
        "Please carefully compare the options with the image and select the correct ones based on the visible structures.\n\n"
        "Options:\n" + option_text
    )

    return {
        "images": [img_path],
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": answer_label}],
        "para": list(true_params.values()),  # 仅保留键
    }

def main():
    with open(INPUT_JSON, encoding='utf-8') as f:
        data = json.load(f)

    with Pool(processes=cpu_count()) as pool:
        output = list(tqdm(pool.imap(build_choice_item, data), total=len(data), desc="构造选择题样本"))

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in output:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ 所有选择题样本已保存到 {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
