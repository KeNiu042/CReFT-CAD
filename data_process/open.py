import json, os, concurrent.futures
from tqdm import tqdm

# ---------- 路径 ----------
INPUT_JSON   = '/home/chenzhuofan/czf/tuzhi_extract/data/test2/metadata.json'
IMG_DIR      = '/home/chenzhuofan/czf/tuzhi_extract/data/test2/img_files'
OUTPUT_JSONL = '/home/chenzhuofan/czf/tuzhi_extract/data/sft/open_test2.jsonl'
# -------------------------

# 有序字段列表（顺序即输出顺序）
ORDERED_FIELDS = [
    ("Cap Beam Cross-Bridge Dimension",    "rect_width"),
    ("Cap Beam Along-Bridge Dimension",    "rect_height"),
    ("Cross-Bridge Pier Column Count",     "rounded_rect_horizontal_count"),
    ("Cross-Bridge Pier Spacing",          "rounded_rect_horizontal_distance"),
    ("Pier Column Cross-Bridge Dimension", "rounded_rect_width"),
    ("Pier Column Along-Bridge Dimension", "rounded_rect_height"),
    ("Chamfer Radius",                     "rounded_rect_radius"),
    ("Cross-Bridge Pile Base Count",       "circle_horizontal_count"),
    ("Along-Bridge Pile Base Count",       "circle_vertical_count"),
    ("Cross-Bridge Pile Spacing",          "circle_horizontal_distance"),
    ("Along-Bridge Pile Spacing",          "circle_vertical_distance"),
    ("Pile Base Radius",                   "circle_radius"),
    ("Pier Column Height",                 "dunzhu_height"),
    ("Cap Beam Height",                    "chengtai_height"),
    ("Pile Base Height",                   "zhuangji_height")
]

PROMPT = (
    "Given the front, top, and side views<image>, please output only the numeric "
    "values of the following parameters in this exact order, separated by commas.\n\n"
    "SPECIAL RULES:\n"
    "• If **Cross-Bridge Pier Column Count = 1**, set **Cross-Bridge Pier Spacing = 0**\n"
    "• If **Cross-Bridge Pile Base Count  = 1**, set **Cross-Bridge Pile Spacing = 0**\n\n" +
    "\n".join(f"{i+1}. {name}" for i, (name, _) in enumerate(ORDERED_FIELDS))
)

def clean_number(x):
    return int(x) if x == int(x) else round(x, 1)

def compute_expression(field_key, params):
    def fmt(val):
        val = clean_number(val)
        return str(val)

    if field_key == "rect_width":
        total = params["rect_width"]
        B = params["rounded_rect_horizontal_count"]
        C = params["rounded_rect_width"]
        D = params["rounded_rect_horizontal_distance"] - C
        if B == 1:
            A = (total - B * C) / 2
            return f"2*{fmt(A)}+{fmt(C)}"
        else:
            A = (total - B * C - (B - 1) * D) / 2
            return f"2*{fmt(A)}+{fmt(B)}*{fmt(C)}+{fmt(D)}"

    elif field_key == "rect_height":
        total = params["rect_height"]
        B = params["rounded_rect_height"]
        A = (total - B) / 2
        return f"2*{fmt(A)}+{fmt(B)}"

    elif field_key == "rounded_rect_horizontal_distance":
        num = params["rounded_rect_horizontal_count"]
        if num > 1:
            total = params["rounded_rect_horizontal_distance"]
            B = params["rounded_rect_width"]
            C = total - B
            return f"{fmt(B)}+{fmt(C)}"
        else:
            return '0'

    elif field_key == "circle_horizontal_distance":
        total = params["circle_horizontal_distance"]
        radius = params["circle_radius"]
        B = radius * 2
        C = total - B
        return f"{fmt(B)}+{fmt(C)}"

    elif field_key == "circle_vertical_distance":
        total = params["circle_vertical_distance"]
        radius = params["circle_radius"]
        B = radius * 2
        C = total - B
        return f"{fmt(B)}+{fmt(C)}"

    elif field_key == "circle_radius":
        radius = params["circle_radius"]
        B = radius * 2
        return f"{fmt(B)}/2"

    # 默认字段返回原值
    return str(params[field_key])

def build_record(item):
    img_path = os.path.join(IMG_DIR, f"{item['id']}.png")
    p = item["params"].copy()

    # 应用特殊规则
    if p.get("rounded_rect_horizontal_count", 0) == 1:
        p["rounded_rect_horizontal_distance"] = 0
    if p.get("circle_horizontal_count", 0) == 1:
        p["circle_horizontal_distance"] = 0

    values = [compute_expression(field_cn, p) for _, field_cn in ORDERED_FIELDS]
    assistant_content = ", ".join(values)

    return {
        "messages": [
            {"role": "user",      "content": PROMPT},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": [img_path]
    }

def main():
    with open(INPUT_JSON, encoding='utf-8') as f:
        data = json.load(f)

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            for rec in tqdm(ex.map(build_record, data), total=len(data), desc="Building value-seq dataset"):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
