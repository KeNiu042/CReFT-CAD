import json, os, concurrent.futures
from tqdm import tqdm

# ---------- 路径 ----------
INPUT_JSON   = '/home/chenzhuofan/czf/tuzhi_extract/data/train_d/metadata.json'
IMG_DIR      = '/home/chenzhuofan/czf/tuzhi_extract/data/train_d/img_files'
OUTPUT_JSONL = '/home/chenzhuofan/czf/tuzhi_extract/data/jsonl/d_test.jsonl'
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
    "values of the following parameters in this exact order, separated by commas\n" +
    "\n".join(f"{i+1}. {name}" for i, (name, _) in enumerate(ORDERED_FIELDS))
)

def build_record(item):
    img_path = os.path.join(IMG_DIR, f"{item['id']}.png")
    values = [str(item["params"][field_cn]) for _, field_cn in ORDERED_FIELDS]
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
