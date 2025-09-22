import json
import os
import random
import concurrent.futures
from tqdm import tqdm  # 导入tqdm库以显示进度条

# Generate distractors for the multiple-choice question
def generate_distractors(correct_answer, num_distractors=3):
    candidates = set()
    while len(candidates) < num_distractors:
        val = random.randint(max(0, correct_answer - 3), correct_answer + 3)
        if val != correct_answer:
            candidates.add(val)
    return list(candidates)

# Construct image path based on id and directory
def add_absolute_path(id_, image_dir):
    return os.path.join(image_dir, f"{id_}.png")

# Generate the task with questions and answers
def generate_task(data_item, image_dir):
    id_ = data_item["id"]
    p = data_item["params"]
    image_path = add_absolute_path(id_, image_dir)

    # Calculate the answers
    circle_answer = p["circle_horizontal_count"] * p["circle_vertical_count"]
    rect_answer = p["rounded_rect_horizontal_count"]

    tasks = []

    # Create questions and options for "Pile Base" (桩基) and "Pier Column" (墩柱)
    for obj_type, answer, question_template in [
        ("Pile Base", circle_answer, "Given the front, top, and side views<image>, what is the quantity of pile bases?"),
        ("Pier Column", rect_answer, "Given the front, top, and side views<image>, what is the quantity of pier columns?")
    ]:
        # Generate distractors (incorrect options)
        distractors = generate_distractors(answer)
        all_options = [answer] + distractors
        all_options = list(set(all_options))[:4]
        random.shuffle(all_options)

        abcd = ["A", "B", "C", "D"]
        correct_index = all_options.index(answer)
        options_text = "\n" + "\n".join([f"{abcd[i]}. {all_options[i]}" for i in range(len(all_options))])
        
        # Formulate the question and answer
        messages = [
            {
                "role": "user",
                "content": question_template + options_text
            },
            {
                "role": "assistant",
                "content": abcd[correct_index]  # Answer is given as A/B/C/D
            }
        ]
        tasks.append({
            "messages": messages,
            "images": [image_path]
        })

    return tasks

# Read data from input JSON and generate the output JSONL file
def generate_jsonl_from_file(input_json_path, image_dir, output_jsonl_path):
    with open(input_json_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        # Add tqdm progress bar to the executor to track progress
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Wrap the data in tqdm to track progress
            results = executor.map(lambda x: generate_task(x, image_dir), tqdm(data, desc="Processing tasks"))
            for result in results:
                for task in result:
                    outfile.write(json.dumps(task, ensure_ascii=False) + "\n")

# Example of how to call the function with paths
generate_jsonl_from_file(
    '/home/chenzhuofan/czf/tuzhi_extract/data/test/metadata.json', 
    "/home/chenzhuofan/czf/tuzhi_extract/data/test/img_files", 
    "/home/chenzhuofan/czf/tuzhi_extract/data/jsonl/component_count_test.jsonl"
)
