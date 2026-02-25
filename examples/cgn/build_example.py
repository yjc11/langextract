from pathlib import Path
import json

from demo import load_material_list_gt, load_ocr_json

ocr_json_dir = Path(r"E:\Vanke\xxxx\eval_dataset\EVAL_B1\X_0_ocr_with_layout_cls")
gt_excel_dir = Path(r'E:\Vanke\xxxx\eval_dataset\EVAL_B1\8_kv_xlsx\标准化')
ocr_json_paths = list(ocr_json_dir.glob("*.json"))[:1]
examples = list()
for ocr_json_path in ocr_json_paths:
    page_texts, page_bboxes, layout_class = load_ocr_json(ocr_json_path)
    if layout_class != '参数表':
        continue

    material_list_gt = load_material_list_gt(gt_excel_dir / f"{ocr_json_path.stem}.xlsx")
    if material_list_gt is None:
        continue
    extractions = list()
    for key, value in material_list_gt.items():
        for v in value:
            extractions.append({
                "extraction_class": key,
                "extraction_text": v,
            })
    examples.append({
        "text": page_texts,
        "extractions": extractions,
    })

with open("examples.json", "w") as f:
    json.dump(examples, f, ensure_ascii=False, indent=4)