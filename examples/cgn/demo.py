"""
Demo: 输入图片 → OCR 识别 → 读取 schema / examples JSON → 调用 LangExtract → 输出抽取结果。

流程：
  1. OCR 模块：对输入图片做文字识别，得到文本。
  2. 配置模块：从 schema.json 读取 prompt_description，从 examples.json 读取 few-shot 示例。
  3. 抽取：用 lx.extract(text, prompt_description, examples, model) 得到结构化结果并输出。

使用前请设置环境变量 LANGEXTRACT_API_KEY 或在本文件中配置 base_url / api_key（勿提交密钥）。
OCR 可选依赖：pip install pytesseract Pillow，并安装 Tesseract。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import langextract as lx
from langextract.providers.openai import OpenAILanguageModel

# 保证无论从项目根还是 examples 目录运行，都能加载同目录下的 ocr_module
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# -----------------------------------------------------------------------------
# 配置（建议 API Key 用环境变量）
# -----------------------------------------------------------------------------
BASE_URL = os.environ.get("LANGEXTRACT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.environ.get("LANGEXTRACT_API_KEY", "")
MODEL_NAME = os.environ.get("LANGEXTRACT_MODEL", "deepseek-v3.2")

# 默认路径（以本文件所在目录为基准）
_EXAMPLES_DIR = _THIS_DIR
DEFAULT_SCHEMA_PATH = _EXAMPLES_DIR / "schema.json"
DEFAULT_EXAMPLES_PATH = _EXAMPLES_DIR / "examples.json"


# -----------------------------------------------------------------------------
# 1. OCR 模块：图片 → 文本
# -----------------------------------------------------------------------------
def run_ocr(image_path: str | Path) -> str:
    """对图片做 OCR，返回识别出的文本。"""
    from ocr_module import ocr_image  # noqa: E402

    return ocr_image(str(image_path))


# -----------------------------------------------------------------------------
# 2. 读取 schema JSON → prompt_description
# -----------------------------------------------------------------------------
def load_schema(schema_path: str | Path) -> str:
    """从 schema JSON 文件加载 prompt_description。

    JSON 格式支持：
      - {"prompt_description": "描述字符串"}
      - {"description": "描述字符串"}
      - 纯字符串文件则整段作为 prompt_description
    """
    path = Path(schema_path)
    if not path.is_file():
        raise FileNotFoundError(f"schema 文件不存在: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("schema 文件为空")
    # 尝试按 JSON 解析
    try:
        data = json.loads(raw)
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return data.get("prompt_description") or data.get("description") or json.dumps(data, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw


# -----------------------------------------------------------------------------
# 3. 读取 examples JSON → List[ExampleData]
# -----------------------------------------------------------------------------
def load_examples(examples_path: str | Path) -> list[lx.data.ExampleData]:
    """从 examples JSON 文件加载 few-shot 示例。

    JSON 格式：数组，每项为
      {"text": "原文", "extractions": [{"extraction_class": "类名", "extraction_text": "抽取文本"}]}
    """
    path = Path(examples_path)
    if not path.is_file():
        raise FileNotFoundError(f"examples 文件不存在: {path}")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        data = [data]
    result = []
    for item in data:
        text = item.get("text", "")
        extractions_raw = item.get("extractions") or []
        extractions = [
            lx.data.Extraction(
                extraction_class=e.get("extraction_class", ""),
                extraction_text=e.get("extraction_text", ""),
            )
            for e in extractions_raw
        ]
        result.append(lx.data.ExampleData(text=text, extractions=extractions))
    return result


# -----------------------------------------------------------------------------
# 4. 调用 LangExtract 并输出结果
# -----------------------------------------------------------------------------
def run_extraction(
    text: str,
    prompt_description: str,
    examples: list[lx.data.ExampleData],
    model: OpenAILanguageModel | None = None,
):
    """使用 lx.extract 做信息抽取，返回 AnnotatedDocument（或列表）。"""
    if model is None:
        if not API_KEY:
            raise ValueError("请设置 LANGEXTRACT_API_KEY 或在代码中配置 API_KEY")
        model = OpenAILanguageModel(
            model_id=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
        )
    return lx.extract(
        text_or_documents=text,
        prompt_description=prompt_description,
        examples=examples,
        model=model,
    )


def main(
    image_path: str | Path | None = None,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    examples_path: str | Path = DEFAULT_EXAMPLES_PATH,
    *,
    use_ocr: bool = True,
    input_text: str | None = None,
) -> None:
    """
    主流程：图片（或直接文本）→ OCR（可选）→ 读 schema/examples → 调用 lx → 打印结果。

    Args:
        image_path: 输入图片路径；与 input_text 二选一。
        schema_path: schema JSON 路径。
        examples_path: examples JSON 路径。
        use_ocr: 当提供 image_path 时是否执行 OCR；False 时需同时提供 input_text。
        input_text: 若不想用图片，可直接传入文本，此时忽略 image_path 与 OCR。
    """
    # 确定待抽取文本
    if input_text is not None and input_text.strip():
        text = input_text.strip()
        print("[Demo] 使用直接输入的文本，跳过 OCR。")
    elif image_path and Path(image_path).is_file():
        if not use_ocr:
            raise ValueError("未启用 OCR 且未提供 input_text，无法得到文本。")
        print(f"[Demo] 对图片做 OCR: {image_path}")
        text = run_ocr(image_path)
        print(f"[Demo] OCR 结果（前 200 字）:\n{text[:200]}...\n" if len(text) > 200 else f"[Demo] OCR 结果:\n{text}\n")
    else:
        # 无图片且无输入文本时，用示例句子便于直接跑通
        text = "東京出身の田中さんはGoogleで働いています。"
        print("[Demo] 未提供图片或 input_text，使用示例句子。")

    # 加载 schema 与 examples
    prompt_description = load_schema(schema_path)
    examples = load_examples(examples_path)
    print(f"[Demo] 已加载 schema 与 {len(examples)} 条 examples。")

    # 调用 lx 并输出
    result = run_extraction(text, prompt_description, examples)
    print("\n[Demo] 抽取结果:")
    print(result)


if __name__ == "__main__":
    # 可直接运行：用示例句子（不依赖图片与 OCR）
    # 若需从图片跑通：main(image_path="path/to/your/image.png")
    # 若已有文本：main(input_text="你的文本...")
    main()
