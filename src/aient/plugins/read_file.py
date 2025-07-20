import os
import json
import chardet
from pdfminer.high_level import extract_text

from .registry import register_tool

# 读取文件内容
@register_tool()
def read_file(file_path):
    """
Description: Request to read the contents of a file at the specified path. Use this when you need to examine the contents of an existing file you do not know the contents of, for example to analyze code, review text files, or extract information from configuration files. Automatically extracts raw text from PDF and DOCX files. May not be suitable for other types of binary files, as it returns the raw content as a string.

注意：
1. pdf 文件 必须使用 read_file 读取，可以使用 read_file 直接读取 PDF。

参数:
    file_path: 要读取的文件路径，(required) The path of the file to read (relative to the current working directory)

返回:
    文件内容的字符串

Usage:
<read_file>
<file_path>File path here</file_path>
</read_file>

Examples:

1. Reading an entire file:
<read_file>
<file_path>frontend.pdf</file_path>
</read_file>

2. Reading multiple files:

<read_file>
<file_path>frontend-config.json</file_path>
</read_file>

<read_file>
<file_path>backend-config.txt</file_path>
</read_file>

...

<read_file>
<file_path>README.md</file_path>
</read_file>
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"<tool_error>文件 '{file_path}' 不存在</tool_error>"

        # 检查是否为文件
        if not os.path.isfile(file_path):
            return f"<tool_error>'{file_path}' 不是一个文件</tool_error>"

        # 检查文件扩展名
        if file_path.lower().endswith('.pdf'):
            # 提取PDF文本
            text_content = extract_text(file_path)

            # 如果提取结果为空
            if not text_content:
                return f"<tool_error>无法从 '{file_path}' 提取文本内容</tool_error>"
        elif file_path.lower().endswith('.ipynb'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    notebook_content = json.load(file)

                for cell in notebook_content.get('cells', []):
                    if cell.get('cell_type') == 'code' and 'outputs' in cell:
                        filtered_outputs = []
                        for output in cell.get('outputs', []):
                            new_output = output.copy()
                            if 'data' in new_output:
                                original_data = new_output['data']
                                filtered_data = {}
                                for key, value in original_data.items():
                                    if key.startswith('image/'):
                                        continue
                                    if key == 'text/html':
                                        html_content = "".join(value) if isinstance(value, list) else value
                                        if isinstance(html_content, str) and '<table class="show_videos"' in html_content:
                                            continue
                                    filtered_data[key] = value
                                if filtered_data:
                                    new_output['data'] = filtered_data
                                    filtered_outputs.append(new_output)
                            elif 'output_type' in new_output and new_output['output_type'] in ['stream', 'error']:
                                filtered_outputs.append(new_output)

                        cell['outputs'] = filtered_outputs

                text_content = json.dumps(notebook_content, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                return f"<tool_error>文件 '{file_path}' 不是有效的JSON格式 (IPython Notebook)。</tool_error>"
            except Exception as e:
                return f"<tool_error>处理IPython Notebook文件 '{file_path}' 时发生错误: {e}</tool_error>"
        else:
            # 更新：修改通用文件读取逻辑以支持多种编码
            # 这部分替换了原有的 else 块内容
            try:
                with open(file_path, 'rb') as file: # 以二进制模式读取
                    raw_data = file.read()

                if not raw_data: # 处理空文件
                    text_content = ""
                else:
                    detected_info = chardet.detect(raw_data)
                    primary_encoding_to_try = detected_info['encoding']
                    confidence = detected_info['confidence']

                    decoded_successfully = False

                    # 尝试1: 使用检测到的编码 (如果置信度高且编码有效)
                    if primary_encoding_to_try and confidence > 0.7: # 您可以根据需要调整置信度阈值
                        try:
                            text_content = raw_data.decode(primary_encoding_to_try)
                            decoded_successfully = True
                        except (UnicodeDecodeError, LookupError): # LookupError 用于处理无效的编码名称
                            # 解码失败，将尝试后备编码
                            pass

                    # 尝试2: UTF-8 (如果第一次尝试失败或未进行)
                    if not decoded_successfully:
                        try:
                            text_content = raw_data.decode('utf-8')
                            decoded_successfully = True
                        except UnicodeDecodeError:
                            # 解码失败，将尝试下一个后备编码
                            pass

                    # 尝试3: UTF-16 (如果之前的尝试都失败)
                    # 'utf-16' 会处理带BOM的LE/BE编码。若无BOM，则假定为本机字节序。
                    # chardet 通常能更准确地检测具体的 utf-16le 或 utf-16be。
                    if not decoded_successfully:
                        try:
                            text_content = raw_data.decode('utf-16')
                            decoded_successfully = True
                        except UnicodeDecodeError:
                            # 所有主要尝试都失败
                            pass

                    if not decoded_successfully:
                        # 所有尝试均失败后的错误信息
                        detected_str_part = ""
                        if primary_encoding_to_try and confidence > 0.7: # 如果有高置信度的检测结果
                            detected_str_part = f"检测到的编码 '{primary_encoding_to_try}' (置信度 {confidence:.2f}), "
                        elif primary_encoding_to_try: # 如果有检测结果但置信度低
                            detected_str_part = f"低置信度检测编码 '{primary_encoding_to_try}' (置信度 {confidence:.2f}), "

                        return f"<tool_error>文件 '{file_path}' 无法解码。已尝试: {detected_str_part}UTF-8, UTF-16。</tool_error>"

            except FileNotFoundError:
                # 此处不太可能触发 FileNotFoundError，因为函数开头已有 os.path.exists 检查
                return f"<tool_error>文件 '{file_path}' 在读取过程中未找到。</tool_error>"
            except Exception as e:
                # 捕获在此块中可能发生的其他错误，例如未被早期检查捕获的文件读取问题
                return f"<tool_error>处理通用文件 '{file_path}' 时发生错误: {e}</tool_error>"

        if file_path.lower().endswith('.csv'):
            lines = text_content.splitlines(True)
            if len(lines) > 500:
                top_lines = lines[:250]
                bottom_lines = lines[-250:]
                omitted_count = len(lines) - 500
                text_content = "".join(top_lines) + f"\n... (中间省略了 {omitted_count} 行) ...\n" + "".join(bottom_lines)

        # 返回文件内容
        return text_content

    except PermissionError:
        return f"<tool_error>没有权限访问文件 '{file_path}'</tool_error>"
    except UnicodeDecodeError:
        # 更新：修改全局 UnicodeDecodeError 错误信息使其更通用
        return f"<tool_error>文件 '{file_path}' 包含无法解码的字符 (UnicodeDecodeError)。</tool_error>"
    except Exception as e:
        return f"<tool_error>读取文件时发生错误: {e}</tool_error>"

if __name__ == "__main__":
    # python -m beswarm.aient.src.aient.plugins.read_file
    result = read_file("./work/cax/Lenia Notebook.ipynb")
    print(result)
    print(len(result))
