import json

input_file = '/Users/asri/.claude/projects/-Users-asri-Projects-friday-file-explorer/118a32b5-5344-4c15-a87a-143bd2c96e97.jsonl'
output_file = '/Users/asri/Downloads/transcript_new.txt'


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get('type')
            if btype == 'text':
                parts.append(block.get('text', ''))
            elif btype == 'thinking':
                parts.append(f"[thinking] {block.get('thinking', '')}")
            elif btype == 'tool_use':
                name = block.get('name', '')
                inp = json.dumps(block.get('input', {}), ensure_ascii=False)
                parts.append(f"[tool_use: {name}] {inp}")
            elif btype == 'tool_result':
                parts.append(f"[tool_result] {extract_text(block.get('content', ''))}")
        return '\n'.join(p for p in parts if p)
    return ''


with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg = data.get('message')
        if not isinstance(msg, dict):
            continue

        role = msg.get('role', data.get('type', 'system')).upper()
        text = extract_text(msg.get('content', ''))
        if not text:
            continue

        f_out.write(f"--- {role} ---\n{text}\n\n")

print(f"Done! Saved to {output_file}")
