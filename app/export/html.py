"""HTML 导出：Jinja2 模板，嵌入原图路径，支持 bbox 标注。"""
from pathlib import Path

from jinja2 import Environment, BaseLoader

from app.export.base import ExporterBase
from app.models import OcrProject

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{{ project.name }}</title>
<style>
  body { font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif;
         background:#1e1e1e; color:#d4d4d4; margin:40px; }
  h1   { color:#9cdcfe; }
  .page-header { color:#569cd6; border-bottom:1px solid #3c3c3c;
                 padding:8px 0; margin-top:32px; }
  .block  { margin:12px 0 12px 16px; }
  .line   { margin:2px 0; line-height:1.8; }
  .flagged { color:#f44336; }
  .modified{ color:#ffa726; }
  .ok      { color:#4caf50; }
  .meta   { font-size:11px; color:#666; margin-left:8px; }
</style>
</head>
<body>
<h1>{{ project.name }}</h1>
<p class="meta">共 {{ project.page_count }} 页，{{ project.total_lines }} 行</p>
{% for page in project.pages %}
<div class="page-header">第 {{ page.page_number }} 页
  <span class="meta">{{ page.image_path }}</span></div>
{% for block in page.text_blocks %}
<div class="block" data-type="{{ block.block_type.value }}"
     data-bbox="{{ block.bbox.x }},{{ block.bbox.y }},{{ block.bbox.w }},{{ block.bbox.h }}">
{% for line in block.lines %}
  <div class="line {% if line.proof_status.value == 'auto_flagged' %}flagged
    {% elif line.proof_status.value == 'modified' %}modified
    {% elif line.proof_status.value == 'ok' %}ok{% endif %}"
       data-conf="{{ '%.2f'|format(line.confidence) }}"
       data-bbox="{{ line.bbox.x }},{{ line.bbox.y }},{{ line.bbox.w }},{{ line.bbox.h }}">
    {{ line.text }}
    <span class="meta">{{ '%.0f'|format(line.confidence*100) }}%</span>
  </div>
{% endfor %}
</div>
{% endfor %}
{% endfor %}
</body>
</html>
"""


class HtmlExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        env = Environment(loader=BaseLoader())
        tmpl = env.from_string(_TEMPLATE)
        html = tmpl.render(project=project)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
