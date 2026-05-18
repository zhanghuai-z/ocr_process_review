"""HTML 导出：Jinja2 模板，嵌入原图路径，支持 bbox 标注。"""
from pathlib import Path

from jinja2 import Environment, BaseLoader

from app.export.base import ExporterBase
from app.models import OcrProject
from app.services.export_service import (
    format_bbox,
    get_block_label,
    get_block_style,
    get_export_text,
    iter_export_blocks,
    iter_export_lines,
    iter_export_pages,
)

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
  .block-label { color:#c586c0; font-size:12px; margin:4px 0; }
  .block-title .line { font-size:1.45em; font-weight:700; color:#ffffff; margin-top:14px; }
  .block-body .line { font-size:1em; color:#d4d4d4; }
  .block-caption .line { font-size:.92em; color:#9cdcfe; font-style:italic; }
  .block-reference .line { font-size:.95em; color:#c8c8c8; }
  .block-equation .line { font-family:"Cambria Math", serif; text-align:center; }
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
{% for page in iter_pages(project) %}
<div class="page-header">第 {{ page.page_number }} 页
  <span class="meta">{{ page.image_path }}</span></div>
{% for block in iter_blocks(page, include_empty=True) %}
<div class="block {{ block_style(block).html_class }}" data-type="{{ block.block_type.value }}"
     data-bbox="{{ format_bbox(block.bbox) }}">
  <div class="block-label">{{ block_label(block) }} #{{ block.order }}</div>
{% for line in iter_lines(block) %}
  <div class="line {% if line.proof_status.value == 'auto_flagged' %}flagged
    {% elif line.proof_status.value == 'modified' %}modified
    {% elif line.proof_status.value == 'ok' %}ok{% endif %}"
       data-conf="{{ '%.2f'|format(line.confidence) }}"
       data-bbox="{{ format_bbox(line.bbox) }}">
    {{ line_text(line) }}
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
        env = Environment(loader=BaseLoader(), autoescape=True)
        tmpl = env.from_string(_TEMPLATE)
        html = tmpl.render(
            project=project,
            iter_pages=iter_export_pages,
            iter_blocks=iter_export_blocks,
            iter_lines=iter_export_lines,
            line_text=get_export_text,
            block_label=get_block_label,
            block_style=get_block_style,
            format_bbox=format_bbox,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
