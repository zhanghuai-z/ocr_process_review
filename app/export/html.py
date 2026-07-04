"""HTML 导出：Jinja2 模板，嵌入 Export IR 元数据。"""

from jinja2 import Environment, BaseLoader

from app.export.base import ExporterBase
from app.export.ir_builder import build_export_ir
from app.export.rendering import rich_reflow_pages
from app.models import OcrProject

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{{ document.project.name }}</title>
<style>
  body { font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif;
         background:#1e1e1e; color:#d4d4d4; margin:40px; }
  h1   { color:#9cdcfe; }
  h2   { color:#ffffff; margin:14px 0 8px 16px; }
  .page-header { color:#569cd6; border-bottom:1px solid #3c3c3c;
                 padding:8px 0; margin-top:32px; }
  .block  { margin:12px 0 12px 16px; }
  .block-label { color:#c586c0; font-size:12px; margin:4px 0; }
  .block-body .line { font-size:1em; color:#d4d4d4; }
  .block-caption .line { font-size:.92em; color:#9cdcfe; font-style:italic; }
  .block-reference .line { font-size:.95em; color:#c8c8c8; }
  .block-equation .line { font-family:"Cambria Math", serif; text-align:center; }
  .line   { margin:2px 0; line-height:1.8; }
  .auto_flagged, .flagged { color:#f44336; }
  .modified{ color:#ffa726; }
  .ok      { color:#4caf50; }
  .meta   { font-size:11px; color:#666; margin-left:8px; }
</style>
</head>
<body>
<h1>{{ document.project.name }}</h1>
<p class="meta">共 {{ document.project.page_count }} 页，{{ document.project.summary.total_lines }} 行</p>
{% for page, blocks in rich_pages %}
<div class="page-header">第 {{ page.page_number }} 页
  <span class="meta">{{ page.source_image }}</span></div>
{% for reflow in blocks %}
{% if reflow.role == "heading" %}
{% for text in reflow.lines %}
<h2 class="block {{ reflow.html_class }}" data-type="{{ reflow.kind }}">{{ text }}</h2>
{% endfor %}
{% else %}
<div class="block {{ reflow.html_class }}" data-type="{{ reflow.kind }}" data-role="{{ reflow.role }}">
  <div class="block-label">{{ reflow.label }} #{{ reflow.order }}</div>
  {% for text in reflow.lines %}
  <p class="line {{ reflow.proof_status }}">{{ text }}</p>
  {% endfor %}
</div>
{% endif %}
{% endfor %}
{% endfor %}
</body>
</html>
"""


class HtmlExporter(ExporterBase):

    def export(self, project: OcrProject, out_path: str) -> None:
        document = build_export_ir(project, "html")
        env = Environment(loader=BaseLoader(), autoescape=True)
        tmpl = env.from_string(_TEMPLATE)
        html = tmpl.render(
            document=document,
            rich_pages=rich_reflow_pages(document),
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
