"""HTML 导出：Jinja2 模板，嵌入 Export IR 元数据。"""

from jinja2 import Environment, BaseLoader

from app.export.base import ExporterBase
from app.export.ir_builder import build_export_ir
from app.export.rendering import bbox_attr, element_label, element_lines, KIND_HTML_CLASS
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
<h1>{{ document.project.name }}</h1>
<p class="meta">共 {{ document.project.page_count }} 页，{{ document.project.summary.total_lines }} 行</p>
{% for page in document.pages %}
<div class="page-header">第 {{ page.page_number }} 页
  <span class="meta">{{ page.source_image }}</span></div>
{% for element in page.elements %}
<div class="block {{ html_class(element.kind) }}" data-type="{{ element.kind }}"
    data-bbox="{{ bbox_attr(element.bbox) }}">
  <div class="block-label">{{ element_label(element) }} #{{ element.order }}</div>
{% for text in element_lines(element) %}
  <div class="line {% if element.proof.status == 'auto_flagged' %}flagged
    {% elif element.proof.status == 'modified' %}modified
    {% elif element.proof.status == 'ok' %}ok{% endif %}"
      data-conf="{{ '%.2f'|format(element.proof.confidence or 0) }}">
    {{ text }}
    <span class="meta">{{ '%.0f'|format((element.proof.confidence or 0)*100) }}%</span>
  </div>
{% endfor %}
{% if element.fallback and element.fallback.used %}
  <div class="meta">fallback={{ element.fallback.mode }} reason={{ element.fallback.reason }}</div>
{% endif %}
</div>
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
            element_label=element_label,
            element_lines=element_lines,
            bbox_attr=bbox_attr,
            html_class=lambda kind: KIND_HTML_CLASS.get(kind, "block-unknown"),
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
