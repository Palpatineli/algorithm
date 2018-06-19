from os.path import splitext, exists
from os import remove
import subprocess as sp
import altair as alt

def save_chart(chart: alt.Chart, file_path: str) -> None:
    base_name, ext = splitext(file_path)
    json_file, svg_file = base_name + ".json", base_name + ".svg"
    json_exists = True if exists(json_file) else False
    chart.savechart(json_file)
    with open(svg_file, 'wb') as svg_fp:
        sp.call(["vl2svg", json_file], stdout=svg_fp)
    if not json_exists:
        remove(json_file)
