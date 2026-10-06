"""Create a print-ready bead chart from the generated transparent cartoon."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "双人卡通_透明背景_拼豆素材.png"
MAPPING = ROOT / "src" / "app" / "colorSystemMapping.json"
OUT = ROOT / "output" / "双人拼豆图纸_96x72"
PDF_OUT = ROOT / "output" / "pdf" / "双人拼豆图纸_MARD_96x72.pdf"
WIDTH, HEIGHT = 96, 72
TILE = 24
TARGET_COLORS = 24


def oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack(
        [
            0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
        ],
        axis=-1,
    )


def nearest(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def kmeans(samples: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261006)
    centers = np.empty((k, 3), dtype=np.float64)
    centers[0] = samples[int(rng.integers(len(samples)))]
    min_dist = ((samples - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        probabilities = min_dist / min_dist.sum()
        centers[i] = samples[int(rng.choice(len(samples), p=probabilities))]
        min_dist = np.minimum(min_dist, ((samples - centers[i]) ** 2).sum(axis=1))
    for _ in range(40):
        labels = nearest(samples, centers)
        updated = centers.copy()
        for i in range(k):
            if (labels == i).any():
                updated[i] = samples[labels == i].mean(axis=0)
        if np.max(np.abs(updated - centers)) < 0.00005:
            centers = updated
            break
        centers = updated
    return centers, nearest(samples, centers)


def load_grid() -> tuple[np.ndarray, list[dict]]:
    rgba = Image.open(SOURCE).convert("RGBA")
    alpha = rgba.getchannel("A").resize((WIDTH, HEIGHT), Image.Resampling.BOX)
    occupied = np.asarray(alpha) >= 112
    # Resize RGBA directly so edge RGB is weighted by its alpha, while the
    # separate occupancy mask decides whether that bead exists at all.
    grid_rgb = np.asarray(rgba.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS))[:, :, :3]
    samples = oklab(grid_rgb[occupied])
    centers, labels = kmeans(samples, TARGET_COLORS)
    populations = np.bincount(labels, minlength=TARGET_COLORS)

    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    palette = []
    for hex_value, brand_codes in mapping.items():
        if brand_codes.get("MARD"):
            palette.append({"hex": hex_value, **brand_codes})
    palette_lab = oklab(
        np.array([[int(c["hex"][n : n + 2], 16) for n in (1, 3, 5)] for c in palette])
    )

    # Map frequent clusters first and use distinct purchasable MARD colors.
    selected = []
    for cluster in np.argsort(-populations):
        distance = ((palette_lab - centers[cluster]) ** 2).sum(axis=1)
        for palette_index in np.argsort(distance):
            if int(palette_index) not in selected:
                selected.append(int(palette_index))
                break
    selected_lab = palette_lab[selected]
    mapped = nearest(samples, selected_lab)
    grid = np.zeros((HEIGHT, WIDTH), dtype=np.int16)
    grid[occupied] = mapped + 1

    # A few tiny isolated light beads are antialiasing noise. Retain all dark
    # one-cell marks because they carry eyes, glasses and hair details.
    lightness = selected_lab[:, 0]
    for row in range(1, HEIGHT - 1):
        for col in range(1, WIDTH - 1):
            code = int(grid[row, col])
            if not code or lightness[code - 1] < 0.45:
                continue
            neighbors = [
                int(grid[rr, cc])
                for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                if grid[rr, cc]
            ]
            if len(neighbors) < 3 or code in neighbors:
                continue
            replacement = Counter(neighbors).most_common(1)[0][0]
            if np.linalg.norm(selected_lab[code - 1] - selected_lab[replacement - 1]) < 0.075:
                grid[row, col] = replacement

    counts = Counter(int(v) for v in grid.flat if v)
    used = sorted(counts, key=lambda i: (-counts[i], palette[selected[i - 1]]["MARD"]))
    # Make legend numbers compact and ordered by bead count.
    renumber = {old: new for new, old in enumerate(used, start=1)}
    grid = np.vectorize(lambda v: renumber.get(int(v), 0), otypes=[np.int16])(grid)
    colors_used = []
    for old in used:
        item = palette[selected[old - 1]].copy()
        item["count"] = counts[old]
        item["number"] = renumber[old]
        colors_used.append(item)
    return grid, colors_used


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", size)


def draw_png(grid: np.ndarray, palette: list[dict], *, numbered: bool) -> Path:
    cell = 24 if numbered else 12
    left, top, right, bottom = (75, 85, 20, 40) if numbered else (48, 55, 12, 26)
    image = Image.new("RGB", (left + WIDTH * cell + right, top + HEIGHT * cell + bottom), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    colors_by_number = {entry["number"]: entry for entry in palette}
    draw.text((left, 18), "双人拼豆图纸  |  96 × 72 格  |  MARD 色号", fill="#17212F", font=font(27 if numbered else 17))
    small = font(12 if numbered else 9)
    cell_font = font(12)
    for row in range(HEIGHT):
        for col in range(WIDTH):
            number = int(grid[row, col])
            x, y = left + col * cell, top + row * cell
            fill = colors_by_number[number]["hex"] if number else "#FFFFFF"
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=fill)
            if numbered and number:
                rgb = tuple(int(fill[i : i + 2], 16) for i in (1, 3, 5))
                l = float(oklab(np.array(rgb))[0])
                ink = "#FFFFFF" if l < 0.56 else "#20252A"
                draw.text((x + cell / 2, y + cell / 2), f"{number:02d}", fill=ink, font=cell_font, anchor="mm")
    line = "#C6CFD8" if numbered else "#DAE0E5"
    for col in range(WIDTH + 1):
        x = left + col * cell
        if numbered or col % TILE == 0:
            draw.line((x, top, x, top + HEIGHT * cell), fill="#4D6177" if col % TILE == 0 else line, width=2 if col % TILE == 0 else 1)
        if numbered and col % 5 == 0 and col < WIDTH:
            draw.text((x + 3, top - 23), str(col + 1), fill="#33465A", font=small)
    for row in range(HEIGHT + 1):
        y = top + row * cell
        if numbered or row % TILE == 0:
            draw.line((left, y, left + WIDTH * cell, y), fill="#4D6177" if row % TILE == 0 else line, width=2 if row % TILE == 0 else 1)
        if numbered and row % 5 == 0 and row < HEIGHT:
            draw.text((left - 38, y + 4), str(row + 1), fill="#33465A", font=small)
    draw.text((left, top + HEIGHT * cell + 11), "空白格不放豆；粗线每 24 格分区。", fill="#33465A", font=small)
    path = OUT / ("总图_格号版.png" if numbered else "总图_色块预览.png")
    image.save(path, optimize=True)
    return path


def save_csv(grid: np.ndarray, palette: list[dict]) -> None:
    with (OUT / "用量与色号.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["图纸编号", "MARD", "COCO", "漫漫", "盼盼", "咪小窝", "HEX", "颗数"])
        for item in palette:
            writer.writerow([f'{item["number"]:02d}', item["MARD"], item["COCO"], item["漫漫"], item["盼盼"], item["咪小窝"], item["hex"], item["count"]])
        writer.writerow(["合计", "", "", "", "", "", "", int(np.count_nonzero(grid))])
    with (OUT / "逐格数据.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["行/列", *range(1, WIDTH + 1)])
        for row in range(HEIGHT):
            writer.writerow([row + 1, *[f"{int(v):02d}" if v else "" for v in grid[row]]])


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
    pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
    pdfmetrics.registerFont(TTFont("ArialBold", "C:/Windows/Fonts/arialbd.ttf"))


def pdf_header(pdf: canvas.Canvas, title: str, subtitle: str) -> None:
    pw, ph = A4
    pdf.setFillColor(colors.HexColor("#182536"))
    pdf.setFont("SimHei", 17)
    pdf.drawString(19 * mm, ph - 21 * mm, title)
    pdf.setFillColor(colors.HexColor("#66788B"))
    pdf.setFont("SimHei", 9)
    pdf.drawString(19 * mm, ph - 29 * mm, subtitle)
    pdf.setStrokeColor(colors.HexColor("#D5DDE5"))
    pdf.line(19 * mm, ph - 34 * mm, pw - 19 * mm, ph - 34 * mm)


def save_pdf(grid: np.ndarray, palette: list[dict], preview_path: Path) -> None:
    register_fonts()
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(PDF_OUT), pagesize=A4, pageCompression=1)
    pw, ph = A4
    bead_count = int(np.count_nonzero(grid))

    pdf_header(pdf, "双人拼豆图纸", "MARD 色号  |  96 × 72 格  |  透明背景  |  总图与分区打印版")
    pdf.drawImage(ImageReader(str(preview_path)), 19 * mm, ph - 190 * mm, width=172 * mm, height=133 * mm, preserveAspectRatio=True, anchor="c")
    pdf.setFont("SimHei", 10)
    pdf.setFillColor(colors.HexColor("#25364A"))
    cover_lines = [
        f"实际拼豆：{bead_count:,} 颗　　颜色：{len(palette)} 种　　空白格：不放豆",
        "按 5 mm 拼豆间距估算，成品最大外框约 48 × 36 cm。",
        "总图用于对照；后续 12 页按 24 × 24 格分区，格内数字对应色号表。",
        "分区页每格 7 mm，便于看清编号；它不是拼板的 1:1 尺寸。",
    ]
    for i, line in enumerate(cover_lines):
        pdf.drawString(20 * mm, ph - (202 + i * 10) * mm, line)
    pdf.setFont("SimHei", 8)
    pdf.setFillColor(colors.HexColor("#627386"))
    pdf.drawString(20 * mm, 19 * mm, "制作提示：先按分区摆放，再对照总图确认面部与眼镜细节。")
    pdf.showPage()

    pdf_header(pdf, "色号与用量", "图纸编号写在每个有豆的格子中；各品牌色号来自项目现有映射表。")
    xs = [19, 31, 55, 78, 103, 128, 154, 181]
    headers = ["编号", "色块", "MARD", "COCO", "漫漫", "盼盼", "咪小窝", "颗数"]
    y_top = ph - 45 * mm
    pdf.setFillColor(colors.HexColor("#EFF3F7"))
    pdf.rect(19 * mm, y_top - 7 * mm, 172 * mm, 8 * mm, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#26374B"))
    pdf.setFont("SimHei", 8)
    for x, header in zip(xs, headers):
        pdf.drawString(x * mm, y_top - 4.3 * mm, header)
    for index, item in enumerate(palette):
        y = y_top - 3 * mm - (index + 1) * 8.15 * mm
        if index % 2:
            pdf.setFillColor(colors.HexColor("#F8FAFC"))
            pdf.rect(19 * mm, y - 5 * mm, 172 * mm, 8.15 * mm, fill=1, stroke=0)
        values = [f'{item["number"]:02d}', "", item["MARD"], item["COCO"], item["漫漫"], item["盼盼"], item["咪小窝"], str(item["count"])]
        pdf.setFillColor(colors.HexColor("#26374B"))
        pdf.setFont("Arial", 8)
        for j, value in enumerate(values):
            if j == 4:
                pdf.setFont("SimHei", 8)
            pdf.drawString(xs[j] * mm, y, value)
            if j == 4:
                pdf.setFont("Arial", 8)
        pdf.setFillColor(colors.HexColor(item["hex"]))
        pdf.setStrokeColor(colors.HexColor("#9BA7B4"))
        pdf.rect(31 * mm, y - 1.4 * mm, 12 * mm, 5.2 * mm, fill=1, stroke=1)
    pdf.setFont("SimHei", 9)
    pdf.setFillColor(colors.HexColor("#26374B"))
    pdf.drawString(20 * mm, 40 * mm, f"合计 {bead_count:,} 颗；建议购买时按每种颜色用量预留少量余量。")
    pdf.drawString(20 * mm, 31 * mm, "不同批次实物颜色可能与屏幕显示略有差异。")
    pdf.showPage()

    lookup = {entry["number"]: entry for entry in palette}
    cell = 7 * mm
    x0 = 21 * mm
    y0 = 54 * mm
    for tile_row in range(HEIGHT // TILE):
        for tile_col in range(WIDTH // TILE):
            name = f"{chr(ord('A') + tile_col)}{tile_row + 1}"
            c0, r0 = tile_col * TILE, tile_row * TILE
            pdf_header(pdf, f"分区 {name}", f"列 {c0 + 1}-{c0 + TILE}　行 {r0 + 1}-{r0 + TILE}　|　打印 100%　|　每格 7 mm")
            for row in range(TILE):
                for col in range(TILE):
                    number = int(grid[r0 + row, c0 + col])
                    x = x0 + col * cell
                    y = y0 + (TILE - row - 1) * cell
                    if number:
                        hex_value = lookup[number]["hex"]
                        pdf.setFillColor(colors.HexColor(hex_value))
                        pdf.rect(x, y, cell, cell, fill=1, stroke=0)
                        l = float(oklab(np.array([int(hex_value[i : i + 2], 16) for i in (1, 3, 5)]))[0])
                        pdf.setFillColor(colors.white if l < 0.56 else colors.HexColor("#17212F"))
                        pdf.setFont("ArialBold", 7.1)
                        pdf.drawCentredString(x + cell / 2, y + cell / 2 - 2.3, f"{number:02d}")
            for i in range(TILE + 1):
                pdf.setStrokeColor(colors.HexColor("#697D91" if i % 5 == 0 else "#C8D1DB"))
                pdf.setLineWidth(0.55 if i % 5 == 0 else 0.22)
                xx, yy = x0 + i * cell, y0 + i * cell
                pdf.line(xx, y0, xx, y0 + TILE * cell)
                pdf.line(x0, yy, x0 + TILE * cell, yy)
            pdf.setFillColor(colors.HexColor("#44586D"))
            pdf.setFont("Arial", 7)
            for i in range(TILE):
                pdf.drawCentredString(x0 + (i + 0.5) * cell, y0 + TILE * cell + 3 * mm, str(c0 + i + 1))
                pdf.drawRightString(x0 - 2 * mm, y0 + (TILE - i - 0.5) * cell - 2.2, str(r0 + i + 1))
            tile_count = int(np.count_nonzero(grid[r0 : r0 + TILE, c0 : c0 + TILE]))
            pdf.setFont("SimHei", 9)
            pdf.drawString(21 * mm, 38 * mm, f"本区 {tile_count} 颗；白色空格不放豆。数字在第 2 页色号表中查找。")
            pdf.setFont("Arial", 8)
            pdf.drawRightString(pw - 20 * mm, 20 * mm, f"{name}  /  {tile_row * (WIDTH // TILE) + tile_col + 1} of 12")
            pdf.showPage()
    pdf.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid, palette = load_grid()
    preview = draw_png(grid, palette, numbered=False)
    draw_png(grid, palette, numbered=True)
    save_csv(grid, palette)
    save_pdf(grid, palette, preview)
    print(f"grid={WIDTH}x{HEIGHT} beads={np.count_nonzero(grid)} colors={len(palette)}")
    print(f"pdf={PDF_OUT}")
    print(f"files={list(OUT.iterdir())}")


if __name__ == "__main__":
    main()
