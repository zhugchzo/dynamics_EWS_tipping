from pypdf import PdfReader, PdfWriter, Transformation

# 读取 PDF
bg = PdfReader('../figures/SFIG16.0.pdf')
ol = PdfReader('../figures/SFIG16.a.pdf')

bg_page = bg.pages[0]
ol_page = ol.pages[0]

# 背景页大小
bg_w = float(bg_page.mediabox.width)
bg_h = float(bg_page.mediabox.height)

# 覆盖页大小
ol_w = float(ol_page.mediabox.width)
ol_h = float(ol_page.mediabox.height)

# 1) 固定缩小比例（比如 50%）
scale = 2.5

ol_w_s = ol_w * scale
ol_h_s = ol_h * scale

# a location
a_rel_x, a_rel_y = 0.02, 0.1  # 相对位置
a_x = bg_w * a_rel_x          # 背景宽度 * 相对比例
a_y = bg_h * a_rel_y          # 背景高度 * 相对比例

# 默认以 overlay 的左下角为锚点
a_t = Transformation().scale(scale).translate(a_x, a_y)

# 3) 合并
bg_page.merge_transformed_page(ol_page, a_t)

# 保存
writer = PdfWriter()
writer.add_page(bg_page)
with open('../figures/SFIG16.pdf', 'wb') as f:
    writer.write(f)