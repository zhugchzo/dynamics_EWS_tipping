from pypdf import PdfReader, PdfWriter
from pypdf._page import PageObject


def merge_horizontal(page1, page2):
    """两个 PDF 页面横向拼接，不缩放"""
    
    w1, h1 = float(page1.mediabox.width), float(page1.mediabox.height)
    w2, h2 = float(page2.mediabox.width), float(page2.mediabox.height)

    width = w1 + w2
    height = max(h1, h2)

    new_page = PageObject.create_blank_page(
        width=width,
        height=height
    )

    # 左
    new_page.merge_translated_page(
        page1,
        tx=0,
        ty=0
    )

    # 右
    new_page.merge_translated_page(
        page2,
        tx=w1,
        ty=0
    )

    return new_page


# =========================
# 读取四个 PDF
# =========================

page1 = PdfReader('../figures/SFIG4.1.pdf').pages[0]
page2 = PdfReader('../figures/SFIG4.2.pdf').pages[0]
page3 = PdfReader('../figures/SFIG4.3.pdf').pages[0]
page4 = PdfReader('../figures/SFIG4.4.pdf').pages[0]


# =========================
# 第一行：14.1 + 14.2
# 第二行：14.3 + 14.4
# =========================

row1 = merge_horizontal(page1, page2)
row2 = merge_horizontal(page3, page4)


# =========================
# 两行纵向拼接
# =========================

w1, h1 = float(row1.mediabox.width), float(row1.mediabox.height)
w2, h2 = float(row2.mediabox.width), float(row2.mediabox.height)

width = max(w1, w2)
height = h1 + h2

final_page = PageObject.create_blank_page(
    width=width,
    height=height
)

# 第一行放上面
final_page.merge_translated_page(
    row1,
    tx=0,
    ty=h2
)

# 第二行放下面
final_page.merge_translated_page(
    row2,
    tx=0,
    ty=0
)


# =========================
# 保存
# =========================

writer = PdfWriter()
writer.add_page(final_page)

with open('../figures/SFIG4.pdf', 'wb') as f:
    writer.write(f)
