from PIL import Image

def png_to_pdf(png_path, pdf_path):
    # 打开图片并转换为 RGB 模式（PDF 通常不支持 RGBA）
    image = Image.open(png_path)
    rgb_image = image.convert('RGB')
    
    # 直接保存为 PDF
    rgb_image.save(pdf_path, "PDF", resolution=100.0)

# 使用示例
png_to_pdf("./output/similarity/layers_similarity.png", "layers_similarity.pdf")
png_to_pdf("./output/similarity/layers_max_abs_comparison.png", "layers_max_abs_comparison.pdf")