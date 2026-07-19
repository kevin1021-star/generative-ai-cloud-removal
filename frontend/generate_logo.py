from PIL import Image, ImageDraw, ImageFont

def generate_isro_logo():
    # Create a transparent image (300 x 150)
    img = Image.new("RGBA", (400, 150), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # 1. Draw the stylized ISRO orange arrow
    # The arrow is composed of a triangle and a curved body
    # Arrow color: ISRO Orange #FF6600
    orange = (255, 102, 0, 255)
    
    # Coordinates for a clean minimalist space arrow
    # Back wing
    draw.polygon([(40, 110), (70, 30), (100, 110), (70, 95)], fill=orange)
    # Highlight lines
    draw.line([(70, 30), (70, 95)], fill=(255, 255, 255, 255), width=2)

    # 2. Draw the ISRO text
    # Let's try to use standard system fonts or default draw.text
    # We will draw "ISRO" in a clean, bold blue color (#0A5C9E)
    blue = (10, 92, 158, 255)
    
    # Since we might not have the exact font file, we can draw the letters using lines/polygons
    # to guarantee they are crisp and vector-like on any OS!
    
    def draw_I(offset):
        draw.rectangle([offset, 40, offset + 12, 110], fill=blue)

    def draw_S(offset):
        # Draw S with blocks
        draw.rectangle([offset, 40, offset + 40, 52], fill=blue) # Top
        draw.rectangle([offset, 40, offset + 12, 75], fill=blue) # Left top
        draw.rectangle([offset, 65, offset + 40, 77], fill=blue) # Mid
        draw.rectangle([offset + 28, 65, offset + 40, 110], fill=blue) # Right bottom
        draw.rectangle([offset, 98, offset + 40, 110], fill=blue) # Bottom

    def draw_R(offset):
        # Draw R with blocks
        draw.rectangle([offset, 40, offset + 12, 110], fill=blue) # Left vertical stem
        draw.rectangle([offset, 40, offset + 40, 52], fill=blue) # Top horizontal
        draw.rectangle([offset + 28, 40, offset + 40, 75], fill=blue) # Right loop vertical
        draw.rectangle([offset, 70, offset + 40, 82], fill=blue) # Middle horizontal
        # Diagonal leg
        draw.polygon([(offset + 20, 75), (offset + 38, 110), (offset + 48, 110), (offset + 28, 75)], fill=blue)

    def draw_O(offset):
        # Draw O with blocks
        draw.rectangle([offset, 40, offset + 40, 110], outline=blue, width=12)

    # Draw the letters with offsets
    draw_I(140)
    draw_S(175)
    draw_R(235)
    draw_O(295)

    # Save to the public folder
    img.save("public/isro_logo.png")
    print("ISRO logo generated successfully!")

if __name__ == "__main__":
    generate_isro_logo()
