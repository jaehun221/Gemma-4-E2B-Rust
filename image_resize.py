from PIL import Image
im = Image.open("gemma4_image_sample.png")
bg = Image.new("RGB", im.size, (255, 255, 255))
bg.paste(im, mask=im.getchannel("A"))
bg.resize((480, 288), Image.LANCZOS).save("test_480x288.png")