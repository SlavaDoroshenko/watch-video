import numpy as np, subprocess, random
from PIL import Image, ImageDraw, ImageFont
W,H,FPS = 540,1170,60
random.seed(1)
# content canvas
CH = 6000
content = Image.new("RGB",(W,CH),(250,250,252))
d = ImageDraw.Draw(content)
f = ImageFont.load_default(size=22)
for i in range(CH//120):
    y=i*120
    col=(random.randint(60,220),random.randint(60,220),random.randint(60,220))
    d.rectangle([16,y+16,104,y+104],fill=col)
    d.text((130,y+30),f"Item {i+1}  lorem ipsum dolor",fill=(20,20,20),font=f)
    d.text((130,y+62),f"subtitle {random.randint(1000,9999)}",fill=(120,120,120),font=f)
    d.line([0,y+119,W,y+119],fill=(225,225,225))
content=np.array(content)
TOP,BOT=60,90
def frame(offset):
    img=np.full((H,W,3),255,np.uint8)
    off=int(round(offset)); view=H-TOP-BOT
    img[TOP:TOP+view]=content[off:off+view]
    pil=Image.fromarray(img); dd=ImageDraw.Draw(pil)
    dd.rectangle([0,0,W,TOP],fill=(30,30,30)); dd.text((20,18),"9:41   ●●●  wifi  batt",fill="white",font=f)
    dd.rectangle([0,H-BOT,W,H],fill=(245,245,245)); dd.line([0,H-BOT,W,H-BOT],fill=(200,200,200))
    for k in range(4): dd.ellipse([60+k*120,H-70,90+k*120,H-40],fill=(90,90,90))
    return np.array(pil)
offsets=[]; off=0.0
for n in range(450):
    if n<30: pass
    elif n<150:
        off+=12
        if n==90: off-=60           # reversal: content jumps back 60px
    elif n<160: pass               # freeze 10 frames
    elif n==160: off+=80           # catch-up jump
    elif n<220: off+=12
    elif n<260: off+=12*(260-n)/40 # decelerate
    elif n<300: pass
    elif n<420: off-=8             # smooth scroll up
    offsets.append(off)
p=subprocess.Popen(["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
   "-c:v","libx264","-preset","fast","-crf","20","-pix_fmt","yuv420p","-g","60","test_cfr.mp4"],stdin=subprocess.PIPE)
for o in offsets: p.stdin.write(frame(o).tobytes())
p.stdin.close(); p.wait()
print("truth: reversal at frame 90 (t=1.500), freeze frames 150-159 (t=2.500-2.650), jump at 160 (t=2.667), scroll-up 300-419 smooth")
