import cv2, numpy as np, json, torch
S=0.75
A=cv2.imread("/home/claude/alive/src.jpg"); H,W=A.shape[:2]
def M(n): return (cv2.imread(f"m_{n}.png",0)>0).astype(np.uint8)
def clean(m,close=9,dil=3,keep=1):
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((close,close),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m)
    if n>2:
        order=sorted(range(1,n),key=lambda i:-st[i][4])[:keep]
        m=np.isin(lab,order).astype(np.uint8)
    # fill holes
    inv=1-m; n,lab,st,_=cv2.connectedComponentsWithStats(inv)
    for i in range(1,n):
        if st[i][4]<4000 and st[i][0]>0 and st[i][1]>0: m[lab==i]=1
    if dil: m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dil+1,2*dil+1)))
    return m
sp={}
sp["splash"]=clean(M("splash"))
sp["lobster"]=clean(M("lobster")|M("glass"))
sp["lobster"][sp["splash"]>0]=0
sp["surfer"]=clean(M("skeleton")|M("board"),close=15)
sp["prop"]=clean(M("prop"))
sp["plane"]=clean(M("plane")); sp["plane"][sp["prop"]>0]=0
sp["gull"]=clean(M("gull"))
sp["cooler"]=clean(M("cooler"))
# letters
hsv=cv2.cvtColor(A,cv2.COLOR_BGR2HSV); Hh,Ss,Vv=cv2.split(hsv)
red=(((Hh<8)|(Hh>172))&(Ss>150)&(Vv>120)).astype(np.uint8)
red=cv2.morphologyEx(red,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
n,lab,st,_=cv2.connectedComponentsWithStats(red)
ids=sorted([i for i in range(1,n) if st[i][4]>8000],key=lambda i:st[i][0])
names=["H1","I","G","H2","R1","O","L1","L2","E","R2"]
assert len(ids)==10,len(ids)
for nm,i in zip(names,ids):
    m=(lab==i).astype(np.uint8)
    # fill holes (counters, white text inside the box)
    x,y,w,h=st[i][:4]; 
    cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    f=np.zeros_like(m); cv2.drawContours(f,cnt,-1,1,-1)
    sp["L_"+nm]=cv2.dilate(f,np.ones((9,9),np.uint8))
# birds: dark pixels in boxes
gray=cv2.cvtColor(A,cv2.COLOR_BGR2GRAY)
for k,(x0,y0,x1,y1) in enumerate([(1540,135,1705,172),(1455,188,1645,228),(1365,243,1535,292)]):
    m=np.zeros((H,W),np.uint8); m[y0:y1,x0:x1]=(gray[y0:y1,x0:x1]<120)
    sp[f"bird{k}"]=cv2.dilate(m,np.ones((7,7),np.uint8))
order=["bird0","bird1","bird2","plane","prop"]+["L_"+n for n in names]+["gull","lobster","splash","cooler","surfer"]
# union + inpaint
U=np.zeros((H,W),np.uint8)
for k in order: U|=sp[k]
Ud=cv2.dilate(U,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
cv2.imwrite("out/union.png",Ud*255)
model=torch.jit.load("big-lama.pt",map_location="cpu").eval()
s2=0.5; w2,h2=int(W*s2)//8*8,int(H*s2)//8*8
img=cv2.resize(A,(w2,h2),interpolation=cv2.INTER_AREA); mk=cv2.resize(Ud,(w2,h2),interpolation=cv2.INTER_NEAREST)
ti=torch.from_numpy(img[:,:,::-1].copy()).permute(2,0,1)[None].float()/255
tm=torch.from_numpy(mk)[None,None].float()
with torch.no_grad(): out=model(ti,tm)
res=(out[0].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)[:,:,::-1]
res=cv2.resize(res,(W,H),interpolation=cv2.INTER_CUBIC)
soft=cv2.GaussianBlur(Ud.astype(np.float32),(0,0),4)[:,:,None]
BG=(A*(1-soft)+res*soft).astype(np.uint8)
ow,oh=int(W*S),int(H*S)
cv2.imwrite("out/bg.jpg",cv2.resize(BG,(ow,oh),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,88])
meta={"w":ow,"h":oh,"sprites":[]}
for k in order:
    m=sp[k]; ys,xs=np.where(m>0); pad=6
    x0,x1,y0,y1=max(0,xs.min()-pad),min(W,xs.max()+pad),max(0,ys.min()-pad),min(H,ys.max()+pad)
    a=cv2.GaussianBlur((m[y0:y1,x0:x1]*255).astype(np.uint8),(0,0),1.2)
    rgba=np.dstack([A[y0:y1,x0:x1],a])
    sw,sh=max(1,round((x1-x0)*S)),max(1,round((y1-y0)*S))
    rgba=cv2.resize(rgba,(sw,sh),interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"out/{k}.webp",rgba,[cv2.IMWRITE_WEBP_QUALITY,90])
    meta["sprites"].append({"id":k,"x":round(x0*S,1),"y":round(y0*S,1),"w":sw,"h":sh})
# region maps (half of output res)
rw,rh=ow//2,oh//2
def poly(pts):
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[np.array(pts,np.int32)],255); return m
water=poly([(500,1080),(700,960),(900,1030),(1200,1130),(1500,1200),(1800,1330),(1932,1420),(1932,2470),(1500,2450),(1050,2450),(950,2350),(930,2150),(880,2000),(830,1900),(800,1700),(780,1500),(560,1350),(500,1200)])
water|=poly([(520,440),(1180,440),(1180,570),(1000,560),(700,500),(520,470)])
water=cv2.GaussianBlur(water,(0,0),25)
regA=np.dstack([clean(M("cloud"),dil=0)*255,clean(M("sun"),dil=0)*255,water])  # BGR -> R=water,G=sun,B=cloud
regB=np.dstack([np.zeros((H,W),np.uint8),clean(M("house"),dil=0)*255,clean(M("lighthouse"),dil=0)*255]) # R=lighthouse,G=house
cv2.imwrite("out/regA.png",cv2.resize(regA,(rw,rh),interpolation=cv2.INTER_AREA))
cv2.imwrite("out/regB.png",cv2.resize(regB,(rw,rh),interpolation=cv2.INTER_AREA))
json.dump(meta,open("out/meta.json","w"),indent=1)
print(meta)
