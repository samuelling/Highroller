import torch, sys, json, numpy as np, cv2
from PIL import Image
from transformers import SamModel, SamProcessor
name="facebook/sam-vit-large"
proc=SamProcessor.from_pretrained(name); model=SamModel.from_pretrained(name).eval()
E=torch.load("emb.pt")
im=Image.open("/home/claude/alive/src.jpg").convert("RGB"); A=np.array(im)
P=json.load(open("prompts.json"))
only=sys.argv[1:] 
for k,v in P.items():
    if only and k not in only: continue
    pts=v.get("pos",[])+v.get("neg",[]); lab=[1]*len(v.get("pos",[]))+[0]*len(v.get("neg",[]))
    kw={}
    if pts: kw["input_points"]=[[pts]]; kw["input_labels"]=[[lab]]
    if "box" in v: kw["input_boxes"]=[[v["box"]]]
    inp=proc(im,return_tensors="pt",**kw)
    args=dict(image_embeddings=E["emb"],multimask_output=v.get("multi",False))
    for kk in ("input_points","input_labels","input_boxes"):
        if kk in inp: args[kk]=inp[kk]
    with torch.no_grad(): out=model(**args)
    m=proc.image_processor.post_process_masks(out.pred_masks,inp["original_sizes"],inp["reshaped_input_sizes"])[0][0]
    sc=out.iou_scores[0,0]; i=int(sc.argmax()) if v.get("multi",False) else 0
    if "pick" in v: i=v["pick"]
    mk=m[i].numpy().astype(np.uint8)*255
    cv2.imwrite(f"m_{k}.png",mk)
    ov=A.copy(); ov[mk>0]=(0.45*ov[mk>0]+0.55*np.array([0,255,255])).astype(np.uint8)
    for (x,y),l in zip(pts,lab): cv2.circle(ov,(x,y),10,(0,255,0) if l else (255,0,0),-1)
    if "box" in v:
        x0,y0,x1,y1=v["box"]; cv2.rectangle(ov,(x0,y0),(x1,y1),(255,0,255),3)
        pad=40; ov=ov[max(0,y0-pad):y1+pad,max(0,x0-pad):x1+pad]
    h,w=ov.shape[:2]; s=min(1,700/max(h,w))
    Image.fromarray(ov).resize((int(w*s),int(h*s))).save(f"ov_{k}.jpg")
    print(k,[round(float(x),3) for x in sc.flatten()],int((mk>0).sum()))
