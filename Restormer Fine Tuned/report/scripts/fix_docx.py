import os,sys
from docx import Document
from docx.shared import Pt, Mm, Emu
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
src,dst=sys.argv[1],sys.argv[2]
d=Document(src)
for s in d.sections:
    s.orientation=WD_ORIENT.LANDSCAPE; s.page_width=Mm(297); s.page_height=Mm(210)
    s.left_margin=s.right_margin=Mm(14); s.top_margin=s.bottom_margin=Mm(14)
maxw=Mm(297-28); maxh=Mm(210-40)
for sh in d.inline_shapes:
    w,h=sh.width,sh.height; k=min(maxw/w, maxh/h)
    sh.width=int(w*k); sh.height=int(h*k)
for st in d.styles:
    try:
        if st.type==1 and st.font is not None and st.name in('Normal','Body Text','First Paragraph','Compact'): st.font.size=Pt(10); st.font.name='Arial'
    except Exception: pass
for t in d.tables:
    tblPr=t._tbl.tblPr
    b=OxmlElement('w:tblBorders')
    for e in ('top','left','bottom','right','insideH','insideV'):
        x=OxmlElement(f'w:{e}'); x.set(qn('w:val'),'single'); x.set(qn('w:sz'),'4'); x.set(qn('w:space'),'0'); x.set(qn('w:color'),'BBBBBB'); b.append(x)
    tblPr.append(b)
    # fixed layout, full page width, content-weighted column widths
    for e in tblPr.findall(qn('w:tblW')): tblPr.remove(e)
    for e in tblPr.findall(qn('w:tblLayout')): tblPr.remove(e)
    W=OxmlElement('w:tblW'); W.set(qn('w:type'),'dxa'); total=int((297-28)/25.4*1440); W.set(qn('w:w'),str(total)); tblPr.append(W)
    lay=OxmlElement('w:tblLayout'); lay.set(qn('w:type'),'fixed'); tblPr.append(lay)
    nc=len(t.columns); lens=[1]*nc
    for row in t.rows:
        for i,c in enumerate(row.cells[:nc]):
            words=c.text.split(); longest=max([len(w) for w in words]+[1])
            lens[i]=max(lens[i],min(len(c.text),60)**0.75, longest*1.15)
    ssum=sum(lens); widths=[int(total*l/ssum) for l in lens]
    grid=t._tbl.find(qn('w:tblGrid'))
    if grid is not None:
        for i,gc in enumerate(grid.findall(qn('w:gridCol'))):
            if i<nc: gc.set(qn('w:w'),str(widths[i]))
    for row in t.rows:
        for i,c in enumerate(row.cells[:nc]):
            tcPr=c._tc.get_or_add_tcPr()
            for e in tcPr.findall(qn('w:tcW')): tcPr.remove(e)
            tw=OxmlElement('w:tcW'); tw.set(qn('w:type'),'dxa'); tw.set(qn('w:w'),str(widths[i])); tcPr.insert(0,tw)
    ncol=len(t.columns); fs=Pt(8.5 if ncol<=7 else (7.5 if ncol<=10 else 6.8))
    for ri,row in enumerate(t.rows):
        for c in row.cells:
            if ri==0:
                tcPr=c._tc.get_or_add_tcPr(); sh=OxmlElement('w:shd'); sh.set(qn('w:val'),'clear'); sh.set(qn('w:color'),'auto'); sh.set(qn('w:fill'),'E8EEF7'); tcPr.append(sh)
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(1); p.paragraph_format.space_before=Pt(1)
                for r in p.runs:
                    r.font.size=fs
                    if ri==0: r.font.bold=True
d.save(dst)
